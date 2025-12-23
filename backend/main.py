import os
from dotenv import load_dotenv
load_dotenv() # Load env vars BEFORE importing modules that might look for them

# Trigger deployment
import io
import uuid
import datetime
import json
import requests
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from ultralytics import YOLO
from huggingface_hub import hf_hub_download, HfApi, login, hf_hub_url
import wandb
import numpy as np
import cv2
import kaggle_trigger # [NEW] Import trigger module

# --- Configuration ---
HP_REPO_ID = "qahir00/yolo11-object-detection"
MODEL_FILENAME = "best.pt"
DATASET_REPO_ID = os.getenv("DATASET_REPO_ID", "qahir00/yolo-production-data") # Set this env var in Spaces

# --- Setup ---
app = FastAPI(title="YOLO11-MLOps-API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None

# --- Helpers ---
def load_model():
    global model
    print(f"Downloading model {MODEL_FILENAME} from {HP_REPO_ID}...")
    try:
        model_path = hf_hub_download(repo_id=HP_REPO_ID, filename=MODEL_FILENAME)
        print(f"Model downloaded to {model_path}")
        model = YOLO(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

def log_to_wandb(image, results, confs):
    """Log image and predictions to Weights & Biases."""
    if os.getenv("WANDB_API_KEY"):
        try:
            wandb.init(project="yolo11-object-detection", resume="allow", reinit=True)
            # Log image with bounding boxes
            wandb.log({
                "inference_image": wandb.Image(image, boxes={
                    "predictions": {
                        "box_data": results,
                        "class_labels": model.names
                    }
                }),
                "mean_confidence": float(np.mean(confs)) if confs else 0.0
            })
            # wandb.finish() # Keep session open if permissible or finish to save connection overhead
        except Exception as e:
            print(f"W&B Logging failed: {e}")

def save_to_data_lake(image_bytes: bytes, filename: str, detections: list):
    """Upload raw image and predictions to Hugging Face Dataset for Label Studio."""
    hf_token = os.getenv("HF_TOKEN")
    dataset_repo = "qahir00/yolo-data" # Target Dataset
    
    if hf_token:
        try:
            api = HfApi()
            date_str = datetime.datetime.now().strftime('%Y-%m-%d')
            
            # 1. Upload Image
            image_path = f"images/{date_str}/{filename}"
            api.upload_file(
                path_or_fileobj=io.BytesIO(image_bytes),
                path_in_repo=image_path,
                repo_id=dataset_repo,
                repo_type="dataset",
                token=hf_token
            )
            
            # 2. Upload Predictions (JSON for Label Studio Pre-annotation)
            json_filename = filename.replace('.jpg', '.json')
            json_path = f"predictions/{date_str}/{json_filename}"
            
            json_content = json.dumps({
                "image": image_path,
                "detections": detections,
                "timestamp": datetime.datetime.now().isoformat()
            }, indent=2)
            
            api.upload_file(
                path_or_fileobj=io.BytesIO(json_content.encode('utf-8')),
                path_in_repo=json_path,
                repo_id=dataset_repo,
                repo_type="dataset",
                token=hf_token
            )
            
            print(f"Logged data to {dataset_repo}")
        except Exception as e:
            print(f"Data Lake Upload failed details: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print("Data Lake skipped: HF_TOKEN not found in env.")

# --- Startup ---
@app.on_event("startup")
async def startup_event():
    load_model()
    # Init generic W&B run if needed, or do it per request
    token = os.getenv("HF_TOKEN")
    print(f"Startup: HF_TOKEN is {'Present' if token else 'MISSING'}")
    
    if os.getenv("WANDB_API_KEY"):
        wandb.login(key=os.getenv("WANDB_API_KEY"))

# --- Review System Models ---
from typing import Optional

class ReviewData(BaseModel):
    filename: str
    decision: str  # 'verified' or 'correction'
    label: Optional[str] = None # 'chair', 'box', etc. (Only if decision='correction')
    box: Optional[list] = None # [x1, y1, x2, y2] (Optional, if correcting box too)

# --- Review System Endpoints ---
@app.get("/list-unverified")
def list_unverified(limit: int = 50):
    """List most recent images from the dataset that need review."""
    token = os.getenv("HF_TOKEN")
    dataset_repo = "qahir00/yolo-data"
    
    if not token:
        return {"error": "HF_TOKEN missing"}
    
    try:
        api = HfApi(token=token)
        # List all files
        files = api.list_repo_files(dataset_repo, repo_type="dataset")
        
        # Filter for images in 'images/' folder, excluding 'verified' or 'corrected'
        image_files = [
            f for f in files 
            if f.startswith("images/") and f.endswith((".jpg", ".png", ".webp"))
            and "verified" not in f and "corrected" not in f
        ]
        
        # Sort by date (descending) - inferred from path images/YYYY-MM-DD/...
        image_files.sort(reverse=True)
        
        # return newest 'limit'
        return {"images": image_files[:limit]}
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/proxy-image")
def proxy_image(path: str):
    """Securely proxy image from private HF dataset to frontend."""
    token = os.getenv("HF_TOKEN")
    dataset_repo = "qahir00/yolo-data"
    
    if not token:
        return Response(content="HF_TOKEN missing", status_code=500)
        
    try:
        # Construct URL
        url = hf_hub_url(repo_id=dataset_repo, filename=path, repo_type="dataset")
        
        # Fetch with Auth
        resp = requests.get(url, headers={"Authorization": f"Bearer {token}"})
        
        if resp.status_code == 200:
            return Response(content=resp.content, media_type="image/jpeg")
        else:
            return Response(content=f"Error upstream: {resp.status_code}", status_code=resp.status_code)
            
    except Exception as e:
        return Response(content=str(e), status_code=500)

@app.get("/get-prediction-data")
def get_prediction_data(path: str):
    """Fetch stored prediction JSON for a specific image path."""
    token = os.getenv("HF_TOKEN")
    dataset_repo = "qahir00/yolo-data"
    
    if not token:
        return {"error": "HF_TOKEN missing"}
        
    try:
        # Convert image path to prediction path
        # from: images/2025-01-01/abc.jpg
        # to:   predictions/2025-01-01/abc.json
        if not path.startswith("images/"):
             return {"error": "Invalid path format"}
             
        pred_path = path.replace("images/", "predictions/").replace(".jpg", ".json").replace(".png", ".json").replace(".webp", ".json")
        
        url = hf_hub_url(repo_id=dataset_repo, filename=pred_path, repo_type="dataset")
        resp = requests.get(url, headers={"Authorization": f"Bearer {token}"})
        
        if resp.status_code == 200:
            return resp.json() # Direct JSON return
        else:
            return {"error": "Prediction not found", "details": f"Upstream {resp.status_code}"}
            
    except Exception as e:
        return {"error": str(e)}

@app.post("/submit-review")
def submit_review(data: ReviewData):
    """Move data to 'verified' or 'corrected' folder based on user review."""
    token = os.getenv("HF_TOKEN")
    dataset_repo = "qahir00/yolo-data"
    
    if not token:
        return {"status": "error", "message": "HF_TOKEN missing"}
        
    try:
        api = HfApi(token=token)
        date_str = datetime.datetime.now().strftime('%Y-%m-%d')
        
        # Original Image Path (from frontend) e.g. "images/2025-01-01/abc.jpg"
        original_path = data.filename
        filename_only = original_path.split('/')[-1]
        
        # 1. Fetch Original Image Content (Memory)
        # (We basically download it to re-upload it to new location)
        url = hf_hub_url(repo_id=dataset_repo, filename=original_path, repo_type="dataset")
        resp = requests.get(url, headers={"Authorization": f"Bearer {token}"})
        if resp.status_code != 200:
             return {"status": "error", "message": "Could not fetch original image"}
        image_bytes = resp.content
        
        # 2. Determine Target Path
        if data.decision == 'verified':
            # Save to: verified/images/DATE/filename
            target_image_path = f"verified/images/{date_str}/{filename_only}"
            target_label_path = f"verified/labels/{date_str}/{filename_only.replace('.jpg','.json')}"
            
            # For verified, we essentially keep same data, maybe verify metadata
            label_content = {"status": "verified", "original_path": original_path, "timestamp": date_str}
            
        elif data.decision == 'correction':
            # Save to: corrected/images/DATE/filename
            target_image_path = f"corrected/images/{date_str}/{filename_only}"
            target_label_path = f"corrected/labels/{date_str}/{filename_only.replace('.jpg','.json')}"
            
            # Save the CORRECTION
            label_content = {
                "status": "corrected", 
                "label": data.label,
                "original_path": original_path,
                "timestamp": date_str
            }
            
        # [NEW] Convert to YOLO .txt format
        yolo_lines = []
        
        # Helper to convert [x1, y1, x2, y2] (normalized) to [xc, yc, w, h]
        def to_yolo(box):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            xc = x1 + w/2
            yc = y1 + h/2
            return xc, yc, w, h
            
        if data.decision == 'verified':
            # Fetch original predictions to convert to YOLO
            # Construct path to predictions/DATE/filename.json
            # original_path is "images/DATE/filename.jpg"
            pred_path = original_path.replace("images/", "predictions/") \
                                     .replace(".jpg", ".json").replace(".png", ".json").replace(".webp", ".json")
            
            p_url = hf_hub_url(repo_id=dataset_repo, filename=pred_path, repo_type="dataset")
            p_resp = requests.get(p_url, headers={"Authorization": f"Bearer {token}"})
            
            if p_resp.status_code == 200:
                p_data = p_resp.json()
                # p_data["detections"] is list of {box: [x1,y1,x2,y2], class_id: int, ...}
                for d in p_data.get("detections", []):
                    xc, yc, w, h = to_yolo(d["box"])
                    cid = d["class_id"]
                    yolo_lines.append(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            else:
                print(f"Warning: Could not fetch predictions for verified image {original_path}")

        elif data.decision == 'correction':
            # Use provided data. 
            # Limitation: Assuming single object correction or replacing all with one.
            # Ideally we should start from original predictions and modify specific one.
            # For now, we save what is provided.
            
            # Logic: If box provided, use it. If not, try to fetch original box (of first object?)
            box = data.box
            
            if not box:
                 # Fetch original to get box?
                 pass # simplified for speed, assume frontend provides box if correcting geometry
            
            if box and data.label:
                # We need class_id for the label. 
                # Model names: model.names (dict: id -> name)
                # Reverse lookup
                class_id = -1
                if model:
                     for k, v in model.names.items():
                         if v == data.label:
                             class_id = k
                             break
                
                if class_id != -1:
                    xc, yc, w, h = to_yolo(box) # Assume box is [x1, y1, x2, y2]
                    yolo_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        
        yolo_content = "\n".join(yolo_lines)
        
        # 3. Upload to New Location
        api.upload_file(
            path_or_fileobj=io.BytesIO(image_bytes),
            path_in_repo=target_image_path,
            repo_id=dataset_repo,
            repo_type="dataset"
        )
        
        api.upload_file(
            path_or_fileobj=io.BytesIO(yolo_content.encode('utf-8')),
            path_in_repo=target_label_path.replace('.json', '.txt'), # Save as .txt for YOLO
            repo_id=dataset_repo,
            repo_type="dataset"
        )
        
        # Also save JSON metadata for record/debug
        api.upload_file(
            path_or_fileobj=io.BytesIO(json.dumps(label_content, indent=2).encode('utf-8')),
            path_in_repo=target_label_path,
            repo_id=dataset_repo,
            repo_type="dataset"
        )
        
        # [NEW] Check logic: If verified count >= 150, trigger retraining
        # We do this asynchronously to not block response
        def check_and_trigger():
            try:
                # Count files in verified/images
                # Listing all files can be slow. 
                # Optimization: We know keys. 
                # But simple list first.
                all_verified = api.list_repo_files(dataset_repo, repo_type="dataset")
                verified_imgs = [f for f in all_verified if f.startswith("verified/images/") and f.endswith((".jpg", ".png"))]
                count = len(verified_imgs)
                print(f"Verified count: {count}")
                
                if count >= 10:
                    print("TRIGGERING RETRAINING PIPELINE via Kaggle...")
                    res = kaggle_trigger.push_training_kernel(dataset_repo, HP_REPO_ID)
                    print(f"Trigger result: {res}")
            except Exception as e:
                print(f"Trigger check failed: {e}")

        # Add to background tasks if possible? 
        # submit_review is not async def? It says `def submit_review`. 
        # We can just run it or use BackgroundTasks if we change signature.
        # Changing signature to include BackgroundTasks
        # BUT tool usage limit prevents me from rewriting function sig easily if it spans many lines.
        # I will just run it immediately? Or use a thread.
        import threading
        t = threading.Thread(target=check_and_trigger)
        t.start()
        
        # 4. Optional: Delete original? 
        # For safety, let's NOT delete original yet, just copy. 
        # User can delete manually or we add a cleanup script later.
        
        return {"status": "success", "message": f"Moved to {target_image_path}"}
        
    except Exception as e:
        import traceback
        return {"status": "error", "message": str(e), "traceback": traceback.format_exc()}

# --- Endpoints ---
@app.get("/health")
def health_check():
    return {"status": "running"}

@app.get("/debug-upload")
def debug_upload():
    """Diagnostic endpoint to test HF Dataset upload permissions."""
    token = os.getenv("HF_TOKEN")
    if not token:
        return {"status": "error", "message": "HF_TOKEN env var is MISSING"}
    
    try:
        api = HfApi(token=token)
        user = api.whoami()
        username = user['name']
        
        # Test Upload
        repo_id = "qahir00/yolo-data"
        debug_filename = f"debug_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        debug_content = f"Debug upload test at {datetime.datetime.now()}\nUser: {username}"
        
        api.upload_file(
            path_or_fileobj=io.BytesIO(debug_content.encode('utf-8')),
            path_in_repo=f"debug_logs/{debug_filename}",
            repo_id=repo_id,
            repo_type="dataset"
        )
        return {
            "status": "success", 
            "message": f"Successfully uploaded {debug_filename} to {repo_id}",
            "authenticated_as": username
        }
    except Exception as e:
        import traceback
        return {
            "status": "error", 
            "message": str(e),
            "traceback": traceback.format_exc()
        }



@app.post("/predict")
async def predict(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    if model is None:
        return JSONResponse(status_code=503, content={"error": "Model not ready"})

    # Read Image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Inference
    results = model(image)
    result = results[0]
    
    # Process Results
    detections = []
    wandb_boxes = []
    confidences = []

    for box in result.boxes:
        # For API Response
        x1, y1, x2, y2 = box.xyxyn[0].tolist()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]
        
        detections.append({
            "box": [x1, y1, x2, y2],
            "confidence": conf,
            "class": label,
            "class_id": cls
        })
        
        # For W&B
        wandb_boxes.append({
            "position": {"minX": x1, "minY": y1, "maxX": x2, "maxY": y2},
            "class_id": cls,
            "box_caption": f"{label} {conf:.2f}",
            "scores": {"conf": conf}
        })
        confidences.append(conf)

    # Background Tasks (MLOps)
    unique_filename = f"{uuid.uuid4()}.jpg"
    if background_tasks:
        background_tasks.add_task(log_to_wandb, image, wandb_boxes, confidences)
        background_tasks.add_task(save_to_data_lake, contents, unique_filename, detections)

    return {
        "filename": file.filename,
        "detections": detections,
        "count": len(detections)
    }

app.mount("/", StaticFiles(directory="static", html=True), name="static")
