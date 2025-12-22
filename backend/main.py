import os # Trigger deployment
import io
import uuid
import datetime
import json
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO
from huggingface_hub import hf_hub_download, HfApi, login
import wandb
import numpy as np
import cv2

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

def save_to_data_lake(image_bytes: bytes, filename: str):
    """Upload raw image to Hugging Face Dataset for future labeling."""
    hf_token = os.getenv("HF_TOKEN")
    if hf_token and DATASET_REPO_ID:
        try:
            api = HfApi()
            # Upload to 'images/' directory in the dataset repo
            path_in_repo = f"images/{datetime.datetime.now().strftime('%Y-%m-%d')}/{filename}"
            api.upload_file(
                path_or_fileobj=io.BytesIO(image_bytes),
                path_in_repo=path_in_repo,
                repo_id=DATASET_REPO_ID,
                repo_type="dataset",
                token=hf_token
            )
            print(f"Uploaded {filename} to {DATASET_REPO_ID}")
        except Exception as e:
            print(f"Data Lake Upload failed: {e}")

# --- Startup ---
@app.on_event("startup")
async def startup_event():
    load_model()
    # Init generic W&B run if needed, or do it per request
    if os.getenv("WANDB_API_KEY"):
        wandb.login(key=os.getenv("WANDB_API_KEY"))

# --- Endpoints ---
@app.get("/")
def health_check():
    return {"status": "running", "model": HP_REPO_ID}

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
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]
        
        detections.append({
            "bbox": [x1, y1, x2, y2],
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
        background_tasks.add_task(save_to_data_lake, contents, unique_filename)

    return {
        "filename": file.filename,
        "detections": detections,
        "count": len(detections)
    }
