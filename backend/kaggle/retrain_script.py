import os
import shutil
import yaml
from ultralytics import YOLO
from huggingface_hub import HfApi, snapshot_download, login

# --- Configuration ---
# 1. Credentials
# Ensure HF_TOKEN is available via Kaggle Secrets
from kaggle_secrets import UserSecretsClient
try:
    user_secrets = UserSecretsClient()
    HF_TOKEN = user_secrets.get_secret("HF_TOKEN")
    print("HF_TOKEN retrieved from secrets.")
except:
    HF_TOKEN = os.getenv("HF_TOKEN")
    print("HF_TOKEN retrieved from env (or None).")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is missing! Please add it to Kaggle Secrets.")

# 2. Repos
DATASET_REPO = "qahir00/yolo-data"
MODEL_REPO = "qahir00/yolo11-object-detection"

def setup_directories(base_path):
    """Ensure standard YOLO directory structure exists."""
    os.makedirs(os.path.join(base_path, "train/images"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "train/labels"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "valid/images"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "valid/labels"), exist_ok=True)

def merge_verified_data(dataset_path):
    """Move verified data into the training set."""
    print("Merging verified data...")
    verified_img_root = os.path.join(dataset_path, "verified/images")
    verified_lbl_root = os.path.join(dataset_path, "verified/labels")
    
    train_img_dest = os.path.join(dataset_path, "train/images")
    train_lbl_dest = os.path.join(dataset_path, "train/labels")
    
    merged_count = 0
    
    if os.path.exists(verified_img_root):
        for root, _, files in os.walk(verified_img_root):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    # source image
                    src_img = os.path.join(root, file)
                    
                    # relative path logic to find matching label
                    # verified/images/2025-01-01/img.jpg
                    rel_dir = os.path.relpath(root, verified_img_root)
                    
                    # source label (backend saves as .txt now)
                    # verified/labels/2025-01-01/img.txt
                    lbl_name = os.path.splitext(file)[0] + ".txt"
                    src_lbl = os.path.join(verified_lbl_root, rel_dir, lbl_name)
                    
                    if os.path.exists(src_lbl):
                        # Move Image
                        shutil.move(src_img, os.path.join(train_img_dest, file))
                        # Move Label
                        shutil.move(src_lbl, os.path.join(train_lbl_dest, lbl_name))
                        merged_count += 1
    
    print(f"Merged {merged_count} verified images into training set.")
    return merged_count

def train_and_evaluate():
    login(token=HF_TOKEN)
    api = HfApi(token=HF_TOKEN)
    
    # 1. Download Dataset
    print(f"Downloading dataset {DATASET_REPO}...")
    dataset_path = snapshot_download(repo_id=DATASET_REPO, repo_type="dataset")
    print(f"Dataset downloaded to {dataset_path}")
    
    # 2. Prepare Data
    setup_directories(dataset_path)
    merge_verified_data(dataset_path)
    
    # 3. Configure data.yaml
    # We need to point data.yaml to absolute paths or relative to execution
    yaml_path = os.path.join(dataset_path, "data.yaml")
    
    # If data.yaml exists, read it to preserve class names
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
    else:
        # Fallback if missing (shouldn't happen if repo is good)
        data_config = {'names': {0: 'object'}, 'nc': 1}
        
    # Update paths to be absolute for safety
    data_config['path'] = dataset_path
    data_config['train'] = "train/images"
    data_config['val'] = "valid/images"
    # 4. Train
    print("Starting Training (New Model)...")
    # We start from 'yolo11n.pt' (or similar) to ensure we incorporate new data structure cleanly, 
    # OR we fine-tune. Fine-tuning is usually better for incremental updates.
    # Let's try to fine-tune the *current* best if available, so we don't catastrophic forget or start from scratch every time.
    start_weights = "yolo11n.pt"
    try:
        current_best_path = hf_hub_download(repo_id=MODEL_REPO, filename="best.pt")
        print(f"Found existing model at {current_best_path}, using as starting point.")
        start_weights = current_best_path
    except:
        print("No existing model found, starting from base yolo11n.pt")
        
    model = YOLO(start_weights)
        
    # Train the NEW candidate
    results = model.train(
        data="local_data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        project="yolo_retrain",
        name="new_candidate", # distinct name
        exist_ok=True,
        verbose=True
    )
    
    # 5. Evaluate NEW Model
    print("Evaluating NEW model...")
    # 'model' object is now the trained one
    new_metrics = model.val() # val on the data.yaml validation set
    new_map = new_metrics.box.map
    print(f"-> New Model mAP50-95: {new_map:.4f}")
    
    # 6. Compare with Current Best (Champion vs Challenger)
    print("Comparing with Current Production Model...")
    deploy_new = False
    
    try:
        # We need to evaluate the *current* production model on this *new/current* validation set
        # to get a fair comparison (apples to apples).
        
        # Download again (if needed) or load
        # Warning: attributes of 'model' are changed by train. Load fresh instance for old model.
        
        # Helper: download to specific location to avoid confusion
        old_model_path = hf_hub_download(repo_id=MODEL_REPO, filename="best.pt", local_dir="comparison", force_download=True)
        old_model = YOLO(old_model_path)
        
        print("Evaluating Old Model on current validation data...")
        old_metrics = old_model.val(data="local_data.yaml", project="yolo_retrain", name="old_baseline")
        old_map = old_metrics.box.map
        print(f"-> Old Model mAP50-95: {old_map:.4f}")
        
        if new_map > old_map:
            print(f"PASSED: New model ({new_map:.4f}) is better than Old model ({old_map:.4f}).")
            deploy_new = True
        else:
            print(f"FAILED: New model ({new_map:.4f}) is NOT better than Old model ({old_map:.4f}).")
            deploy_new = False
            
    except Exception as e:
        print(f"Warning: Could not enable comparison (maybe no old model?): {e}")
        print("Defaulting to DEPLOY_NEW since no baseline exists.")
        deploy_new = True

    # 7. Deployment Decision
    if deploy_new:
        # Upload new best.pt to Model Repo
        # The best weights from training are in runs/detect/new_candidate/weights/best.pt
        best_weights = os.path.join("yolo_retrain", "new_candidate", "weights", "best.pt")
        
        if os.path.exists(best_weights):
            print(f"Uploading {best_weights} to {MODEL_REPO}...")
            api.upload_file(
                path_or_fileobj=best_weights,
                path_in_repo="best.pt",
                repo_id=MODEL_REPO,
                repo_type="model",
                commit_message=f"Auto-retrain: Improvement! mAP {new_map:.4f} > old {old_map if 'old_map' in locals() else 'N/A'}"
            )
        else:
            print("Error: best.pt build failed.")
    else:
        print("Skipping deployment. Keeping current production model.")
        
    # 8. Sync Dataset State
    # We should Merge Clean UP regardless of model success?
    # YES. Because the data IS verified. The model just simply failed to improve.
    # We don't want to re-train on the same verified images again and again hoping for a different result 
    # (unless we change hyperparams, which we aren't).
    # So we ALWAYS assume the data is 'consumed'.        
    # 7. Sync Dataset State (Optional/Advanced)
    # The merged files are now in 'train' LOCALLY. 
    # To reflect this in HF, we would need to upload the entire modified 'train' folder 
    # and delete 'verified'. This is heavy.
    # STRATEGY: 
    # We upload the MOVED files to 'train' and DELETE them from 'verified' in the repo.
    # This ensures next run doesn't re-merge duplicates or count them.
    
    # For simplicity in this script, we can skip the complex sync if we rely on "verified count" logic.
    # BUT, the backend checks "verified count". If we don't clear it, it triggers infinitely.
    # So we MUST clear 'verified' in the remote repo.
    
    print("Cleaning up remote verified folder...")
    # This is tricky with `upload_folder` as it adds, doesn't delete.
    # `delete_folder` via API is safer.
    
    # 1. Upload the NEW train/images and train/labels (only the ones we added?)
    # snapshot_download gave us everything. We moved files.
    # We can just upload `train/images` and `train/labels`. 
    # HfApi.upload_folder w/ delete_patterns? No.
    
    # Robust way: 
    # 1. Upload current `train` folder (it has new files).
    # 2. Delete `verified` folder in repo.
    
    print("Syncing 'train' folder updates...")
    api.upload_folder(
        folder_path=os.path.join(dataset_path, "train"),
        path_in_repo="train",
        repo_id=DATASET_REPO,
        repo_type="dataset"
    )
    
    print("Deleting 'verified' folder...")
    # HfApi doesn't have delete_folder directly easily? 
    # delete_file loop is safer.
    # Or use `commit_operation` with delete.
    
    # We need to list files in verified again (remote) or track what we moved.
    # For now, let's use a "nuke" approach for verified folder if possible?
    # commit_operation is best.
    
    try:
        # Get list of files in verified to delete
        all_files = api.list_repo_files(repo_id=DATASET_REPO, repo_type="dataset")
        verified_files = [f for f in all_files if f.startswith("verified/")]
        
        if verified_files:
            operations = [("delete", f) for f in verified_files]
            # Chunking might be needed if too many
            # Max 50-100 ops?
            create_commit = api.create_commit(
                repo_id=DATASET_REPO,
                repo_type="dataset",
                operations=[
                    {"path": f, "operation": "delete"} for f in verified_files
                ],
                commit_message="Cleanup verified folder after merging"
            )
            print("Verified folder cleared.")
    except Exception as e:
        print(f"Cleanup failed: {e}")

if __name__ == "__main__":
    train_and_evaluate()
