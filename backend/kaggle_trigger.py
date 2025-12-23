import os
import json
# from kaggle.api.kaggle_api_extended import KaggleApi # Lazy loaded inside functions

def authenticate_kaggle():
    """Authenticates using environment variables."""
    if not os.getenv("KAGGLE_USERNAME") or not os.getenv("KAGGLE_KEY"):
        print("Error: Kaggle credentials not found in environment.")
        return None
    
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api

def push_training_kernel(dataset_repo, model_repo):
    """
    Triggers the Kaggle training kernel.
    Since we can't easily 'trigger' a static kernel without a competition submission or similar,
    we usually 'push' a kernel version to run it.
    
    This function assumes we have the kernel code locally in `../kaggle` relative to backend,
    or we construct it here.
    """
    api = authenticate_kaggle()
    if not api:
        return {"status": "error", "message": "Kaggle auth failed"}

    try:
        # We need to make sure the kernel metadata points to the right user/slug
        # and checking if we need to update any variables in the script (like HF_TOKEN)
        # However, passing secrets to Kernels is tricky via API.
        # Best practice: relying on User Secrets in Kaggle, but verifying they exist is hard.
        # We will assume the user has set HF_TOKEN in Kaggle Secrets as requested.
        
        # Path to kernel directory
        # Path to kernel directory
        # Since we moved kaggle folder INSIDE backend for Docker compatibility:
        kernel_dir = os.path.join(os.path.dirname(__file__), "kaggle")
        
        # Push the kernel
        # This creates a new version and runs it.
        result = api.kernel_push(kernel_dir)
        
        return {"status": "success", "message": f"Kaggle Kernel pushed: {result}"}

    except Exception as e:
        print(f"Kaggle Push Error: {e}")
        return {"status": "error", "message": str(e)}
