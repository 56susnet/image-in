
import os
import sys
from huggingface_hub import HfApi

def rescue():
    print("--- RESCUE MISSION: UPLOADING IPUNKTEST-17 ---")
    
    # Target file specific to your failed log
    target_file = "last.safetensors"
    
    # Search locations (Docker path AND Host path)
    search_paths = [
        "/app/checkpoints",
        "./checkpoints",
        ".",
        os.getcwd()
    ]
    target_repo = "Jordansky/ipunktest-17"
    
    found_path = None
    
    # 1. Search for the file
    print(f"Searching for {target_file}...")
    
    for base_path in search_paths:
        if not os.path.exists(base_path): continue
        
        print(f"Scanning {base_path}...")
        for root, dirs, files in os.walk(base_path):
        if target_file in files:
            # Check if this folder looks like the right task
            if "ipunktest-17" in root or "18de36d8" in root:
                found_path = os.path.join(root, target_file)
                print(f"FOUND: {found_path}")
                break
    
    if not found_path:
        print(f"CRITICAL: Could not find {target_file} in {base_search_path}")
        print("Checking if it was moved to output...")
        # Fallback check
        if os.path.exists(f"/app/checkpoints/{target_file}"):
             found_path = f"/app/checkpoints/{target_file}"
        else:
             print("Aborting. File really gone.")
             return

    # 2. Upload
    print(f"Uploading to HuggingFace: {target_repo}...")
    api = HfApi()
    
    try:
        api.upload_file(
            path_or_fileobj=found_path,
            path_in_repo="last.safetensors",
            repo_id=target_repo,
            repo_type="model",
            commit_message="Rescue upload manually"
        )
        print("\n✅ SUCCESS! File uploaded successfully.")
        print(f"Check here: https://huggingface.co/{target_repo}/tree/main")
    except Exception as e:
        print(f"\n❌ FAILED again: {e}")
        print("Tips: Check your internet connection or HF_TOKEN.")

if __name__ == "__main__":
    rescue()
