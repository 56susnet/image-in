import json
import hashlib
import os
import sys

def calculate_hash(model_name):
    return hashlib.sha256(model_name.encode('utf-8')).hexdigest()

def audit_file(filename):
    print(f"\n--- AUDITING {filename} ---")
    file_path = os.path.join(os.path.dirname(__file__), "lrs", filename)
    
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
    except Exception as e:
        print(f"ERROR: Could not load {filename}: {e}")
        return

    items = data.get("data", {})
    mismatch_count = 0
    
    for key_hash, config in items.items():
        if not isinstance(config, dict): continue
        
        model_name = config.get("model_name")
        if not model_name:
            print(f"Skipping key {key_hash[:8]}... : No model_name found")
            continue
            
        calculated = calculate_hash(model_name)
        
        if calculated != key_hash:
            print(f"[FAIL] Mismatch for '{model_name}':")
            print(f"       JSON Key : {key_hash}")
            print(f"       Calculated: {calculated}")
            mismatch_count += 1
        # else:
            # print(f"[PASS] {model_name}")

    if mismatch_count == 0:
        print(f"✅ {filename}: ALL HASHES VALID ({len(items)} models)")
    else:
        print(f"❌ {filename}: FOUND {mismatch_count} HASH MISMATCHES")

if __name__ == "__main__":
    audit_file("person_config.json")
    audit_file("style_config.json")
