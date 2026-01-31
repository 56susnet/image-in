#!/usr/bin/env python3
"""
Manual Tokenizer Downloader
This script downloads CLIP and T5 tokenizers directly to /cache/hf_cache
with the correct folder structure that transformers library expects.
"""

import os
import sys

# Set HF_HOME to /cache/hf_cache
os.environ['HF_HOME'] = '/cache/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/cache/hf_cache'

print("=" * 60)
print("MANUAL TOKENIZER DOWNLOADER FOR FLUX")
print("=" * 60)
print(f"Target directory: {os.environ['HF_HOME']}")
print()

try:
    from transformers import CLIPTokenizer, T5TokenizerFast
    
    print("[1/3] Downloading CLIP tokenizer (openai/clip-vit-large-patch14)...")
    CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
    print("✅ CLIP tokenizer downloaded successfully!")
    print()
    
    print("[2/3] Downloading CLIP-bigG tokenizer (laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)...")
    CLIPTokenizer.from_pretrained('laion/CLIP-ViT-bigG-14-laion2B-39B-b160k')
    print("✅ CLIP-bigG tokenizer downloaded successfully!")
    print()
    
    print("[3/3] Downloading T5 tokenizer (google/t5-v1_1-xxl)...")
    T5TokenizerFast.from_pretrained('google/t5-v1_1-xxl')
    print("✅ T5 tokenizer downloaded successfully!")
    print()
    
    print("=" * 60)
    print("ALL TOKENIZERS DOWNLOADED SUCCESSFULLY!")
    print("=" * 60)
    print()
    print("Verifying downloads...")
    
    # Verify the downloads
    cache_dir = os.environ['HF_HOME']
    expected_dirs = [
        'models--openai--clip-vit-large-patch14',
        'models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k',
        'models--google--t5-v1_1-xxl'
    ]
    
    all_found = True
    for dir_name in expected_dirs:
        full_path = os.path.join(cache_dir, dir_name)
        if os.path.exists(full_path):
            print(f"✅ Found: {dir_name}")
        else:
            print(f"❌ Missing: {dir_name}")
            all_found = False
    
    print()
    if all_found:
        print("🎉 All tokenizers are ready for offline training!")
        sys.exit(0)
    else:
        print("⚠️  Some tokenizers are missing. Please check the errors above.")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Error downloading tokenizers: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
