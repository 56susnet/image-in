#!/usr/bin/env python3

# PHASE I = PONDASI & IMPORT

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import re
import time
import yaml
import toml
from huggingface_hub import HfApi, login

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import core.constants as cst
import trainer.constants as train_cst
import trainer.utils.training_paths as train_paths
from core.config.config_handler import save_config, save_config_toml
from core.dataset.prepare_diffusion_dataset import prepare_dataset
from core.models.utility_models import ImageModelType

# PHASE II - KUMPULAN ALAT DETEKTIF DAN PENGHITUNG OTOMATIS
# CARI PATH MODEL
def get_model_path(path: str, model_type: str = "") -> str:
    if os.path.isdir(path):
        # Khusus Flux: Jika ini adalah repo folder (ada vae/unet), return folder-nya saja
        if model_type == "flux" and any(os.path.isdir(os.path.join(path, d)) for d in ["vae", "unet", "transformer", "text_encoder"]):
            return path
            
        files = [f for f in os.listdir(path) if f.endswith(".safetensors")]
        if files:
            # Ambil file safetensors dengan ukuran terbesar
            biggest_file = max(files, key=lambda x: os.path.getsize(os.path.join(path, x)))
            return os.path.join(path, biggest_file)
    return path

# GABUNG SETTINGAN
def merge_model_config(default_config: dict, model_config: dict) -> dict:
    merged = {}
    if isinstance(default_config, dict):
        for k, v in default_config.items():
            if v is not None:
                merged[k] = v

    if isinstance(model_config, dict):
        for k, v in model_config.items():
            if v is not None:
                merged[k] = v

    return merged if merged else None

# SENSUS JUMLAH FOTO
def count_images_in_directory(directory_path: str) -> int:
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
    count = 0
    
    try:
        if not os.path.exists(directory_path):
            print(f"Directory not found: {directory_path}", flush=True)
            return 0
        
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.startswith('.'):
                    continue
                
                _, ext = os.path.splitext(file.lower())
                if ext in image_extensions:
                    count += 1
    except Exception as e:
        print(f"Error counting images in directory: {e}", flush=True)
        return 0
    
    return count

# BEDAH CACHE (UKURAN)
def find_surgical(files_found, name, golden_min, golden_max, must_contain=None, avoid=["part", "of-", "sharded"]):
    matches = []
    for entry in files_found:
        p, sz, root = entry["path"], entry["size"], entry["root"]
        if golden_min <= sz <= golden_max:
            if avoid and any(a in p.lower() for a in avoid):
                continue
            if must_contain and must_contain not in p.lower():
                continue
            score = 0
            if "flux" in p.lower() or "flux" in root.lower():
                score += 100
            if name.lower() in p.lower() or name.lower() in root.lower():
                score += 50
            matches.append((score, sz, p))
    if matches:
        matches.sort(key=lambda x: (-x[0], -x[1]))
        return matches[0][2]
    return None

# SCAN AUTOEPOCH
def load_size_based_config(model_type: str, is_style: bool, dataset_size: int) -> dict:
    config_dir = os.path.join(script_dir, "autoepoch") 
    
    if model_type == ImageModelType.FLUX.value:
        config_file = os.path.join(config_dir, "a-epochflux.json")
    elif is_style:
        config_file = os.path.join(config_dir, "a-epochstyle.json")
    else:
        config_file = os.path.join(config_dir, "a-epochperson.json")
    
    try:
        if not os.path.exists(config_file):
            print(f"Warning: Autoepoch config file not found: {config_file}", flush=True)
            return None
            
        with open(config_file, 'r', encoding='utf-8-sig') as f:
            size_config = json.load(f)
        
        size_ranges = size_config.get("size_ranges", [])
        for size_range in size_ranges:
            min_size = size_range.get("min", 0)
            max_size = size_range.get("max", float('inf'))
            
            if min_size <= dataset_size <= max_size:
                print(f"Using size-based config for {dataset_size} images (range: {min_size}-{max_size})", flush=True)
                return size_range.get("config", {})
        
        default_config = size_config.get("default", {})
        if default_config:
            print(f"Using default size-based config for {dataset_size} images", flush=True)
        return default_config
        
    except Exception as e:
        print(f"Warning: Could not load autoepoch config from {config_file}: {e}", flush=True)
        return None

# CEK KLASIFIKASI JUMLAH DATASET
def get_dataset_size_category(dataset_size: int) -> str:
    """MAP DATASET SIZE TO CATEGORY LABELS USED IN LRS CONFIG."""
    if dataset_size <= 20:
        cat = "small"
    elif dataset_size <= 40:
        cat = "medium"
    else:
        cat = "large"
    
    print(f"DEBUG_LRS: Image count {dataset_size} mapped to category -> [{cat.upper()}]", flush=True)
    return cat

# AMBIL RESEP LRS (OPTIMASI)
def get_config_for_model(lrs_config: dict, model_hash: str, dataset_size: int = None, raw_model_name: str = None) -> dict:
    if not isinstance(lrs_config, dict):
        return None

    data = lrs_config.get("data")
    default_config = lrs_config.get("default", {})
    
    target_config = None

    # IDENTIFIKASI & PENCARIAN RESEP LRS
    clean_name = raw_model_name.strip().strip("'").strip('"') if raw_model_name else None

    # CEK SIDIK JARI (HASH)
    if isinstance(data, dict):
        if model_hash in data:
            target_config = data.get(model_hash)
            print(f"DEBUG_LRS: MATCH [HASH] -> {model_hash}", flush=True)
            
        # CEK NAMA MODEL (DIRECT)
        elif clean_name:
             if clean_name in data:
                 target_config = data.get(clean_name)
                 print(f"DEBUG_LRS: MATCH [DIRECT KEY] -> {clean_name}", flush=True)
             else:
                 # SCAN DATABASE (ITERATIVE)
                 for key, val in data.items():
                     if isinstance(val, dict) and val.get("model_name") == clean_name:
                         target_config = val
                         print(f"DEBUG_LRS: MATCH [FIELD SCAN] -> {clean_name} (Key: {key})", flush=True)
                         break
        # ADAPTASI JUMLAH FOTO (SMART MERGE)
        if not target_config and clean_name:
             print(f"DEBUG_LRS: FAIL lookup for '{clean_name}'. Hash was '{model_hash}'", flush=True)

    if target_config:
        # IF DATASET_SIZE PROVIDED AND MODEL_CONFIG HAS SIZE CATEGORIES, MERGE THEM
        if dataset_size is not None and isinstance(target_config, dict):
            size_category = get_dataset_size_category(dataset_size)
            
            # CHECK IF MODEL_CONFIG HAS SIZE-SPECIFIC SETTINGS
            if size_category in target_config:
                size_specific_config = target_config.get(size_category, {})
                # PENGGABUNGAN RESEP FINAL & FALLBACK
                base_model_config = {k: v for k, v in target_config.items() if k not in ["small", "medium", "large"]}
                merged = merge_model_config(default_config, base_model_config)
                print(f"DEBUG_LRS: Merged Size Config ({size_category})", flush=True)
                return merge_model_config(merged, size_specific_config)
        
        return merge_model_config(default_config, target_config)

    if default_config:
        print("DEBUG_LRS: Using Default Config", flush=True)
        return default_config

    return None

# BUKA BRANKAS JSON
def load_lrs_config(model_type: str, is_style: bool) -> dict:
    config_dir = os.path.join(script_dir, "lrs")

    if model_type == ImageModelType.FLUX.value:
        config_file = os.path.join(config_dir, "flux.json")
    elif model_type == ImageModelType.QWEN_IMAGE.value:
        config_file = os.path.join(config_dir, "qwen.json")
    elif model_type == ImageModelType.Z_IMAGE.value:
        config_file = os.path.join(config_dir, "zimage.json")
    elif is_style:
        config_file = os.path.join(config_dir, "style_config.json")
    else:
        config_file = os.path.join(config_dir, "person_config.json")
    
    try:
        with open(config_file, 'r', encoding='utf-8-sig') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load LRS config from {config_file}: {e}", flush=True)
        return None

# DETEKSI PERSON/STYLE
def detect_is_style(train_data_dir):
    """DETECT IF THE DATASET CONTAINS STYLE-BASED PROMPTS USING BALANCED LOGIC."""
    try:
        # PENGUMPULAN TEKS PROMPT DATASET
        sub_dirs = [d for d in os.listdir(train_data_dir) if os.path.isdir(os.path.join(train_data_dir, d))]
        prompts = []
        
        for sub in sub_dirs:
           
            if "_" in sub and sub.split("_")[0].isdigit():
                prompts_path = os.path.join(train_data_dir, sub)
                for file in os.listdir(prompts_path):
                    if file.endswith(".txt"):
                        with open(os.path.join(prompts_path, file), "r", encoding='utf-8') as f:
                            prompts.append(f.read().strip().lower())
        
        if not prompts:
            return False
        # SCANNING KATA KUNCI (STYLE VS PERSON)
        style_list = [
            "Watercolor Painting", "Oil Painting", "Digital Art", "Pencil Sketch", "Comic Book Style",
            "Cyberpunk", "Steampunk", "Impressionist", "Pop Art", "Minimalist", "Gothic", "Art Nouveau",
            "Pixel Art", "Anime", "3D Render", "Low Poly", "Photorealistic", "Vector Art",
            "Abstract Expressionism", "Realism", "Futurism", "Cubism", "Surrealism", "Baroque",
            "Renaissance", "Fantasy Illustration", "Sci-Fi Illustration", "Ukiyo-e", "Line Art",
            "Black and White Ink Drawing", "Graffiti Art", "Stencil Art", "Flat Design", "Isometric Art",
            "Retro 80s Style", "Vaporwave", "Dreamlike", "High Fantasy", "Dark Fantasy", "Medieval Art",
            "Art Deco", "Hyperrealism", "Sculpture Art", "Caricature", "Chibi", "Noir Style",
            "Lowbrow Art", "Psychedelic Art", "Vintage Poster", "Manga", "Holographic", "Kawaii",
            "Monochrome", "Geometric Art", "Photocollage", "Mixed Media", "Ink Wash Painting",
            "Charcoal Drawing", "Concept Art", "Digital Matte Painting", "Pointillism", "Expressionism",
            "Sumi-e", "Retro Futurism", "Pixelated Glitch Art", "Neon Glow", "Street Art",
            "Acrylic Painting", "Bauhaus", "Flat Cartoon Style", "Carved Relief Art", "Fantasy Realism"
        ]
        
        person_keywords = ["man", "woman", "girl", "boy", "person", "lady", "male", "female", "face", "hair", "solo"]
        
        person_count = 0
        style_matches = {s: 0 for s in style_list}
        
        for prompt in prompts:
            if any(word in prompt for word in person_keywords):
                person_count += 1
            
            for style in style_list:
                if style.lower() in prompt:
                    style_matches[style] += 1
        
        # KALKULASI DOMINASI & KEPUTUSAN FINAL
        prompt_total = len(prompts)
        person_ratio = person_count / prompt_total if prompt_total > 0 else 0
        
        max_style_ratio = 0
        top_style = "None"
        if prompt_total > 0:
            for style, count in style_matches.items():
                ratio = count / prompt_total
                if ratio > max_style_ratio:
                    max_style_ratio = ratio
                    top_style = style
        
        print(f"DEBUG_CLASSIFY: Person Ratio: {person_ratio:.2f}, Max Style Ratio: {max_style_ratio:.2f} ({top_style})", flush=True)

        # LOGIC DETEKSI BARU (LEBIH SENSITIF TERHADAP STYLE)
        if max_style_ratio >= 0.20:
            print("DEBUG_CLASSIFY: Decision -> [STYLE] (Strong Signal)", flush=True)
            return True
        if max_style_ratio >= 0.10 and person_ratio < 0.20:
            print("DEBUG_CLASSIFY: Decision -> [STYLE] (Moderate Signal)", flush=True)
            return True
        
        print("DEBUG_CLASSIFY: Decision -> [PERSON] (Default)", flush=True)
        return False
        
    except Exception as e:
        print(f"Warning during style detection: {e}", flush=True)
        return False

# PHASE III - MENGUNCI TIPE TUGAS DAN MERACIK FILE INSTRUKSI
def create_config(task_id, model_path, model_name, model_type, expected_repo_name, trigger_word, hours_to_complete=None):
    train_data_dir = os.path.join(train_cst.IMAGE_CONTAINER_IMAGES_PATH, task_id)
    
    is_style = detect_is_style(train_data_dir)
    detection_method = "Dataset-driven"
    
    if not is_style:
         if expected_repo_name and "style" in expected_repo_name.lower() and "person" not in expected_repo_name.lower():
              is_style = True
              detection_method = "Metadata-forced"

    task_type = "style" if is_style else "person"
    print(f"DEBUG_TYPE: Task detected as [{task_type.upper()}] via {detection_method}", flush=True)

    dataset_size = count_images_in_directory(train_data_dir)
    is_ai_toolkit = model_type in [ImageModelType.Z_IMAGE.value, ImageModelType.QWEN_IMAGE.value]
    output_dir = train_paths.get_checkpoints_output_path(task_id, expected_repo_name or "output")
    
    config_dir = os.path.join(script_dir, "core", "config")
    potential_templates = [
        f"base_diffusion_{model_type}_{task_type}.toml",
        f"base_diffusion_{model_type}_{task_type}.yaml",
        f"base_diffusion_{model_type}.toml",
        f"base_diffusion_{model_type}.yaml",
        f"base_{model_type}.yaml" 
    ]
    
    config_template_path = None
    for template_name in potential_templates:
        found_path = os.path.join(config_dir, template_name)
        if os.path.exists(found_path):
            config_template_path = found_path
            break
    
    if not config_template_path:
        raise FileNotFoundError(f"Could not find a valid config template for {model_type} in {config_dir}")

    # DAPUR RANK MODEL
    network_config_person = {
        "stabilityai/stable-diffusion-xl-base-1.0": 235, "Lykon/dreamshaper-xl-1-0": 235, "Lykon/art-diffusion-xl-0.9": 235,
        "SG161222/RealVisXL_V4.0": 467, "stablediffusionapi/protovision-xl-v6.6": 235, "stablediffusionapi/omnium-sdxl": 235,
        "GraydientPlatformAPI/realism-engine2-xl": 235, "GraydientPlatformAPI/albedobase2-xl": 467, "KBlueLeaf/Kohaku-XL-Zeta": 235,
        "John6666/hassaku-xl-illustrious-v10style-sdxl": 228, "John6666/nova-anime-xl-pony-v5-sdxl": 235, "cagliostrolab/animagine-xl-4.0": 699,
        "dataautogpt3/CALAMITY": 235, "dataautogpt3/ProteusSigma": 235, "dataautogpt3/ProteusV0.5": 467, "dataautogpt3/TempestV0.1": 500,
        "ehristoforu/Visionix-alpha": 235, "femboysLover/RealisticStockPhoto-fp16": 467, "fluently/Fluently-XL-Final": 228,
        "mann-e/Mann-E_Dreams": 456, "misri/leosamsHelloworldXL_helloworldXL70": 235, "misri/zavychromaxl_v90": 235,
        "openart-custom/DynaVisionXL": 228, "recoilme/colorfulxl": 228, "zenless-lab/sdxl-aam-xl-anime-mix": 456,
        "zenless-lab/sdxl-anima-pencil-xl-v5": 228, "zenless-lab/sdxl-anything-xl": 228, "zenless-lab/sdxl-blue-pencil-xl-v7": 467,
        "Corcelio/mobius": 228, "GHArt/Lah_Mysterious_SDXL_V4.0_xl_fp16": 235, "OnomaAIResearch/Illustrious-xl-early-release-v0": 228,
        "bghira/terminus-xl-velocity-v2": 235, "ifmain/UltraReal_Fine-Tune": 467
    }

    network_config_style = {
        "stabilityai/stable-diffusion-xl-base-1.0": 235, "Lykon/dreamshaper-xl-1-0": 235, "Lykon/art-diffusion-xl-0.9": 235,
        "SG161222/RealVisXL_V4.0": 235, "stablediffusionapi/protovision-xl-v6.6": 235, "stablediffusionapi/omnium-sdxl": 235,
        "GraydientPlatformAPI/realism-engine2-xl": 235, "GraydientPlatformAPI/albedobase2-xl": 235, "KBlueLeaf/Kohaku-XL-Zeta": 235,
        "John6666/hassaku-xl-illustrious-v10style-sdxl": 235, "John6666/nova-anime-xl-pony-v5-sdxl": 235, "cagliostrolab/animagine-xl-4.0": 235,
        "dataautogpt3/CALAMITY": 235, "dataautogpt3/ProteusSigma": 235, "dataautogpt3/ProteusV0.5": 235, "dataautogpt3/TempestV0.1": 500,
        "ehristoforu/Visionix-alpha": 235, "femboysLover/RealisticStockPhoto-fp16": 235, "fluently/Fluently-XL-Final": 235,
        "mann-e/Mann-E_Dreams": 235, "misri/leosamsHelloworldXL_helloworldXL70": 235, "misri/zavychromaxl_v90": 235,
        "openart-custom/DynaVisionXL": 235, "recoilme/colorfulxl": 235, "zenless-lab/sdxl-aam-xl-anime-mix": 235,
        "zenless-lab/sdxl-anima-pencil-xl-v5": 235, "zenless-lab/sdxl-anything-xl": 235, "zenless-lab/sdxl-blue-pencil-xl-v7": 235,
        "Corcelio/mobius": 235, "GHArt/Lah_Mysterious_SDXL_V4.0_xl_fp16": 235, "OnomaAIResearch/Illustrious-xl-early-release-v0": 235,
        "bghira/terminus-xl-velocity-v2": 235, "ifmain/UltraReal_Fine-Tune": 235
    }

    network_config_flux = {
        "dataautogpt3/FLUX-MonochromeManga": 350, "mikeyandfriends/PixelWave_FLUX.1-dev_03": 350,
        "rayonlabs/FLUX.1-dev": 350, "mhnakif/fluxunchained-dev": 350
    }

    config_mapping = {
        228: {"network_dim": 32, "network_alpha": 32, "network_args": ["conv_dim=8", "conv_alpha=8", "algo=locon"]},
        235: {"network_dim": 32, "network_alpha": 32, "network_args": ["conv_dim=8", "conv_alpha=8", "algo=locon"]},
        456: {"network_dim": 64, "network_alpha": 64, "network_args": ["conv_dim=16", "conv_alpha=16", "algo=locon"]},
        467: {"network_dim": 64, "network_alpha": 64, "network_args": ["conv_dim=16", "conv_alpha=16", "algo=locon"]},
        699: {"network_dim": 96, "network_alpha": 96, "network_args": ["conv_dim=32", "conv_alpha=32", "algo=locon"]},
        900: {"network_dim": 128, "network_alpha": 128, "network_args": ["conv_dim=32", "conv_alpha=32", "algo=locon"]},
        500: {"network_dim": 64, "network_alpha": 64, "network_args": ["conv_dim=4", "conv_alpha=4", "dropout=0"]},
        999: {"network_dim": 64, "network_alpha": 64, "network_args": ["conv_dim=32", "conv_alpha=32"]},
        350: {"network_dim": 128, "network_alpha": 128, "network_args": ["train_double_block_indices=all", "train_single_block_indices=all", "train_t5xxl=True"]}
    }

    if model_type == ImageModelType.Z_IMAGE.value:
        config_id = 999
    elif model_type == ImageModelType.FLUX.value:
        config_id = network_config_flux.get(model_name, 350)
    elif model_type == ImageModelType.QWEN_IMAGE.value:
        config_id = 350 
    else:
        target_dict = network_config_style if is_style else network_config_person
        config_id = target_dict.get(model_name, 235)

    model_params = config_mapping.get(config_id, config_mapping[235]) if config_id else {"network_dim": 32, "network_alpha": 32, "network_args": []}
    net_dim = model_params["network_dim"]

    print(f"[CONFIG SPECIFICATION - LAYER II] Model '{model_name}' Rank {net_dim}", flush=True)

    lrs_settings = None
    size_config = None
    
    lrs_config = load_lrs_config(model_type, is_style)
    if lrs_config:
        clean_model_name = model_name.strip().strip("'").strip('"')
        model_hash = hashlib.sha256(clean_model_name.encode('utf-8')).hexdigest()
        lrs_settings = get_config_for_model(lrs_config, model_hash, dataset_size, clean_model_name)

    if dataset_size > 0:
        size_config = load_size_based_config(model_type, is_style, dataset_size)

    if is_ai_toolkit:
        with open(config_template_path, "r") as file:
            config = yaml.safe_load(file)
        
        if 'config' in config and 'process' in config['config']:
            for process in config['config']['process']:
                if 'model' in process:
                    if model_path.endswith(".safetensors"):
                        process['model']['name_or_path'] = os.path.dirname(model_path)
                    else:
                        process['model']['name_or_path'] = model_path

                    if model_type == ImageModelType.Z_IMAGE.value:
                        process['model']['assistant_lora_path'] = os.path.join(train_cst.HUGGINGFACE_CACHE_PATH, "zimage_turbo_training_adapter_v2.safetensors")
                        
                    if 'training_folder' in process:
                        if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
                        process['training_folder'] = output_dir
                
                if 'datasets' in process:
                    rep_factor = cst.DIFFUSION_SDXL_REPEATS if model_type == ImageModelType.SDXL.value else cst.DIFFUSION_FLUX_REPEATS
                    dataset_path = train_data_dir
                    for dataset in process['datasets']:
                        dataset['folder_path'] = dataset_path

                if 'train' not in process: process['train'] = {}
                
                def calculate_steps(epochs):
                    batch_size = process.get('train', {}).get('batch_size', 1)
                    if dataset_size == 0: return epochs
                    return int(epochs * (dataset_size * rep_factor / batch_size))

                if lrs_settings:
                    for key, value in lrs_settings.items():
                        if key in ["unet_lr", "text_encoder_lr", "learning_rate"]: process['train']['lr'] = value
                        elif key in ["optimizer_type", "optimizer"]: process['train']['optimizer'] = value
                        elif key in ["max_train_epochs", "steps"]:
                            process['train']['steps'] = calculate_steps(value) if key == "max_train_epochs" else value
                        elif key in ["rank", "alpha"]:
                            if 'network' not in process: process['network'] = {}
                            if key == "rank": process['network']['linear'] = value
                            elif key == "alpha": process['network']['linear_alpha'] = value
                        elif key == "optimizer_args" and isinstance(value, list):
                            opt_params = {}
                            for item in value:
                                if "=" in item:
                                    k, v = item.split("=", 2)
                                    opt_params[k.strip()] = v
                            process['train']['optimizer_params'] = opt_params
                        else: process['train'][key] = value

                if trigger_word:
                    process['trigger_word'] = trigger_word
        
        config_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}.yaml")
        save_config(config, config_path)
        return config_path, output_dir
    else:
        with open(config_template_path, "r") as file:
            config = toml.load(file)

        config['train_data_dir'] = train_data_dir
        if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
        config['output_dir'] = output_dir

        # TENTUKAN ALAMAT MODEL (BIAR KOHYA GAK NYASAR)
        if model_type == "flux":
            # Cek apakah ada model pre-installed di /app/flux (Validator Mode)
            def is_valid_flux_resource(p):
                if not os.path.exists(p): return False
                if os.path.isfile(p): return p.endswith(".safetensors")
                for root, _, files in os.walk(p):
                    if any(f.endswith(".safetensors") for f in files): return True
                return False

            if is_valid_flux_resource("/app/flux/unet"):
                print("[FLUX-ADAPT] Using working pre-installed /app/flux path", flush=True)
                config["pretrained_model_name_or_path"] = "/app/flux/unet"
                config["ae"] = "/app/flux/ae.safetensors"
                config["clip_l"] = "/app/flux/clip_l.safetensors"
                config["t5xxl"] = "/app/flux/t5xxl_fp16.safetensors"
            else:
                # Offline/Standalone Mode: Gunakan model_path hasil download
                print(f"[FLUX-ADAPT] /app/flux not found or poisoned. Final model path: {model_path}", flush=True)
                
                # Jika model_path adalah directory (Diffusers), biarkan Kohya handle otomatis
                if os.path.isdir(model_path):
                    print("[FLUX-ADAPT] Full Repo Folder detected. Letting Kohya handle components.", flush=True)
                    config["pretrained_model_name_or_path"] = model_path
                    # Hapus manual override agar tidak tabrakan dengan internal loader Kohya
                    for k in ["ae", "clip_l", "t5xxl"]: 
                        if k in config: del config[k]
                else:
                    config["pretrained_model_name_or_path"] = model_path
                    # Single File Mode: Cari komponen pelengkap di folder yang sama
                    model_dir = os.path.dirname(model_path)
                    def find_flux_component(name_pattern, preferred_subdir):
                        pref = os.path.join(model_dir, preferred_subdir)
                        if os.path.isdir(pref):
                            for r, _, fs in os.walk(pref):
                                for f in fs:
                                    if f.endswith(".safetensors"): return os.path.join(r, f)
                        for r, _, fs in os.walk(model_dir):
                            for f in fs:
                                if name_pattern.lower() in f.lower() and f.endswith(".safetensors"):
                                    return os.path.join(r, f)
                        return None

                    flux_ae = find_flux_component("ae", "vae")
                    flux_clip = find_flux_component("clip_l", "text_encoder")
                    flux_t5 = find_flux_component("t5xxl", "text_encoder_2")
                    if flux_ae: config["ae"] = flux_ae
                    if flux_clip: config["clip_l"] = flux_clip
                    if flux_t5: config["t5xxl"] = flux_t5
        else:
            config["pretrained_model_name_or_path"] = model_path

        # Suntik Bumbu SDXL Bos (DIM, ALPHA, & ARGS)
        config["network_dim"] = model_params.get("network_dim", 32)
        config["network_alpha"] = model_params.get("network_alpha", 32)
        
        if "network_args" in model_params:
            if "network_args" not in config: config["network_args"] = []
            for arg in model_params["network_args"]:
                if arg not in config["network_args"]:
                    config["network_args"].append(arg)

        # SIMPAN & SUNTIK DATA FINAL
        section_map = {
             "optimizer_type": (None, "optimizer_type"), "optimizer_args": (None, "optimizer_args"),
             "unet_lr": (None, "unet_lr"), "text_encoder_lr": (None, "text_encoder_lr"),
             "network_dim": (None, "network_dim"), "network_alpha": (None, "network_alpha"),
             "t5xxl": (None, "t5xxl"), "clip_l": (None, "clip_l"), "ae": (None, "ae"),
             "noise_offset": (None, "noise_offset"), "multires_noise_iterations": (None, "multires_noise_iterations"),
             "multires_noise_discount": (None, "multires_noise_discount"),
             "max_train_epochs": (None, "max_train_epochs"), "max_train_steps": (None, "max_train_steps")
        }
        
        # APPLY OVERRIDES (Priority: Autoepoch < LRS)
        configs_to_apply = []
        if size_config: configs_to_apply.append(("Size-Based", size_config))
        if lrs_settings: configs_to_apply.append(("LRS-Override", lrs_settings))
            
        for name, cfg in configs_to_apply:
            for key, value in cfg.items():
                if key in section_map:
                    sec, target = section_map[key]
                    if sec:
                        if sec not in config: config[sec] = {}
                        config[sec][target] = value
                    else: config[target] = value
                else:
                    if key == "max_train_epochs":
                        if "max_train_steps" in config or (lrs_settings and "max_train_steps" in lrs_settings):
                            if "max_train_epochs" in config: del config["max_train_epochs"]
                            continue 
                        if "max_train_steps" in config: del config["max_train_steps"]
                    config[key] = value

        # UNIVERSAL CONFIG STRATEGY (FINAL OVERRIDE - MICRO-DATASET FIX)
        # Ini ditaruh terakhir agar tidak tertindih oleh LRS JSON
        if dataset_size < 15:
             print("[STRATEGY] Applying Universal Config for Micro-Dataset (<15 images) - FINAL OVERRIDE (SAFE)", flush=True)
             if "optimizer_args" not in config: config["optimizer_args"] = []
             
             # Hanya suntik d_coef=1.0 jika optimizerniernya adalah Prodigy (Biar Flux/Lion gak crash)
             is_prodigy = config.get("optimizer_type", "").lower() == "prodigy"
             if is_prodigy:
                 if "d_coef=1.0" not in config["optimizer_args"]: config["optimizer_args"].append("d_coef=1.0")
             
             # Pastikan tidak ada d_coef ganda (Membersihkan sisa-sisa ID model jika ada)
             if is_prodigy:
                 config["optimizer_args"] = [arg for arg in config["optimizer_args"] if not arg.startswith("d_coef=") or arg == "d_coef=1.0"]

        config_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}.toml")
        save_config_toml(config, config_path)
        print(f"Created config at {config_path}", flush=True)
        return config_path, output_dir

# PHASE IV - TRAINING EXECUTION
def run_training(model_type, config_path, output_dir, hours_to_complete=None, script_start_time=None, model_path=None):
    print(f"Starting training with config: {config_path}", flush=True)

    is_ai_toolkit = model_type in [ImageModelType.Z_IMAGE.value, ImageModelType.QWEN_IMAGE.value]
    env = os.environ.copy()
    
    target_cache = "/cache/hf_cache" 
    env.update({
        "HF_HOME": target_cache,
        "TRANSFORMERS_CACHE": target_cache,
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1"
    })

    if is_ai_toolkit:
        training_command = ["python3", "/app/ai-toolkit/run.py", config_path]
    else:
        training_command = [
            "accelerate", "launch",
            "--dynamo_backend", "no",
            "--dynamo_mode", "default",
            "--mixed_precision", "bf16",
            "--num_processes", "1",
            "--num_machines", "1",
            "--num_cpu_threads_per_process", "2",
            f"/app/sd-script/{model_type}_train_network.py",
            "--config_file", config_path
        ]

    try:
        print(f"Launching {model_type.upper()} training with command: {' '.join(training_command)}", flush=True)
        process = subprocess.Popen(training_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        for line in process.stdout: print(line, end="", flush=True)
        return_code = process.wait()
        if return_code != 0: raise subprocess.CalledProcessError(return_code, training_command)
        print("Training subprocess completed successfully.", flush=True)
    except subprocess.CalledProcessError as e:
        print("Training subprocess failed!", flush=True)
        raise RuntimeError(f"Training subprocess failed with exit code {e.returncode}")

# PHASE V - MEMBERI IDENTITAS DAN MERAPIKAN MODEL
def detect_subfolder(base_folder: str) -> str | None: 
    if not os.path.isdir(base_folder): return None
    for item in os.listdir(base_folder):
        item_path = os.path.join(base_folder, item)
        if not os.path.isdir(item_path): continue
        for file in os.listdir(item_path):
            if file.endswith('.safetensors'): return item_path
    return None

def patch_model_metadata(output_dir: str, base_model_id: str):
    try:
        adapter_config_path = os.path.join(output_dir, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, "r") as f: config = json.load(f)
            config["base_model_name_or_path"] = base_model_id
            with open(adapter_config_path, "w") as f: json.dump(config, f, indent=2)
        readme_path = os.path.join(output_dir, "README.md")
        if os.path.exists(readme_path):
            with open(readme_path, "r") as f: lines = f.readlines()
            new_lines = [f"base_model: {base_model_id}\n" if l.strip().startswith("base_model:") else l for l in lines]
            with open(readme_path, "w") as f: f.writelines(new_lines)
    except Exception as e: print(f"[METADATA] WARNING: {e}", flush=True)

# PHASE VI - MENGATUR URUTAN EKSEKUSI
async def main():
    script_start_time = time.time()
    print("--------------------------------------------------", flush=True)
    print("ONLY NINJA CAN STOP ME NOW", flush=True)
    print("--------------------------------------------------", flush=True)
    print("---STARTING TRAINING ---", flush=True)
    
    parser = argparse.ArgumentParser(description="Image Model Training Script")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-zip", required=True)
    parser.add_argument("--model-type", required=True, choices=["sdxl", "flux", "qwen-image", "z-image"])
    parser.add_argument("--expected-repo-name")
    parser.add_argument("--trigger-word")
    parser.add_argument("--hours-to-complete", type=float, required=True)
    args = parser.parse_args()

    os.makedirs(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, exist_ok=True)
    os.makedirs(train_cst.IMAGE_CONTAINER_IMAGES_PATH, exist_ok=True)
    os.makedirs(train_cst.HUGGINGFACE_CACHE_PATH, exist_ok=True)

    model_path = get_model_path(train_paths.get_image_base_model_path(args.model), model_type=args.model_type)
    print("Preparing dataset...", flush=True)

    prepare_dataset(
        training_images_zip_path=train_paths.get_image_training_zip_save_path(args.task_id),
        training_images_repeat=cst.DIFFUSION_SDXL_REPEATS if args.model_type == ImageModelType.SDXL.value else cst.DIFFUSION_FLUX_REPEATS,
        instance_prompt=cst.DIFFUSION_DEFAULT_INSTANCE_PROMPT,
        class_prompt=cst.DIFFUSION_DEFAULT_CLASS_PROMPT,
        job_id=args.task_id,
        output_dir=train_cst.IMAGE_CONTAINER_IMAGES_PATH
    )

    config_path, output_dir = create_config(args.task_id, model_path, args.model, args.model_type, args.expected_repo_name, args.trigger_word, args.hours_to_complete)
    run_training(args.model_type, config_path, output_dir, args.hours_to_complete, script_start_time, model_path=model_path)

# PHASE VII - FINAL CHECK DAN UPLOAD HASIL
    hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    hf_user = os.getenv("HUGGINGFACE_USERNAME") or os.getenv("HF_USERNAME")
    repo_name = args.expected_repo_name
    
    if not all([hf_token, hf_user, repo_name]):
        print(f"[UPLOAD] SKIPPING UPLOAD: MISSING CREDENTIALS", flush=True)
        return

    repo_id = f"{hf_user}/{repo_name}"
    print(f"[UPLOAD] STARTING TO HTTPS://HUGGINGFACE.CO/{repo_id}", flush=True)

    try:
        login(token=hf_token); api = HfApi()
        final_upload_dir = output_dir
        checkpoint_sub = detect_subfolder(output_dir)
        if checkpoint_sub: final_upload_dir = checkpoint_sub

        for root, dirs, files in os.walk(final_upload_dir):
            for file in files:
                if file.endswith((".safetensors", ".json", ".md", ".yaml", ".toml")):
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(final_upload_dir, file)
                    if src_path != dst_path:
                        if os.path.exists(dst_path): os.remove(dst_path)
                        shutil.move(src_path, dst_path)

        target_model = args.model
        if args.model_type == ImageModelType.SDXL.value and ("visionix" in str(args.model).lower() or "sdxl" in str(output_dir).lower()):
            target_model = "stabilityai/stable-diffusion-xl-base-1.0"
        
        patch_model_metadata(final_upload_dir, target_model)
        api.create_repo(repo_id=repo_id, token=hf_token, exist_ok=True)
        api.upload_folder(repo_id=repo_id, folder_path=final_upload_dir, commit_message=f"Upload {args.task_id}")
        print(f"[UPLOAD] SUCCESS! VIEW AT HTTPS://HUGGINGFACE.CO/{repo_id}", flush=True)
    except Exception as e: print(f"[UPLOAD] ERROR: {e}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
