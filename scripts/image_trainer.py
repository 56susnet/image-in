#!/usr/bin/env python3
"""
# ING MADYA MANGUN KARSA
"""

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



def get_model_path(path: str) -> str:
    if os.path.isdir(path):
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        if len(files) == 1 and files[0].endswith(".safetensors"):
            return os.path.join(path, files[0])
    return path


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


def load_size_based_config(model_type: str, is_style: bool, dataset_size: int) -> dict:
    config_dir = os.path.join(script_dir, "autoepoch") 
    
    if model_type == ImageModelType.FLUX.value:
        config_file = os.path.join(config_dir, "a-epochflux.json")
    elif model_type == ImageModelType.QWEN_IMAGE.value:
        config_file = os.path.join(config_dir, "a-epochqwen.json")
    elif model_type == ImageModelType.Z_IMAGE.value:
        config_file = os.path.join(config_dir, "a-epochz.json")
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


def get_dataset_size_category(dataset_size: int) -> str:
    """Map dataset size to category labels used in LRS config."""
    if dataset_size <= 20:
        cat = "small"
    elif dataset_size <= 40:
        cat = "medium"
    else:
        cat = "large"
    
    print(f"DEBUG_LRS: Image count {dataset_size} mapped to category -> [{cat.upper()}]", flush=True)
    return cat


def get_config_for_model(lrs_config: dict, model_hash: str, dataset_size: int = None, raw_model_name: str = None) -> dict:
    if not isinstance(lrs_config, dict):
        return None

    data = lrs_config.get("data")
    default_config = lrs_config.get("default", {})
    
    target_config = None

    # Sanitize input name if provided
    clean_name = raw_model_name.strip().strip("'").strip('"') if raw_model_name else None

    # 1. Try Hash Lookup
    if isinstance(data, dict):
        if model_hash in data:
            target_config = data.get(model_hash)
            print(f"DEBUG_LRS: MATCH [HASH] -> {model_hash}", flush=True)
            
        # 2. Try Raw Name Lookup (Fallback)
        elif clean_name:
             # Direct lookup
             if clean_name in data:
                 target_config = data.get(clean_name)
                 print(f"DEBUG_LRS: MATCH [DIRECT KEY] -> {clean_name}", flush=True)
             else:
                 # Iterative lookup (scan 'model_name' field)
                 for key, val in data.items():
                     if isinstance(val, dict) and val.get("model_name") == clean_name:
                         target_config = val
                         print(f"DEBUG_LRS: MATCH [FIELD SCAN] -> {clean_name} (Key: {key})", flush=True)
                         break
        
        if not target_config and clean_name:
             print(f"DEBUG_LRS: FAIL lookup for '{clean_name}'. Hash was '{model_hash}'", flush=True)

    if target_config:
        # If dataset_size provided and model_config has size categories, merge them
        if dataset_size is not None and isinstance(target_config, dict):
            size_category = get_dataset_size_category(dataset_size)
            
            # Check if model_config has size-specific settings
            if size_category in target_config:
                size_specific_config = target_config.get(size_category, {})
                # Merge Config
                base_model_config = {k: v for k, v in target_config.items() if k not in ["small", "medium", "large"]}
                merged = merge_model_config(default_config, base_model_config)
                print(f"DEBUG_LRS: Merged Size Config ({size_category})", flush=True)
                return merge_model_config(merged, size_specific_config)
        
        return merge_model_config(default_config, target_config)

    if default_config:
        print("DEBUG_LRS: Using Default Config", flush=True)
        return default_config

    return None


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


def detect_is_style(train_data_dir):
    """Detect if the dataset contains style-based prompts using balanced logic."""
    try:
        sub_dirs = [d for d in os.listdir(train_data_dir) if os.path.isdir(os.path.join(train_data_dir, d))]
        prompts = []
        
        for sub in sub_dirs:
            # Check all subdirectories starting with numbers (repeats)
            if "_" in sub and sub.split("_")[0].isdigit():
                prompts_path = os.path.join(train_data_dir, sub)
                for file in os.listdir(prompts_path):
                    if file.endswith(".txt"):
                        with open(os.path.join(prompts_path, file), "r", encoding='utf-8') as f:
                            prompts.append(f.read().strip().lower())
        
        if not prompts:
            return False

        # Keywords that strongly indicate a PERSON task
        person_keywords = ["man", "woman", "girl", "boy", "person", "lady", "gentleman", "male", "female", "guy", "people", "portrait"]
        
        # Keywords that strongly indicate a STYLE task
        style_keywords = [
            "painting", "art style", "sketch", "comic", "cyberpunk", "steampunk", "impressionist", 
            "minimalist", "gothic", "pixel art", "anime style", "3d render", "vector art", 
            "abstract", "illustration", "manga", "watercolor", "digital art", "pop art"
        ]
        
        person_count = 0
        style_count = 0
        
        for prompt in prompts:
            is_person = any(re.search(rf"\b{word}\b", prompt) for word in person_keywords)
            is_style = any(re.search(rf"\b{word}\b", prompt) for word in style_keywords)
            
            if is_person: person_count += 1
            if is_style: style_count += 1
        
        prompt_total = len(prompts)
        person_ratio = person_count / prompt_total
        style_ratio = style_count / prompt_total
        
        print(f"DEBUG_CLASSIFY: Person Ratio: {person_ratio:.2f}, Style Ratio: {style_ratio:.2f}", flush=True)

        # LOGIC: If person keywords are present in > 20% of prompts, it's almost certainly a PERSON task
        if person_ratio >= 0.20:
            return False
            
        # If it's not a person task and style keywords are present in > 25% of prompts, it's a STYLE task
        return style_ratio >= 0.25
        
    except Exception as e:
        print(f"Warning during style detection: {e}", flush=True)
        return False
def create_config(task_id, model_path, model_name, model_type, expected_repo_name, trigger_word, hours_to_complete=None):
    train_data_dir = os.path.join(train_cst.IMAGE_CONTAINER_IMAGES_PATH, task_id)
    
    # --- TASK CLASSIFICATION ---
    # 1. SMART DETECTION (Dataset-based - The Champion's Way)
    is_style = detect_is_style(train_data_dir)
    detection_method = "Dataset-driven"
    
    # 2. FALLBACK DETECTION (Metadata-based)
    if not is_style:
        meta_style = "style" in model_name.lower() or "style" in task_id.lower() or (expected_repo_name and "style" in expected_repo_name.lower())
        if meta_style:
            is_style = True
            detection_method = "Metadata-driven"
        else:
            detection_method = "Default (Person)"

    task_type = "style" if is_style else "person"
    print(f"DEBUG_TYPE: Task detected as [{task_type.upper()}] via {detection_method}", flush=True)

    dataset_size = count_images_in_directory(train_data_dir)
    is_ai_toolkit = model_type in [ImageModelType.Z_IMAGE.value, ImageModelType.QWEN_IMAGE.value]
    output_dir = train_paths.get_checkpoints_output_path(task_id, expected_repo_name or "output")
    
    # UNIFIED CONFIG RESOLUTION
    config_dir = os.path.join(script_dir, "core", "config")
    
    # TASK SPECIFIC TEMPLATE
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

    # DICTIONARY & MAPPING 
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
        350: {"network_dim": 128, "network_alpha": 128, "network_args": ["train_double_block_indices=all", "train_single_block_indices=all", "train_t5xxl=True"]},
        999: {"network_dim": 32, "network_alpha": 32, "network_args": ["conv_dim=32", "conv_alpha=32"]}
    }

    # Model Config ID
    if model_type == ImageModelType.Z_IMAGE.value:
        config_id = 999
    elif model_type == ImageModelType.FLUX.value:
        config_id = network_config_flux.get(model_name, 350)
    elif model_type == ImageModelType.QWEN_IMAGE.value:
        config_id = None # Fully handled by LRS/qwen.json
    else:
        target_dict = network_config_style if is_style else network_config_person
        config_id = target_dict.get(model_name, 235)

    model_params = config_mapping.get(config_id, config_mapping[235]) if config_id else {"network_dim": 32, "network_alpha": 32, "network_args": []}
    net_dim = model_params["network_dim"]

    print(f"[CONFIG SPESIFICATION - LAYER II] Model '{model_name}' Rank {net_dim}", flush=True)

    # --- LRS & OVERRIDES ---
    lrs_settings = None
    size_config = None
    
    lrs_config = load_lrs_config(model_type, is_style)
    if lrs_config:
        # Sanitize model name for robust hashing
        clean_model_name = model_name.strip().strip("'").strip('"')
        model_hash = hash_model(clean_model_name)
        lrs_settings = get_config_for_model(lrs_config, model_hash, dataset_size, clean_model_name)

    if dataset_size > 0:
        size_config = load_size_based_config(model_type, is_style, dataset_size)

    if is_ai_toolkit:
        with open(config_template_path, "r") as file:
            config = yaml.safe_load(file)
        
        if 'config' in config and 'process' in config['config']:
            for process in config['config']['process']:
                if 'model' in process:
                    # AI-Toolkit usually expects a directory containing the model or a repo ID
                    # If it's a path to a .safetensors file, get the directory
                    if model_path.endswith(".safetensors"):
                        process['model']['name_or_path'] = os.path.dirname(model_path)
                    else:
                        process['model']['name_or_path'] = model_path

                    # Follow Yaya-Simplified Logic (Strict Paths)
                    if model_type == ImageModelType.Z_IMAGE.value:
                        process['model']['assistant_lora_path'] = os.path.join(train_cst.HUGGINGFACE_CACHE_PATH, "zimage_turbo_training_adapter_v2.safetensors")
                        
                    if 'training_folder' in process:
                        if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
                        process['training_folder'] = output_dir
                
                if 'datasets' in process:
                    # AI-Toolkit expects images directly in folder_path.
                    rep = cst.DIFFUSION_SDXL_REPEATS if model_type == ImageModelType.SDXL.value else cst.DIFFUSION_FLUX_REPEATS
                    sub_pref = f"{rep}_"
                    dataset_path = train_data_dir
                    if os.path.exists(train_data_dir):
                        for d in os.listdir(train_data_dir):
                            if d.startswith(sub_pref) and os.path.isdir(os.path.join(train_data_dir, d)):
                                dataset_path = os.path.join(train_data_dir, d)
                                break
                    
                    for dataset in process['datasets']:
                        dataset['folder_path'] = dataset_path

                # --- ADVANCED AUTO-SCALING (JORDANSKY TUNING) ---
                if 'train' not in process: process['train'] = {}
                
                # Determine repeats factor for relative step calculation
                rep_factor = cst.DIFFUSION_SDXL_REPEATS if model_type == ImageModelType.SDXL.value else cst.DIFFUSION_FLUX_REPEATS
                
                def calculate_steps(epochs):
                    batch_size = process.get('train', {}).get('batch_size', 1)
                    if dataset_size == 0: return epochs
                    # Actual steps = Epochs * (Images * Repeats / Batch)
                    return int(epochs * (dataset_size * rep_factor / batch_size))

                # 1. APPLY AE (SIZE-BASED DEFAULTS)
                if size_config:
                    for key, value in size_config.items():
                        if key == "max_train_epochs": process['train']['steps'] = calculate_steps(value)
                        elif key == "optimizer_type": process['train']['optimizer'] = value
                        elif key in ["rank", "alpha"]:
                            block = 'network' if 'network' in process else 'adapter'
                            if block not in process: process[block] = {}
                            process[block][key if block == 'adapter' else ('linear' if key == 'rank' else 'linear_alpha')] = value
                        else: process['train'][key] = value

                # 2. APPLY LRS OVERRIDES
                if lrs_settings:
                    for key, value in lrs_settings.items():
                        if key in ["unet_lr", "text_encoder_lr", "learning_rate"]: process['train']['lr'] = value
                        elif key in ["optimizer_type", "optimizer"]: process['train']['optimizer'] = value
                        elif key in ["max_train_epochs", "steps"]:
                            process['train']['steps'] = calculate_steps(value) if key == "max_train_epochs" else value
                        elif key in ["rank", "alpha", "conv_rank", "conv_alpha"]:
                            # AI-Toolkit strict network block mapping
                            if 'network' not in process: process['network'] = {}
                            if key == "rank": process['network']['linear'] = value
                            elif key == "alpha": process['network']['linear_alpha'] = value
                            else: process['network'][key] = value # handles conv_rank, conv_alpha
                        elif key == "optimizer_args" and isinstance(value, list):
                            opt_params = {}
                            for item in value:
                                if "=" in item:
                                    k, v = item.split("=", 2)
                                    if v.lower() == "true": v = True
                                    elif v.lower() == "false": v = False
                                    else:
                                        try: v = float(v) if "." in v else int(v)
                                        except ValueError: pass
                                    opt_params[k.strip()] = v
                            process['train']['optimizer_params'] = opt_params
                        else: process['train'][key] = value


                if trigger_word:
                    process['trigger_word'] = trigger_word

                # 3. AUTO-ADAPTIVE STEPS (DURATION BASED) - JORDANSKY ENGINE
                if hours_to_complete and hours_to_complete > 0:
                    dynamic_steps = int(hours_to_complete * 800)
                    print(f"[AUTO-ADAPTIVE] Adjusting steps to {dynamic_steps} for {hours_to_complete}h duration (800 steps/h).", flush=True)
                    if 'train' in process:
                        process['train']['steps'] = dynamic_steps
        
        config_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}.yaml")
        save_config(config, config_path)
        print(f"Created ai-toolkit config at {config_path} with Auto-Scaling", flush=True)
        return config_path, output_dir
    else:
        with open(config_template_path, "r") as file:
            config = toml.load(file)

        config['pretrained_model_name_or_path'] = model_path
        
        # FLUX COMPONENT
        if model_type == "flux":
            print("\n[FLUX GOD MODE] Starting precision asset fingerprinting...", flush=True)
            
            # 1. HARD-PRIORITY
            std_paths = {
                'ae': "/cache/models/ae.safetensors",
                'clip_l': "/cache/models/clip_l.safetensors",
                't5xxl': "/cache/models/t5xxl.safetensors"
            }
            
            def set_flux_arg(k, v):
                config[k] = v
                if 'model_arguments' not in config: config['model_arguments'] = {}
                config['model_arguments'][k] = v

            for key, path in std_paths.items():
                if os.path.exists(path):
                    set_flux_arg(key, path)
                    print(f"   [VALIDATOR] Found {key} at {path}", flush=True)

            # 2. FALLBACK/DISCOVERY
            missing = [k for k in ['ae', 'clip_l', 't5xxl'] if not os.path.exists(config.get(k, ""))]
            if missing:
                def search_for_flux_files():
                    search_bases = ["/cache/models", "/app/models", "/app/flux", "/workspace/models", os.path.dirname(model_path)]
                    found = []
                    for b_dir in search_bases:
                        if not os.path.exists(b_dir): continue
                        for root, _, files in os.walk(b_dir):
                            for f in files:
                                if f.endswith(".safetensors"):
                                    p = os.path.join(root, f)
                                    sz = os.path.getsize(p) / (1024**3)
                                    found.append({"path": p, "size": sz, "root": root})
                    return found

                files_found = search_for_flux_files()

                if 'ae' in missing:
                    path = find_surgical(files_found, "AE", 0.3, 0.45, must_contain="ae")
                    if path: set_flux_arg('ae', path)
                if 'clip_l' in missing:
                    path = find_surgical(files_found, "CLIP", 0.2, 0.45) or "/app/models/clip_l.safetensors"
                    if path: set_flux_arg('clip_l', path)
                if 't5xxl' in missing:
                    path = find_surgical(files_found, "T5", 4.3, 11.0, avoid=["part", "of-", "shard"])
                    if path: set_flux_arg('t5xxl', path)

            # 3. CRITICAL COHERENCE
            final_ae = config.get('ae')
            final_clip = config.get('clip_l')
            final_t5 = config.get('t5xxl')

            if not (final_ae and final_clip and final_t5 and os.path.exists(final_clip)):
                print("[GOD MODE FAILURE] Missing vital FLUX components!", flush=True)
                print(f"   Current Resolution: AE={final_ae}, CLIP={final_clip}, T5={final_t5}", flush=True)

            print(f"[ASSET SYNC] AE: {final_ae}, CLIP: {final_clip}, T5: {final_t5}", flush=True)

        config['train_data_dir'] = train_data_dir
        if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
        config['output_dir'] = output_dir

        # Apply Overrides (Priority: Autoepoch < LRS)
        section_map = {}
        
        # FLUX Specific Direct Overrides (G.O.D Style - All Flat)
        if model_type == "flux":
            section_map["unet_lr"] = (None, "unet_lr")
            section_map["text_encoder_lr"] = (None, "text_encoder_lr")
            section_map["optimizer_type"] = (None, "optimizer_type")
            section_map["optimizer_args"] = (None, "optimizer_args")

        # Apply Overrides (Priority: Autoepoch < LRS)
        # Order: Apply size_config first, then LRS to let LRS win.
        configs_to_apply = []
        if size_config:
            configs_to_apply.append(("Size-Based", size_config))
        if lrs_settings:
            configs_to_apply.append(("LRS-Override", lrs_settings))
            
        for name, cfg in configs_to_apply:
            for key, value in cfg.items():
                if key in section_map:
                    sec, target = section_map[key]
                    if sec:
                        if sec not in config: config[sec] = {}
                        config[sec][target] = value
                    else:
                        config[target] = value
                else:
                    # Direct injection for root keys (max_train_epochs, train_batch_size, etc.)
                    if key == "max_train_epochs":
                        print(f"   [OVERRIDE] Setting {key} = {value} from {name}", flush=True)
                        if "max_train_steps" in config:
                            del config["max_train_steps"]
                    config[key] = value

        config_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}.toml")
        save_config_toml(config, config_path)
        print(f"Created config at {config_path}", flush=True)
        return config_path, output_dir



def run_training(model_type, config_path, output_dir, hours_to_complete=None, script_start_time=None):
    print(f"Starting training with config: {config_path}", flush=True)
    
    # Set aggressive safety margin: 5 minutes (300 seconds) before the absolute deadline
    if hours_to_complete and script_start_time:
        safe_duration = (hours_to_complete * 3600) - 300
        deadline = script_start_time + safe_duration
        print(f"[SAFE-STOP] Total Task duration: {hours_to_complete}h. Absolute deadline set for {safe_duration/60:.1f} minutes (5m buffer).", flush=True)
    else:
        deadline = None

    is_ai_toolkit = model_type in [ImageModelType.Z_IMAGE.value, ImageModelType.QWEN_IMAGE.value]
    env = os.environ.copy()
    env.update({
        "HF_HOME": train_cst.HUGGINGFACE_CACHE_PATH,
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
        process = subprocess.Popen(
            training_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env
        )

        for line in process.stdout:
            print(line, end="", flush=True)
            
            # Check if we've reached the safety deadline
            if deadline and time.time() > deadline:
                print(f"\n[SAFE-STOP] Approaching task deadline! Terminating training gracefully to allow upload...", flush=True)
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                break

        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, training_command)

        print("Training subprocess completed successfully.", flush=True)

    except subprocess.CalledProcessError as e:
        print("Training subprocess failed!", flush=True)
        print(f"Exit Code: {e.returncode}", flush=True)
        print(f"Command: {' '.join(e.cmd) if isinstance(e.cmd, list) else e.cmd}", flush=True)
        raise RuntimeError(f"Training subprocess failed with exit code {e.returncode}")

    # --- FIX: MOVE FILE IF SAVED IN WRONG LOCATION ---
    if output_dir:
        try:
            default_loc = "/app/checkpoints/last.safetensors"
            if os.path.exists(default_loc):
                print(f"[FIX] Moving checkpoint from {default_loc} to {output_dir}", flush=True)
                os.makedirs(output_dir, exist_ok=True)
                shutil.move(default_loc, os.path.join(output_dir, "last.safetensors"))
                print(f"[FIX] Successfully moved to {output_dir}/last.safetensors", flush=True)
        except Exception as e:
            print(f"[FIX] Error moving checkpoint: {e}", flush=True)
    # ------------------------------------------------

def hash_model(model: str) -> str:
    model_bytes = model.encode('utf-8')
    hashed = hashlib.sha256(model_bytes).hexdigest()
    return hashed 

async def main():
    script_start_time = time.time()
    print("--------------------------------------------------", flush=True)
    print("ONLY NINJA CAN STOP ME NOW", flush=True)
    print("--------------------------------------------------", flush=True)
    print("---STARTING TRAINING ---", flush=True)
    # PARSE COMMAND LINE ARGUMENTS
    parser = argparse.ArgumentParser(description="Image Model Training Script")
    parser.add_argument("--task-id", required=True, help="Task ID")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--dataset-zip", required=True, help="Link to dataset zip file")
    parser.add_argument("--model-type", required=True, choices=["sdxl", "flux", "qwen-image", "z-image"], help="Model type")
    parser.add_argument("--expected-repo-name", help="Expected repository name")
    parser.add_argument("--trigger-word", help="Trigger word for the training")
    parser.add_argument("--hours-to-complete", type=float, required=True, help="Number of hours to complete the task")
    args = parser.parse_args()

    os.makedirs(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, exist_ok=True)
    os.makedirs(train_cst.IMAGE_CONTAINER_IMAGES_PATH, exist_ok=True)

    model_path = get_model_path(train_paths.get_image_base_model_path(args.model))

    print("Preparing dataset...", flush=True)

    prepare_dataset(
        training_images_zip_path=train_paths.get_image_training_zip_save_path(args.task_id),
        training_images_repeat=cst.DIFFUSION_SDXL_REPEATS if args.model_type == ImageModelType.SDXL.value else cst.DIFFUSION_FLUX_REPEATS,
        instance_prompt=cst.DIFFUSION_DEFAULT_INSTANCE_PROMPT,
        class_prompt=cst.DIFFUSION_DEFAULT_CLASS_PROMPT,
        job_id=args.task_id,
        output_dir=train_cst.IMAGE_CONTAINER_IMAGES_PATH
    )

    config_path, output_dir = create_config(
        args.task_id,
        model_path,
        args.model,
        args.model_type,
        args.expected_repo_name,
        args.trigger_word,
        hours_to_complete=args.hours_to_complete
    )

    run_training(args.model_type, config_path, output_dir, args.hours_to_complete, script_start_time)


if __name__ == "__main__":
    asyncio.run(main())
