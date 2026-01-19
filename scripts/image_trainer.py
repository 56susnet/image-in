#!/usr/bin/env python3

"""
Jordansky
"""

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import re
import time
import yaml
import toml

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
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

def load_size_based_config(model_type: str, is_style: bool, dataset_size: int) -> dict:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, "autoepoch") # Point to autoepoch dir
    
    if model_type == "flux":
        config_file = os.path.join(config_dir, "a-epochflux.json")
    elif model_type == "qwen-image":
        config_file = os.path.join(config_dir, "a-epochqwen.json")
    elif model_type == "z-image":
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
    if dataset_size <= 15:
        return "small"
    elif dataset_size <= 35:
        return "medium"
    else:
        return "large"

def get_config_for_model(lrs_config: dict, model_name: str, dataset_size: int = None) -> dict:
    if not isinstance(lrs_config, dict):
        return None

    data = lrs_config.get("data")
    default_config = lrs_config.get("default", {})

    if isinstance(data, dict) and model_name in data:
        model_config = data.get(model_name)
        
        # If dataset_size provided and model_config has size categories, merge them
        if dataset_size is not None and isinstance(model_config, dict):
            size_category = get_dataset_size_category(dataset_size)
            
            # Check if model_config has size-specific settings
            if size_category in model_config:
                size_specific_config = model_config.get(size_category, {})
                # Merge: default → model_config (non-size keys) → size_specific
                base_model_config = {k: v for k, v in model_config.items() if k not in ["small", "medium", "large"]}
                merged = merge_model_config(default_config, base_model_config)
                return merge_model_config(merged, size_specific_config)
        
        return merge_model_config(default_config, model_config)

    if default_config:
        return default_config

    return None

def load_lrs_config(model_type: str, is_style: bool) -> dict:
    script_dir = os.path.dirname(os.path.abspath(__file__))
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

def create_config(task_id, model_path, model_name, model_type, expected_repo_name, trigger_word):
    is_style = "style" in model_name.lower() or "style" in task_id.lower()
    train_data_dir = os.path.join(train_cst.IMAGE_CONTAINER_IMAGES_PATH, task_id)
    
    # --- UNIFIED CONFIG RESOLUTION (PERSIMPLE) ---
    config_dir = os.path.join(script_dir, "core", "config")
    task_type = "style" if is_style else "person"
    
    # Priority: Task-specific > Base, TOML > YAML
    potential_templates = [
        f"base_diffusion_{model_type}_{task_type}.toml",
        f"base_diffusion_{model_type}_{task_type}.yaml",
        f"base_diffusion_{model_type}.toml",
        f"base_diffusion_{model_type}.yaml",
        f"base_{model_type}.yaml" # Legacy fallback
    ]
    
    config_template_path = None
    for template_name in potential_templates:
        found_path = os.path.join(config_dir, template_name)
        if os.path.exists(found_path):
            config_template_path = found_path
            break
    
    if not config_template_path:
        raise FileNotFoundError(f"Could not find a valid config template for {model_type} in {config_dir}")

    # --- LAYER 2: DICTIONARY & MAPPING (CHAMPION LOGIC) ---
    network_config_person = {
        "stabilityai/stable-diffusion-xl-base-1.0": 235,
        "Lykon/dreamshaper-xl-1-0": 235,
        "Lykon/art-diffusion-xl-0.9": 235,
        "SG161222/RealVisXL_V4.0": 467,
        "stablediffusionapi/protovision-xl-v6.6": 235,
        "stablediffusionapi/omnium-sdxl": 235,
        "GraydientPlatformAPI/realism-engine2-xl": 235,
        "GraydientPlatformAPI/albedobase2-xl": 467,
        "KBlueLeaf/Kohaku-XL-Zeta": 235,
        "John6666/hassaku-xl-illustrious-v10style-sdxl": 228,
        "John6666/nova-anime-xl-pony-v5-sdxl": 235,
        "cagliostrolab/animagine-xl-4.0": 699,
        "dataautogpt3/CALAMITY": 235,
        "dataautogpt3/ProteusSigma": 235,
        "dataautogpt3/ProteusV0.5": 467,
        "dataautogpt3/TempestV0.1": 500,
        "ehristoforu/Visionix-alpha": 235,
        "femboysLover/RealisticStockPhoto-fp16": 467,
        "fluently/Fluently-XL-Final": 228,
        "mann-e/Mann-E_Dreams": 456,
        "misri/leosamsHelloworldXL_helloworldXL70": 235,
        "misri/zavychromaxl_v90": 235,
        "openart-custom/DynaVisionXL": 228,
        "recoilme/colorfulxl": 228,
        "zenless-lab/sdxl-aam-xl-anime-mix": 456,
        "zenless-lab/sdxl-anima-pencil-xl-v5": 228,
        "zenless-lab/sdxl-anything-xl": 228,
        "zenless-lab/sdxl-blue-pencil-xl-v7": 467,
        "Corcelio/mobius": 228,
        "GHArt/Lah_Mysterious_SDXL_V4.0_xl_fp16": 235,
        "OnomaAIResearch/Illustrious-xl-early-release-v0": 228,
        "bghira/terminus-xl-velocity-v2": 235,
        "ifmain/UltraReal_Fine-Tune": 467
    }

    network_config_style = {
        "stabilityai/stable-diffusion-xl-base-1.0": 235,
        "Lykon/dreamshaper-xl-1-0": 235,
        "Lykon/art-diffusion-xl-0.9": 235,
        "SG161222/RealVisXL_V4.0": 235,
        "stablediffusionapi/protovision-xl-v6.6": 235,
        "stablediffusionapi/omnium-sdxl": 235,
        "GraydientPlatformAPI/realism-engine2-xl": 235,
        "GraydientPlatformAPI/albedobase2-xl": 235,
        "KBlueLeaf/Kohaku-XL-Zeta": 235,
        "John6666/hassaku-xl-illustrious-v10style-sdxl": 235,
        "John6666/nova-anime-xl-pony-v5-sdxl": 235,
        "cagliostrolab/animagine-xl-4.0": 235,
        "dataautogpt3/CALAMITY": 235,
        "dataautogpt3/ProteusSigma": 235,
        "dataautogpt3/ProteusV0.5": 235,
        "dataautogpt3/TempestV0.1": 500,
        "ehristoforu/Visionix-alpha": 235,
        "femboysLover/RealisticStockPhoto-fp16": 235,
        "fluently/Fluently-XL-Final": 235,
        "mann-e/Mann-E_Dreams": 235,
        "misri/leosamsHelloworldXL_helloworldXL70": 235,
        "misri/zavychromaxl_v90": 235,
        "openart-custom/DynaVisionXL": 235,
        "recoilme/colorfulxl": 235,
        "zenless-lab/sdxl-aam-xl-anime-mix": 235,
        "zenless-lab/sdxl-anima-pencil-xl-v5": 235,
        "zenless-lab/sdxl-anything-xl": 235,
        "zenless-lab/sdxl-blue-pencil-xl-v7": 235,
        "Corcelio/mobius": 235,
        "GHArt/Lah_Mysterious_SDXL_V4.0_xl_fp16": 235,
        "OnomaAIResearch/Illustrious-xl-early-release-v0": 235,
        "bghira/terminus-xl-velocity-v2": 235,
        "ifmain/UltraReal_Fine-Tune": 235
    }

    network_config_flux = {
        "dataautogpt3/FLUX-MonochromeManga": 350,
        "mikeyandfriends/PixelWave_FLUX.1-dev_03": 350,
        "rayonlabs/FLUX.1-dev": 350,
        "mhnakif/fluxunchained-dev": 350
    }

    network_config_qwen = {
        "gradients-io-tournaments/Qwen-Image": 888,
        "gradients-io-tournaments/Qwen-Image-Jib-Mix": 888
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
        888: {"network_dim": 96, "network_alpha": 96, "network_args": []},
        999: {"network_dim": 32, "network_alpha": 32, "network_args": ["conv_dim=32", "conv_alpha=32"]}
    }

    # Determine Model Config ID
    if model_type == "z-image":
        config_id = 999
    elif model_type == "flux":
        config_id = network_config_flux.get(model_name, 350)
    elif model_type == "qwen-image":
        config_id = network_config_qwen.get(model_name, 888)
    else:
        target_dict = network_config_style if is_style else network_config_person
        config_id = target_dict.get(model_name, 235)

    model_params = config_mapping.get(config_id, config_mapping[235])
    net_dim = model_params["network_dim"]
    net_alpha = model_params["network_alpha"]
    net_args = model_params["network_args"]

    print(f"⚡ JORDANSKY LAYER 2: Model '{model_name}' Rank {net_dim}", flush=True)

    is_ai_toolkit = model_type in [ImageModelType.Z_IMAGE.value, ImageModelType.QWEN_IMAGE.value]
    
    # --- PREPARE OVERRIDES (LRS & AUTOEPOCH) ---
    lrs_settings = None
    size_config = None
    
    lrs_config = load_lrs_config(model_type, is_style)
    if lrs_config:
        model_hash = hash_model(model_name)
        dataset_size = count_images_in_directory(train_data_dir)
        lrs_settings = get_config_for_model(lrs_config, model_hash, dataset_size)

    dataset_size = count_images_in_directory(train_data_dir)
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
                        
                    elif model_type == ImageModelType.QWEN_IMAGE.value:
                        # Qwen quantization path is handled via base template or LRS
                        pass
                        
                    if 'training_folder' in process:
                        output_dir = train_paths.get_checkpoints_output_path(task_id, expected_repo_name or "output")
                        if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
                        process['training_folder'] = output_dir
                
                if 'datasets' in process:
                    for dataset in process['datasets']:
                        # AI-Toolkit expects images directly in folder_path.
                        # prepare_dataset creates a subfolder like "1_lora style"
                        rep = cst.DIFFUSION_SDXL_REPEATS if model_type == ImageModelType.SDXL.value else cst.DIFFUSION_FLUX_REPEATS
                        sub_pref = f"{rep}_"
                        dataset_path = train_data_dir
                        if os.path.exists(train_data_dir):
                            for d in os.listdir(train_data_dir):
                                if d.startswith(sub_pref) and os.path.isdir(os.path.join(train_data_dir, d)):
                                    dataset_path = os.path.join(train_data_dir, d)
                                    break
                        dataset['folder_path'] = dataset_path

                # --- ADVANCED AUTO-SCALING (JORDANSKY TUNING) ---
                if 'train' not in process: process['train'] = {}
                
                def calculate_steps(epochs):
                    batch_size = process.get('train', {}).get('batch_size', 1)
                    dataset_size = count_images_in_directory(train_data_dir)
                    if dataset_size == 0: return epochs
                    return int(epochs * (dataset_size / batch_size))

                # 1. Apply Autoepoch (Size-based defaults)
                if size_config:
                    for key, value in size_config.items():
                        if key == "max_train_epochs": process['train']['steps'] = calculate_steps(value)
                        elif key == "optimizer_type": process['train']['optimizer'] = value
                        elif key in ["rank", "alpha"]:
                            block = 'network' if 'network' in process else 'adapter'
                            if block not in process: process[block] = {}
                            process[block][key if block == 'adapter' else ('linear' if key == 'rank' else 'linear_alpha')] = value
                        else: process['train'][key] = value

                # 2. Apply LRS Override (Task-specific precision)
                if lrs_settings:
                    for key, value in lrs_settings.items():
                        if key in ["unet_lr", "text_encoder_lr", "learning_rate"]: process['train']['lr'] = value
                        elif key in ["optimizer_type", "optimizer"]: process['train']['optimizer'] = value
                        elif key in ["max_train_epochs", "steps"]:
                            process['train']['steps'] = calculate_steps(value) if key == "max_train_epochs" else value
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
        
        config_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}.yaml")
        save_config(config, config_path)
        print(f"Created ai-toolkit config at {config_path} with Auto-Scaling", flush=True)
        return config_path
    else:
        with open(config_template_path, "r") as file:
            config = toml.load(file)

        config['pretrained_model_name_or_path'] = model_path
        
        # FLUX Component Auto-Pathing (G.O.D ALIGNMENT)
        if model_type == "flux":
            print("\n[FLUX GOD MODE] Starting precision asset fingerprinting...", flush=True)
            
            # 1. HARD-PRIORITY: Standard Validator/GOD Paths
            # Alignment: G.O.D often places them in /cache/models or /app/flux
            std_paths = {
                'ae': "/cache/models/ae.safetensors",
                'clip_l': "/cache/models/clip_l.safetensors",
                't5xxl': "/cache/models/t5xxl.safetensors"
            }
            
            # G.O.D Template compatibility: Ensure keys exist at top level or in model_arguments
            def set_flux_arg(k, v):
                config[k] = v # GOD style (Flat)
                if 'model_arguments' not in config: config['model_arguments'] = {}
                config['model_arguments'][k] = v # Legacy style

            for key, path in std_paths.items():
                if os.path.exists(path):
                    set_flux_arg(key, path)
                    print(f"   [VALIDATOR] Found {key} at {path}", flush=True)

            # 2. FALLBACK/DISCOVERY: If any components are still missing or invalid
            missing = [k for k in ['ae', 'clip_l', 't5xxl'] if not os.path.exists(config.get(k, ""))]
            if missing:
                search_bases = ["/cache/models", "/app/models", "/app/flux", "/workspace/models", os.path.dirname(model_path)]
                files_found = []
                for b_dir in search_bases:
                    if not os.path.exists(b_dir): continue
                    for root, _, files in os.walk(b_dir):
                        for f in files:
                            if f.endswith(".safetensors"):
                                p = os.path.join(root, f)
                                sz = os.path.getsize(p) / (1024**3)
                                files_found.append({"path": p, "size": sz, "root": root})

                def find_surgical(name, golden_min, golden_max, must_contain=None, avoid=["part", "of-", "sharded"]):
                    matches = []
                    for entry in files_found:
                        p, sz, root = entry["path"], entry["size"], entry["root"]
                        if golden_min <= sz <= golden_max:
                            if avoid and any(a in p.lower() for a in avoid): continue
                            if must_contain and must_contain not in p.lower(): continue
                            score = 0
                            if "flux" in p.lower() or "flux" in root.lower(): score += 100
                            if name.lower() in p.lower() or name.lower() in root.lower(): score += 50
                            matches.append((score, sz, p))
                    if matches:
                        matches.sort(key=lambda x: (-x[0], -x[1]))
                        return matches[0][2]
                    return None

                if 'ae' in missing:
                    path = find_surgical("AE", 0.3, 0.45, must_contain="ae")
                    if path: set_flux_arg('ae', path)
                if 'clip_l' in missing:
                    path = find_surgical("CLIP", 0.2, 0.45) or "/app/models/clip_l.safetensors"
                    if path: set_flux_arg('clip_l', path)
                if 't5xxl' in missing:
                    path = find_surgical("T5", 4.3, 11.0, avoid=["part", "of-", "shard"])
                    if path: set_flux_arg('t5xxl', path)

            # 3. CRITICAL COHERENCE CHECK
            final_ae = config.get('ae')
            final_clip = config.get('clip_l')
            final_t5 = config.get('t5xxl')

            if not (final_ae and final_clip and final_t5 and os.path.exists(final_clip)):
                print("❌ [GOD MODE FAILURE] Missing vital FLUX components!", flush=True)
                print(f"   Current Resolution: AE={final_ae}, CLIP={final_clip}, T5={final_t5}", flush=True)

            print(f"✅ [ASSET SYNC] AE: {final_ae}, CLIP: {final_clip}, T5: {final_t5}", flush=True)

        config['train_data_dir'] = train_data_dir
        output_dir = train_paths.get_checkpoints_output_path(task_id, expected_repo_name)
        if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
        config['output_dir'] = output_dir

        # Apply Overrides (Priority: Autoepoch < LRS)
        # We now use FLAT injection for everything to ensure reliable overwriting
        section_map = {}
        
        # FLUX Specific Direct Overrides (G.O.D Style - All Flat)
        if model_type == "flux":
            section_map["unet_lr"] = (None, "unet_lr")
            section_map["text_encoder_lr"] = (None, "text_encoder_lr")
            section_map["optimizer_type"] = (None, "optimizer_type")
            section_map["optimizer_args"] = (None, "optimizer_args")

        # Apply Overrides (Priority: Autoepoch < LRS)
        configs_to_apply = [size_config, lrs_settings]
        
        for cfg in configs_to_apply:
            if not cfg:
                continue
                
            for key, value in cfg.items():
                if key in section_map:
                    sec, target = section_map[key]
                    if sec:
                        if sec not in config: config[sec] = {}
                        config[sec][target] = value
                    else:
                        config[target] = value # Flat injection
                else:
                    # Direct injection for root keys (max_train_epochs, train_batch_size, etc.)
                    config[key] = value
                    
                    # COHERENCE: If we set max_train_epochs, clear max_train_steps to avoid conflicts
                    if key == "max_train_epochs":
                        if "max_train_steps" in config:
                            print(f"   [COHERENCE] Clearing max_train_steps to prioritize epoch scaling ({value} epochs)", flush=True)
                            del config["max_train_steps"]

        config_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}.toml")
        save_config_toml(config, config_path)
        print(f"Created config at {config_path}", flush=True)
        return config_path

def ensure_offline_tokenizers():
    """Bridge Hugging Face HUB structure to SD-Scripts FLAT structure.
    Surgically find snapshots and create flat folders for offline loading.
    """
    import shutil
    cache_root = train_cst.HUGGINGFACE_CACHE_PATH
    
    # Mapping for SDXL tokenizers
    mapping = {
        "openai/clip-vit-large-patch14": "openai_clip-vit-large-patch14",
        "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k": "laion_CLIP-ViT-bigG-14-laion2B-39B-b160k",
        "google/t5-v1_1-xxl": "google_t5-v1_1-xxl"
    }
    
    print(f"🔍 [OFFLINE SYNC] Scanning {cache_root} for tokenizers...", flush=True)
    
    # Check what's actually in there
    if not os.path.exists(cache_root):
        print(f"❌ [OFFLINE SYNC] ERROR: Cache root {cache_root} does not exist!", flush=True)
        return

    # Helper to find the actual snapshot directory recursively
    def find_snapshot_dir(target_repo):
        search_slug = target_repo.replace("/", "--")
        for root, dirs, files in os.walk(cache_root):
            if search_slug in root and "snapshots" in root:
                # We are in the snapshots folder, return the first child directory (the hash)
                if dirs:
                    return os.path.join(root, dirs[0])
        return None

    for repo_id, flat_name in mapping.items():
        flat_path = os.path.join(cache_root, flat_name)
        
        # If already exists and has content, skip
        if os.path.exists(flat_path) and os.path.exists(os.path.join(flat_path, "config.json")):
            print(f"✅ [OFFLINE SYNC] {flat_name} already exists and is valid.", flush=True)
            continue
            
        print(f"🕵️ [OFFLINE SYNC] Searching for {repo_id} snapshot...", flush=True)
        src_dir = find_snapshot_dir(repo_id)
        
        if src_dir and os.path.exists(src_dir):
            print(f"🔗 [OFFLINE SYNC] Found snapshot at {src_dir}. Replicating to {flat_path}...", flush=True)
            try:
                os.makedirs(flat_path, exist_ok=True)
                for item in os.listdir(src_dir):
                    s = os.path.join(src_dir, item)
                    d = os.path.join(flat_path, item)
                    if os.path.isfile(s):
                        shutil.copy2(s, d)
                print(f"✨ [OFFLINE SYNC] Successfully prepared {flat_name}", flush=True)
            except Exception as e:
                print(f"⚠️ [OFFLINE SYNC] Error replicating {repo_id}: {e}", flush=True)
        else:
            print(f"❌ [OFFLINE SYNC] Could not find any snapshots for {repo_id} in {cache_root}", flush=True)

    # --- QWEN SPECIFIC FIX: Ensure 'transformer' subfolder exists ---
    qwen_root = "/cache/models/gradients-io-tournaments--Qwen-Image"
    transformer_dir = os.path.join(qwen_root, "transformer")
    
    # Check if we have the model but not the folder structure
    if os.path.exists(qwen_root):
        # Look for the .bin or .safetensors file in root
        found_model = None
        for f in os.listdir(qwen_root):
             if f.endswith(".bin") or f.endswith(".safetensors"):
                 found_model = f
                 break
        
        if found_model and not os.path.exists(transformer_dir):
            print(f"🔧 [OFFLINE FIX] Correcting Qwen structure. Moving {found_model} to transformer/...", flush=True)
            try:
                os.makedirs(transformer_dir, exist_ok=True)
                # Copy config.json too if exists
                if os.path.exists(os.path.join(qwen_root, "config.json")):
                    shutil.copy2(os.path.join(qwen_root, "config.json"), os.path.join(transformer_dir, "config.json"))
                
                # Copy the model file
                # Use COPY instead of MOVE to be safe if original is symlinked
                shutil.copy2(os.path.join(qwen_root, found_model), os.path.join(transformer_dir, "diffusion_pytorch_model.bin"))
                print("✅ [OFFLINE FIX] Qwen structure corrected.", flush=True)
            except Exception as e:
                 print(f"⚠️ [OFFLINE FIX] Failed to restructure Qwen: {e}", flush=True)

def run_training(model_type, config_path):
    # Ensure tokenizers are ready for offline sd-scripts
    # Ensure tokenizers are ready for offline sd-scripts
    if model_type in ["sdxl", "flux", "z-image", "qwen-image"]:
        ensure_offline_tokenizers()
    
    print(f"Starting training with config: {config_path}", flush=True)
    
    is_ai_toolkit = model_type in [ImageModelType.Z_IMAGE.value, ImageModelType.QWEN_IMAGE.value]
    
    if is_ai_toolkit:
        training_command = [
            "python3",
            "/app/ai-toolkit/run.py",
            config_path
        ]
        # AI-Toolkit Offline Guard
        env = os.environ.copy()
        env["HF_DATASETS_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_HUB_OFFLINE"] = "1"
    else:
        if model_type == "sdxl":
            training_command = [
                "accelerate", "launch",
                "--dynamo_backend", "no",
                "--dynamo_mode", "default",
                "--mixed_precision", "bf16",
                "--num_processes", "1",
                "--num_machines", "1",
                "--num_cpu_threads_per_process", "2",
                f"/app/sd-scripts/{model_type}_train_network.py",
                "--config_file", config_path,
                "--tokenizer_cache_dir", train_cst.HUGGINGFACE_CACHE_PATH
            ]
        elif model_type == "flux":
            training_command = [
                "accelerate", "launch",
                "--dynamo_backend", "no",
                "--dynamo_mode", "default",
                "--mixed_precision", "bf16",
                "--num_processes", "1",
                "--num_machines", "1",
                "--num_cpu_threads_per_process", "2",
                f"/app/sd-scripts/{model_type}_train_network.py",
                "--config_file", config_path,
                "--tokenizer_cache_dir", train_cst.HUGGINGFACE_CACHE_PATH
            ]
    
    try:
        env = os.environ.copy()
        env["HF_HOME"] = train_cst.HUGGINGFACE_CACHE_PATH
        env["PYTHONUNBUFFERED"] = "1"
        
        # --- OFFLINE HUB GUARD (CRITICAL) ---
        env["HF_DATASETS_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_HUB_OFFLINE"] = "1"

        print(f"🚀 Launching {model_type.upper()} training with command: {' '.join(training_command)}", flush=True)
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

        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, training_command)

        print("Training subprocess completed successfully.", flush=True)
        
        # --- FIX: MOVE FILE IF SAVED IN WRONG LOCATION ---
        try:
            # Parse config to get intended output directory
            intended_output_dir = None
            if config_path.endswith(".toml"):
                import toml
                with open(config_path, "r") as f:
                    c = toml.load(f)
                    intended_output_dir = c.get("output_dir")
            
            if intended_output_dir:
                default_loc = "/app/checkpoints/last.safetensors"
                if os.path.exists(default_loc):
                    print(f"[FIX] Moving checkpoint from {default_loc} to {intended_output_dir}", flush=True)
                    os.makedirs(intended_output_dir, exist_ok=True)
                    import shutil
                    shutil.move(default_loc, os.path.join(intended_output_dir, "last.safetensors"))
                    print(f"[FIX] Successfully moved to {intended_output_dir}/last.safetensors", flush=True)
        except Exception as e:
            print(f"[FIX] Error moving checkpoint: {e}", flush=True)
        # ------------------------------------------------

    except subprocess.CalledProcessError as e:
        print("Training subprocess failed!", flush=True)
        print(f"Exit Code: {e.returncode}", flush=True)
        print(f"Command: {' '.join(e.cmd) if isinstance(e.cmd, list) else e.cmd}", flush=True)
        raise RuntimeError(f"Training subprocess failed with exit code {e.returncode}")

def hash_model(model: str) -> str:
    model_bytes = model.encode('utf-8')
    hashed = hashlib.sha256(model_bytes).hexdigest()
    return hashed 

async def main():
    print("--------------------------------------------------", flush=True)
    print("🚀 EMPIRE STANDALONE TRAINER V2.0 (OFFLINE PATCHED)", flush=True)
    print("--------------------------------------------------", flush=True)
    print("---STARTING IMAGE TRAINING SCRIPT---", flush=True)
    # Parse command line arguments
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

    config_path = create_config(
        args.task_id,
        model_path,
        args.model,
        args.model_type,
        args.expected_repo_name,
        args.trigger_word,
    )

    run_training(args.model_type, config_path)


if __name__ == "__main__":
    asyncio.run(main())
