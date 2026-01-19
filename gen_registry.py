import hashlib

models = [
    "cagliostrolab/animagine-xl-4.0",
    "misri/leosamsHelloworldXL_helloworldXL70",
    "dataautogpt3/TempestV0.1",
    "SG161222/RealVisXL_V4.0",
    "GraydientPlatformAPI/albedobase2-xl",
    "dataautogpt3/ProteusV0.5",
    "femboysLover/RealisticStockPhoto-fp16",
    "zenless-lab/sdxl-blue-pencil-xl-v7",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "Lykon/dreamshaper-xl-1-0",
    "Lykon/art-diffusion-xl-0.9",
    "stablediffusionapi/protovision-xl-v6.6",
    "stablediffusionapi/omnium-sdxl",
    "GraydientPlatformAPI/realism-engine2-xl",
    "KBlueLeaf/Kohaku-XL-Zeta",
    "John6666/hassaku-xl-illustrious-v10style-sdxl",
    "John6666/nova-anime-xl-pony-v5-sdxl",
    "dataautogpt3/CALAMITY",
    "dataautogpt3/ProteusSigma",
    "ehristoforu/Visionix-alpha",
    "fluently/Fluently-XL-Final",
    "mann-e/Mann-E_Dreams",
    "openart-custom/DynaVisionXL",
    "recoilme/colorfulxl",
    "zenless-lab/sdxl-aam-xl-anime-mix",
    "zenless-lab/sdxl-anima-pencil-xl-v5",
    "zenless-lab/sdxl-anything-xl",
    "Corcelio/mobius",
    "GHArt/Lah_Mysterious_SDXL_V4.0_xl_fp16",
    "OnomaAIResearch/Illustrious-xl-early-release-v0",
    "bghira/terminus-xl-velocity-v2",
    "ifmain/UltraReal_Fine-Tune"
]

for m in models:
    h = hashlib.sha256(m.encode('utf-8')).hexdigest()
    print(f"{m} : {h}")
