#!/bin/bash

TASK_ID="1c93dd95-2e89-48d9-813d-e0f521599cfd"
MODEL="gradients-io-tournaments/Qwen-Image"
DATASET_ZIP="https://gradients.s3.eu-north-1.amazonaws.com/dc9853fb35c40bd4_train_data.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVVZOOA7SA4UOFLPI%2F20251221%2Feu-north-1%2Fs3%2Faws4_request&X-Amz-Date=20251221T212609Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=865abddfccce78e1964b0abb468c3fc7a591019820d3a3024f7d4220757da588"
MODEL_TYPE="qwen-image"
EXPECTED_REPO_NAME="test_qwenimage-1"

HUGGINGFACE_USERNAME=""
HUGGINGFACE_TOKEN=""
LOCAL_FOLDER="/app/checkpoints/$TASK_ID/$EXPECTED_REPO_NAME"

CHECKPOINTS_DIR="$(pwd)/secure_checkpoints"
OUTPUTS_DIR="$(pwd)/outputs"
mkdir -p "$CHECKPOINTS_DIR"
chmod 700 "$CHECKPOINTS_DIR"
mkdir -p "$OUTPUTS_DIR"
chmod 700 "$OUTPUTS_DIR"

mkdir -p "$CHECKPOINTS_DIR/tmp" "$CHECKPOINTS_DIR/hf_cache" "$CHECKPOINTS_DIR/models" "$CHECKPOINTS_DIR/datasets" "$CHECKPOINTS_DIR/images" "$CHECKPOINTS_DIR/configs"

echo "Downloading model and dataset..."
docker run --rm --volume "$CHECKPOINTS_DIR:/cache:rw" --env TMPDIR=/cache/tmp --env HF_HOME=/cache/hf_cache --env TRANSFORMERS_CACHE=/cache/hf_cache --name downloader-image trainer-downloader --task-id "$TASK_ID" --model "$MODEL" --dataset "$DATASET_ZIP" --task-type "ImageTask" --model-type "$MODEL_TYPE" 

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
echo "Starting image training..."
docker run --rm --gpus all   --security-opt=no-new-privileges   --cap-drop=ALL   --memory=32g   --cpus=8   --network none   --env TRANSFORMERS_CACHE=/cache/hf_cache   --volume "$CHECKPOINTS_DIR:/cache:rw"   --volume "$OUTPUTS_DIR:/app/checkpoints/:rw"   --volume "$SCRIPT_DIR/../scripts:/workspace/scripts:ro"   --volume "$SCRIPT_DIR/../scripts/core:/workspace/core:ro"   --name image-trainer-example   standalone-image-toolkit-trainer   --task-id "$TASK_ID"   --model "$MODEL"   --dataset-zip "$DATASET_ZIP"   --model-type "$MODEL_TYPE"   --expected-repo-name "$EXPECTED_REPO_NAME"   --hours-to-complete 1


echo "Uploading model to HuggingFace..."
docker run --rm --gpus all --volume "$CHECKPOINTS_DIR:/cache:rw" --volume "$OUTPUTS_DIR:/app/checkpoints/:rw" --env TMPDIR=/cache/tmp --env HF_HOME=/cache/hf_cache --env HUGGINGFACE_TOKEN="$HUGGINGFACE_TOKEN" --env HUGGINGFACE_USERNAME="$HUGGINGFACE_USERNAME" --env TASK_ID="$TASK_ID" --env EXPECTED_REPO_NAME="$EXPECTED_REPO_NAME" --env LOCAL_FOLDER="$LOCAL_FOLDER" --env HF_REPO_SUBFOLDER="checkpoints" --name hf-uploader hf-uploader
