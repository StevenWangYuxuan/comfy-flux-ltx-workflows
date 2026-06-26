#!/usr/bin/env bash
set -e

MODELS_DIR="/home/steven/comfy/ComfyUI/models"
export https_proxy="http://127.0.0.1:12450/"
export http_proxy="http://127.0.0.1:12450/"

download() {
    local url="$1"
    local dest_dir="$2"
    local filename="$3"
    mkdir -p "$dest_dir"
    local dest="$dest_dir/$filename"
    if [ -f "$dest" ] && [ -s "$dest" ]; then
        local size=$(du -h "$dest" | cut -f1)
        echo "  [SKIP] $filename ($size)"
        return 0
    fi
    echo "  [DOWNLOAD] $filename → $dest_dir"
    curl -L --progress-bar -o "$dest" "$url" 2>&1
    local ret=$?
    if [ $ret -eq 0 ] && [ -s "$dest" ]; then
        local final_size=$(du -h "$dest" | cut -f1)
        echo "  [DONE] $filename ($final_size)"
    else
        echo "  [FAILED] $filename"
        return 1
    fi
}

echo "============================================"
echo "Downloading Additional Models"
echo "============================================"
echo ""

echo "[1/3] Gemma 3 12B FP8 Text Encoder (~11GB)"
download \
    "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors" \
    "$MODELS_DIR/text_encoders" \
    "gemma_3_12B_it_fp8_scaled.safetensors"

echo ""
echo "[2/3] LTX-2.3 Spatial Upscaler x2"
download \
    "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.0.safetensors" \
    "$MODELS_DIR/latent_upscale_models" \
    "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"

echo ""
echo "[3/3] LTX-2.3 Audio VAE"
download \
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_audio_vae_bf16.safetensors" \
    "$MODELS_DIR/vae" \
    "LTX23_audio_vae_bf16.safetensors"

echo ""
echo "Done!"
