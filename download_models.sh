#!/usr/bin/env bash
set -e

MODELS_DIR="/home/steven/comfy/ComfyUI/models"
export https_proxy="http://127.0.0.1:12450/"
export http_proxy="http://127.0.0.1:12450/"
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"

download() {
    local url="$1"
    local dest_dir="$2"
    local filename="$3"

    mkdir -p "$dest_dir"
    local dest="$dest_dir/$filename"
    local tmp="${dest}.tmp"

    if [ -f "$dest" ] && [ -s "$dest" ]; then
        local size=$(du -h "$dest" | cut -f1)
        echo "  [SKIP] $filename ($size)"
        return 0
    fi

    echo "  [DOWNLOAD] $filename"
    echo "    → $dest_dir"

    # Use curl with resume support
    # -L: follow redirects, -C -: resume, -o: output, --progress-bar: show progress
    if [ -f "$tmp" ] && [ -s "$tmp" ]; then
        echo "  [RESUME] Continuing partial download..."
        curl -L -C - --progress-bar -o "$dest" "$url" 2>&1
    else
        curl -L --progress-bar -o "$dest" "$url" 2>&1
    fi
    local ret=$?

    if [ $ret -eq 0 ] && [ -s "$dest" ]; then
        local final_size=$(du -h "$dest" | cut -f1)
        echo "  [DONE] $filename ($final_size)"
    else
        echo "  [FAILED] $filename (exit: $ret)"
        rm -f "$tmp"
        return 1
    fi
}

echo "============================================"
echo "Downloading FLUX.1-DEV & LTX-2.3 Models for ComfyUI"
echo "Destination: $MODELS_DIR"
echo "============================================"
echo ""

echo "[1/8] FLUX.1-DEV Diffusion Model (~23GB)"
download \
    "https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev.safetensors" \
    "$MODELS_DIR/diffusion_models" \
    "flux1-dev.safetensors"

echo ""
echo "[2/8] CLIP-L Text Encoder (~235MB)"
download \
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors" \
    "$MODELS_DIR/text_encoders" \
    "clip_l.safetensors"

echo ""
echo "[3/8] T5-XXL FP16 Text Encoder (~9GB)"
download \
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors" \
    "$MODELS_DIR/text_encoders" \
    "t5xxl_fp16.safetensors"

echo ""
echo "[4/8] FLUX VAE (~320MB)"
download \
    "https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/resolve/main/split_files/vae/ae.safetensors" \
    "$MODELS_DIR/vae" \
    "ae.safetensors"

echo ""
echo "[5/8] LTX-2.3 Distilled Transformer mxfp8 (~11GB)"
download \
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/diffusion_models/ltx-2.3-22b-distilled-1.1_transformer_only_mxfp8_block32.safetensors" \
    "$MODELS_DIR/diffusion_models" \
    "ltx-2.3-22b-distilled-1.1_transformer_only_mxfp8_block32.safetensors"

echo ""
echo "[6/8] LTX-2.3 Video VAE (~200MB)"
download \
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_video_vae_bf16.safetensors" \
    "$MODELS_DIR/vae" \
    "LTX23_video_vae_bf16.safetensors"

echo ""
echo "[7/8] LTX-2.3 Text Projection (~100MB)"
download \
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/text_encoders/ltx-2.3_text_projection_bf16.safetensors" \
    "$MODELS_DIR/text_encoders" \
    "ltx-2.3_text_projection_bf16.safetensors"

echo ""
echo "[8/8] LTX-2.3 Distilled LoRA (~1.5GB)"
download \
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/loras/ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors" \
    "$MODELS_DIR/loras" \
    "ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors"

echo ""
echo "============================================"
echo "DOWNLOAD SUMMARY"
echo "============================================"
for f in \
    "$MODELS_DIR/diffusion_models/flux1-dev.safetensors" \
    "$MODELS_DIR/text_encoders/clip_l.safetensors" \
    "$MODELS_DIR/text_encoders/t5xxl_fp16.safetensors" \
    "$MODELS_DIR/vae/ae.safetensors" \
    "$MODELS_DIR/diffusion_models/ltx-2.3-22b-distilled-1.1_transformer_only_mxfp8_block32.safetensors" \
    "$MODELS_DIR/vae/LTX23_video_vae_bf16.safetensors" \
    "$MODELS_DIR/text_encoders/ltx-2.3_text_projection_bf16.safetensors" \
    "$MODELS_DIR/loras/ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors"; do
    if [ -f "$f" ] && [ -s "$f" ]; then
        size=$(du -h "$f" | cut -f1)
        echo "  ✅ $(basename $f) ($size)"
    else
        echo "  ❌ $(basename $f)"
    fi
done

echo ""
echo "Note: LTX-2.3 needs the Gemma 3 12B text encoder."
echo "Check: $MODELS_DIR/text_encoders/gemma-3-12b-it/"
