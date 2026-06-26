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

    if [ -f "$dest" ] && [ -s "$dest" ]; then
        local size=$(du -h "$dest" | cut -f1)
        echo "  [SKIP] $filename ($size)"
        return 0
    fi

    echo "  [DOWNLOAD] $filename"
    echo "    → $dest_dir"
    curl -L --progress-bar -o "$dest" "$url" 2>&1
    local ret=$?

    if [ $ret -eq 0 ] && [ -s "$dest" ]; then
        local final_size=$(du -h "$dest" | cut -f1)
        echo "  [DONE] $filename ($final_size)"
    else
        echo "  [FAILED] $filename (exit: $ret)"
        rm -f "$dest"
        return 1
    fi
}

echo "============================================"
echo " ComfyUI 全部模型下载"
echo " 目标: $MODELS_DIR"
echo "============================================"

# ═══════════════════════════════════════════════
# FLUX.1-DEV — 写实文生图/图生图
# ═══════════════════════════════════════════════
echo ""
echo "── FLUX.1-DEV 写实模型 ──"

echo "[1/5] FLUX.1-DEV UNet fp8 (~23GB)"
download \
    "https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev.safetensors" \
    "$MODELS_DIR/diffusion_models" \
    "flux1-dev.safetensors"

echo "[2/5] CLIP-L 文本编码器 (~235MB)"
download \
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors" \
    "$MODELS_DIR/text_encoders" \
    "clip_l.safetensors"

echo "[3/5] T5-XXL FP16 文本编码器 (~9.2GB)"
download \
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors" \
    "$MODELS_DIR/text_encoders" \
    "t5xxl_fp16.safetensors"

echo "[4/5] T5-XXL FP8 文本编码器 (~4.6GB, v2推荐)"
download \
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors" \
    "$MODELS_DIR/text_encoders" \
    "t5xxl_fp8_e4m3fn.safetensors"

echo "[5/5] FLUX VAE (~320MB)"
download \
    "https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/resolve/main/split_files/vae/ae.safetensors" \
    "$MODELS_DIR/vae" \
    "ae.safetensors"

# ═══════════════════════════════════════════════
# LTX Video 2.3 — 图生视频/文生视频
# ═══════════════════════════════════════════════
echo ""
echo "── LTX Video 2.3 视频模型 ──"

echo "[1/6] LTX-2.3 UNet mxfp8 block32 (~23GB)"
download \
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/diffusion_models/ltx-2.3-22b-distilled-1.1_transformer_only_mxfp8_block32.safetensors" \
    "$MODELS_DIR/diffusion_models" \
    "ltx-2.3-22b-distilled-1.1_transformer_only_mxfp8_block32.safetensors"

echo "[2/6] LTX Video VAE (~1.4GB)"
download \
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_video_vae_bf16.safetensors" \
    "$MODELS_DIR/vae" \
    "LTX23_video_vae_bf16.safetensors"

echo "[3/6] LTX 文本投影层 → text_encoders/ (~2.2GB)"
download \
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/text_encoders/ltx-2.3_text_projection_bf16.safetensors" \
    "$MODELS_DIR/text_encoders" \
    "ltx-2.3_text_projection_bf16.safetensors"

echo "[4/6] LTX 文本投影层 → checkpoints/ (LTXAVTextEncoderLoader读这里!)"
download \
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/text_encoders/ltx-2.3_text_projection_bf16.safetensors" \
    "$MODELS_DIR/checkpoints" \
    "ltx-2.3_text_projection_bf16.safetensors"

echo "[5/6] Gemma 3 12B FP8 文本编码器 (~13GB)"
download \
    "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors" \
    "$MODELS_DIR/text_encoders" \
    "gemma_3_12B_it_fp8_scaled.safetensors"

echo "[6/6] LTX LoRA 动态增强 (~2.5GB)"
download \
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/loras/ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors" \
    "$MODELS_DIR/loras" \
    "ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors"

# ═══════════════════════════════════════════════
# SDXL 动漫 — Animagine XL 4.0
# ═══════════════════════════════════════════════
echo ""
echo "── SDXL 动漫模型 ──"

echo "[1/1] Animagine XL 4.0 动漫模型 (~6.5GB)"
download \
    "https://huggingface.co/cagliostrolab/animagine-xl-4.0/resolve/main/animagine-xl-4.0.safetensors" \
    "$MODELS_DIR/checkpoints" \
    "animagine-xl-4.0.safetensors"

# ═══════════════════════════════════════════════
# IP-Adapter — 人脸参考生成
# ═══════════════════════════════════════════════
echo ""
echo "── IP-Adapter 人脸参考 ──"

echo "[1/2] IP-Adapter SDXL Plus (~809MB)"
download \
    "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors" \
    "$MODELS_DIR/ipadapter" \
    "ip-adapter-plus_sdxl_vit-h.safetensors"

echo "[2/2] CLIP Vision ViT-H (~2.4GB)"
download \
    "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors" \
    "$MODELS_DIR/clip_vision" \
    "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"

# ═══════════════════════════════════════════════
# 汇总
# ═══════════════════════════════════════════════
echo ""
echo "============================================"
echo " 下载汇总"
echo "============================================"

check() {
    if [ -f "$1" ] && [ -s "$1" ]; then
        size=$(du -h "$1" | cut -f1)
        echo "  ✅ $2 ($size)"
    else
        echo "  ❌ $2 — 缺失!"
    fi
}

echo "FLUX.1-DEV:"
check "$MODELS_DIR/diffusion_models/flux1-dev.safetensors"               "flux1-dev UNet"
check "$MODELS_DIR/text_encoders/clip_l.safetensors"                      "CLIP-L"
check "$MODELS_DIR/text_encoders/t5xxl_fp16.safetensors"                  "T5-XXL FP16"
check "$MODELS_DIR/text_encoders/t5xxl_fp8_e4m3fn.safetensors"            "T5-XXL FP8"
check "$MODELS_DIR/vae/ae.safetensors"                                     "FLUX VAE"

echo "LTX Video 2.3:"
check "$MODELS_DIR/diffusion_models/ltx-2.3-22b-distilled-1.1_transformer_only_mxfp8_block32.safetensors"  "LTX-2.3 UNet"
check "$MODELS_DIR/vae/LTX23_video_vae_bf16.safetensors"                   "LTX Video VAE"
check "$MODELS_DIR/text_encoders/ltx-2.3_text_projection_bf16.safetensors" "投影层 (text_encoders)"
check "$MODELS_DIR/checkpoints/ltx-2.3_text_projection_bf16.safetensors"   "投影层 (checkpoints)"
check "$MODELS_DIR/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors"    "Gemma 3 12B"
check "$MODELS_DIR/loras/ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors" "LTX LoRA"

echo "SDXL 动漫:"
check "$MODELS_DIR/checkpoints/animagine-xl-4.0.safetensors"              "Animagine XL 4.0"

echo "IP-Adapter:"
check "$MODELS_DIR/ipadapter/ip-adapter-plus_sdxl_vit-h.safetensors"      "IP-Adapter SDXL Plus"
check "$MODELS_DIR/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors" "CLIP Vision"

echo ""
echo "── 📌 安装后检查 ──"
echo "  1. IP-Adapter 插件: git clone ComfyUI_IPAdapter_plus → custom_nodes/"
echo "  2. 重启 ComfyUI 以加载新模型和插件"
echo "  3. 投影层必须在 checkpoints/ 和 text_encoders/ 各有一份"
