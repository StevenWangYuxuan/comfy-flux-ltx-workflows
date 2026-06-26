#!/usr/bin/env bash
set -e

MODELS_DIR="/home/steven/comfy/ComfyUI/models"
export https_proxy="http://127.0.0.1:12450/"
export http_proxy="http://127.0.0.1:12450/"

download() {
    local url="$1"; local dest_dir="$2"; local filename="$3"
    mkdir -p "$dest_dir"; local dest="$dest_dir/$filename"
    if [ -f "$dest" ] && [ -s "$dest" ]; then
        echo "  [SKIP] $filename ($(du -h "$dest" | cut -f1))"
        return 0
    fi
    echo "  [DOWNLOAD] $filename → $dest_dir"
    curl -L --progress-bar -o "$dest" "$url"
    [ -s "$dest" ] && echo "  [DONE] $filename ($(du -h "$dest" | cut -f1))" || { echo "  [FAILED]"; return 1; }
}

echo "============================================"
echo " 可选额外模型下载"
echo " (核心模型请先运行 download_models.sh)"
echo "============================================"

echo ""
echo "[1/3] LTX-2.3 空间超分 x2 (可选)"
download \
    "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.0.safetensors" \
    "$MODELS_DIR/latent_upscale_models" \
    "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"

echo ""
echo "[2/3] LTX-2.3 Audio VAE (可选, 如不需要音频可不下载)"
download \
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_audio_vae_bf16.safetensors" \
    "$MODELS_DIR/vae" \
    "LTX23_audio_vae_bf16.safetensors"

echo ""
echo "[3/3] IP-Adapter 插件安装 (可选, 如不需要人脸参考可不装)"
if [ -d "$HOME/comfy/ComfyUI/custom_nodes/ComfyUI_IPAdapter_plus" ]; then
    echo "  [SKIP] ComfyUI_IPAdapter_plus 已安装"
else
    echo "  [INSTALL] 安装 ComfyUI_IPAdapter_plus..."
    cd "$HOME/comfy/ComfyUI/custom_nodes"
    git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git
    echo "  [DONE] IP-Adapter 插件已安装, 重启 ComfyUI 生效"
fi

echo ""
echo "Done! 已下载的额外模型:"
[ -f "$MODELS_DIR/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors" ] && echo "  ✅ 空间超分 x2" || echo "  ❌ 空间超分 x2"
[ -f "$MODELS_DIR/vae/LTX23_audio_vae_bf16.safetensors" ]                 && echo "  ✅ Audio VAE"        || echo "  ❌ Audio VAE"
[ -d "$HOME/comfy/ComfyUI/custom_nodes/ComfyUI_IPAdapter_plus" ]           && echo "  ✅ IP-Adapter 插件"   || echo "  ❌ IP-Adapter 插件"
