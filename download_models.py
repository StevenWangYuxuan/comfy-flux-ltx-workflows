#!/usr/bin/env python3
"""下载 ComfyUI 全部模型 — FLUX + LTX + SDXL动漫 + IP-Adapter"""
import os
import sys

os.environ.pop("ALL_PROXY", None)
os.environ.pop("all_proxy", None)
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:12450/"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:12450/"

from huggingface_hub import hf_hub_download
from pathlib import Path

COMFYUI_DIR = Path("/home/steven/comfy/ComfyUI")
MODELS_DIR = COMFYUI_DIR / "models"

DOWNLOADS = [
    # ═══════════════════════════════════════════════════════════
    # FLUX.1-DEV — 写实文生图/图生图
    # ═══════════════════════════════════════════════════════════
    {
        "name": "FLUX.1-DEV UNet (fp8)",
        "repo_id": "Comfy-Org/flux1-dev",
        "filename": "flux1-dev.safetensors",
        "local_dir": MODELS_DIR / "diffusion_models",
        "section": "FLUX.1-DEV 写实模型",
    },
    {
        "name": "CLIP-L 文本编码器",
        "repo_id": "comfyanonymous/flux_text_encoders",
        "filename": "clip_l.safetensors",
        "local_dir": MODELS_DIR / "text_encoders",
        "section": "FLUX.1-DEV 写实模型",
    },
    {
        "name": "T5-XXL FP16 文本编码器",
        "repo_id": "comfyanonymous/flux_text_encoders",
        "filename": "t5xxl_fp16.safetensors",
        "local_dir": MODELS_DIR / "text_encoders",
        "section": "FLUX.1-DEV 写实模型",
    },
    {
        "name": "T5-XXL FP8 文本编码器 (v2推荐, 省显存)",
        "repo_id": "comfyanonymous/flux_text_encoders",
        "filename": "t5xxl_fp8_e4m3fn.safetensors",
        "local_dir": MODELS_DIR / "text_encoders",
        "section": "FLUX.1-DEV 写实模型",
    },
    {
        "name": "FLUX VAE",
        "repo_id": "Comfy-Org/Lumina_Image_2.0_Repackaged",
        "filename": "split_files/vae/ae.safetensors",
        "local_dir": MODELS_DIR / "vae",
        "section": "FLUX.1-DEV 写实模型",
    },

    # ═══════════════════════════════════════════════════════════
    # LTX Video 2.3 — 图生视频/文生视频
    # ═══════════════════════════════════════════════════════════
    {
        "name": "LTX-2.3 UNet (mxfp8 block32)",
        "repo_id": "Kijai/LTX2.3_comfy",
        "filename": "diffusion_models/ltx-2.3-22b-distilled-1.1_transformer_only_mxfp8_block32.safetensors",
        "local_dir": MODELS_DIR / "diffusion_models",
        "section": "LTX Video 2.3 视频模型",
    },
    {
        "name": "LTX Video VAE",
        "repo_id": "Kijai/LTX2.3_comfy",
        "filename": "vae/LTX23_video_vae_bf16.safetensors",
        "local_dir": MODELS_DIR / "vae",
        "section": "LTX Video 2.3 视频模型",
    },
    {
        "name": "LTX 文本投影层 (text_encoders/)",
        "repo_id": "Kijai/LTX2.3_comfy",
        "filename": "text_encoders/ltx-2.3_text_projection_bf16.safetensors",
        "local_dir": MODELS_DIR / "text_encoders",
        "section": "LTX Video 2.3 视频模型",
    },
    {
        "name": "LTX 文本投影层 (checkpoints/) ← LTXAVTextEncoderLoader 读这里!",
        "repo_id": "Kijai/LTX2.3_comfy",
        "filename": "text_encoders/ltx-2.3_text_projection_bf16.safetensors",
        "local_dir": MODELS_DIR / "checkpoints",
        "section": "LTX Video 2.3 视频模型",
    },
    {
        "name": "LTX Gemma 3 12B FP8 文本编码器 (13GB)",
        "repo_id": "Kijai/LTX2.3_comfy",
        "filename": "text_encoders/gemma_3_12B_it_fp8_scaled.safetensors",
        "local_dir": MODELS_DIR / "text_encoders",
        "section": "LTX Video 2.3 视频模型",
    },
    {
        "name": "LTX LoRA 动态增强",
        "repo_id": "Kijai/LTX2.3_comfy",
        "filename": "loras/ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors",
        "local_dir": MODELS_DIR / "loras",
        "section": "LTX Video 2.3 视频模型",
    },

    # ═══════════════════════════════════════════════════════════
    # SDXL 动漫 — Animagine XL 4.0
    # ═══════════════════════════════════════════════════════════
    {
        "name": "Animagine XL 4.0 (SDXL 动漫, 6.5GB)",
        "repo_id": "cagliostrolab/animagine-xl-4.0",
        "filename": "animagine-xl-4.0.safetensors",
        "local_dir": MODELS_DIR / "checkpoints",
        "section": "SDXL 动漫模型",
    },

    # ═══════════════════════════════════════════════════════════
    # IP-Adapter — 人脸参考生成
    # ═══════════════════════════════════════════════════════════
    {
        "name": "IP-Adapter SDXL Plus (809MB)",
        "repo_id": "h94/IP-Adapter",
        "filename": "sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors",
        "local_dir": MODELS_DIR / "ipadapter",
        "section": "IP-Adapter 人脸参考",
    },
    {
        "name": "CLIP Vision ViT-H (2.4GB)",
        "repo_id": "h94/IP-Adapter",
        "filename": "models/image_encoder/model.safetensors",
        "local_dir": MODELS_DIR / "clip_vision",
        "section": "IP-Adapter 人脸参考",
    },
]


def download_file(item):
    """Download a single file"""
    name = item["name"]
    local_dir = item["local_dir"]
    target_name = os.path.basename(item["filename"])
    # CLIP Vision needs special rename
    if target_name == "model.safetensors":
        target_name = "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
    target_path = local_dir / target_name

    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"  ⏳ {target_name}")
    if target_path.exists():
        size_gb = target_path.stat().st_size / (1024**3)
        print(f"    已有文件: {size_gb:.1f} GB (跳过)")
        return str(target_path)

    try:
        path = hf_hub_download(
            repo_id=item["repo_id"],
            filename=item["filename"],
            local_dir=str(local_dir),
        )
        # Rename for CLIP Vision
        if target_name == "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors":
            src = Path(path)
            dst = src.parent / target_name
            if src != dst:
                src.rename(dst)
                path = str(dst)
        size_gb = os.path.getsize(path) / (1024**3)
        print(f"  ✅ 完成: {target_name} ({size_gb:.1f} GB)")
        return path
    except Exception as e:
        print(f"  ❌ 失败: {e}")
        return None


def main():
    print("=" * 65)
    print("  ComfyUI 全部模型下载")
    print(f"  目标: {MODELS_DIR}")
    print("=" * 65)

    # Group by section
    sections = {}
    for item in DOWNLOADS:
        sec = item.get("section", "其他")
        sections.setdefault(sec, []).append(item)

    results = {}
    for section, items in sections.items():
        total_items = len(items)
        print(f"\n── {section} ({total_items}个文件) ──")
        for item in items:
            results[item["name"]] = download_file(item)

    # Summary
    print("\n" + "=" * 65)
    print("  下载汇总")
    print("=" * 65)
    ok = fail = 0
    for name, path in results.items():
        status = "✅" if path else "❌"
        if path: ok += 1
        else: fail += 1
        print(f"  {status} {name}")
    print(f"\n  成功: {ok} / 失败: {fail} / 总计: {len(results)}")

    # Post-install notes
    print("\n── 📌 安装后检查 ──")
    print("  1. IP-Adapter 插件: git clone ComfyUI_IPAdapter_plus → custom_nodes/")
    print("  2. 重启 ComfyUI 以加载新模型和插件")
    print("  3. 检查投影层在 checkpoints/ 和 text_encoders/ 各有一份")

if __name__ == "__main__":
    main()
