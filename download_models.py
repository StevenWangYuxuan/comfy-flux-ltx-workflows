#!/usr/bin/env python3
"""Download FLUX.1-DEV and LTX-2.3 models for ComfyUI"""
import os
import sys

# MUST be set before any imports
os.environ.pop("ALL_PROXY", None)
os.environ.pop("all_proxy", None)
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:12450/"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:12450/"

from huggingface_hub import hf_hub_download, snapshot_download
from pathlib import Path

COMFYUI_DIR = Path("/home/steven/comfy/ComfyUI")
MODELS_DIR = COMFYUI_DIR / "models"

DOWNLOADS = [
    # ─── FLUX.1-DEV Diffusion Model (FP16) ──────────────────
    {
        "name": "FLUX.1-DEV Diffusion Model",
        "repo_id": "Comfy-Org/flux1-dev",
        "filename": "flux1-dev.safetensors",
        "local_dir": MODELS_DIR / "diffusion_models",
    },
    # ─── FLUX.1-DEV CLIP-L ────────────────────────────────
    {
        "name": "CLIP-L Text Encoder",
        "repo_id": "comfyanonymous/flux_text_encoders",
        "filename": "clip_l.safetensors",
        "local_dir": MODELS_DIR / "text_encoders",
    },
    # ─── FLUX.1-DEV T5-XXL (FP16) ─────────────────────────
    {
        "name": "T5-XXL FP16 Text Encoder",
        "repo_id": "comfyanonymous/flux_text_encoders",
        "filename": "t5xxl_fp16.safetensors",
        "local_dir": MODELS_DIR / "text_encoders",
    },
    # ─── FLUX VAE ────────────────────────────────────────
    {
        "name": "FLUX VAE (ae.safetensors)",
        "repo_id": "Comfy-Org/Lumina_Image_2.0_Repackaged",
        "filename": "split_files/vae/ae.safetensors",
        "local_dir": MODELS_DIR / "vae",
    },
    # ─── LTX-2.3 Diffusion Model (distilled, FP8) ─────────
    {
        "name": "LTX-2.3 Distilled Transformer (mxfp8)",
        "repo_id": "Kijai/LTX2.3_comfy",
        "filename": "diffusion_models/ltx-2.3-22b-distilled-1.1_transformer_only_mxfp8_block32.safetensors",
        "local_dir": MODELS_DIR / "diffusion_models",
    },
    # ─── LTX-2.3 Video VAE ─────────────────────────────────
    {
        "name": "LTX-2.3 Video VAE",
        "repo_id": "Kijai/LTX2.3_comfy",
        "filename": "vae/LTX23_video_vae_bf16.safetensors",
        "local_dir": MODELS_DIR / "vae",
    },
    # ─── LTX-2.3 Text Projection ───────────────────────────
    {
        "name": "LTX-2.3 Text Projection",
        "repo_id": "Kijai/LTX2.3_comfy",
        "filename": "text_encoders/ltx-2.3_text_projection_bf16.safetensors",
        "local_dir": MODELS_DIR / "text_encoders",
    },
    # ─── LTX-2.3 Distilled LoRA (optional) ────────────────
    {
        "name": "LTX-2.3 Distilled LoRA",
        "repo_id": "Kijai/LTX2.3_comfy",
        "filename": "loras/ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors",
        "local_dir": MODELS_DIR / "loras",
    },
]

def download_file(item):
    """Download a single file with progress"""
    name = item["name"]
    repo_id = item["repo_id"]
    filename = item["filename"]
    local_dir = item["local_dir"]

    # Ensure dir exists
    local_dir.mkdir(parents=True, exist_ok=True)

    target_name = os.path.basename(filename)
    target_path = local_dir / target_name

    print(f"  ⏳ Downloading: {target_name}...")
    print(f"     → {local_dir}")
    if target_path.exists():
        size_gb = target_path.stat().st_size / (1024**3)
        print(f"     Existing file: {size_gb:.1f} GB (will resume if incomplete)")

    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
        size_gb = os.path.getsize(path) / (1024**3)
        print(f"  ✅ Done: {target_name} ({size_gb:.1f} GB)")
        return path
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return None

def main():
    print("=" * 60)
    print("Downloading FLUX.1-DEV & LTX-2.3 Models for ComfyUI")
    print(f"Destination: {MODELS_DIR}")
    print("=" * 60)

    results = {}
    for item in DOWNLOADS:
        print(f"\n[{item['name']}]")
        results[item['name']] = download_file(item)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, path in results.items():
        status = "✅" if path else "❌"
        print(f"  {status} {name}")

    # Check for Gemma 3 text encoder (needed by LTX-2.3)
    gemma_path = MODELS_DIR / "text_encoders" / "gemma-3-12b-it"
    if not gemma_path.exists():
        print(f"\n  ⚠️ Gemma 3 text encoder not found at: {gemma_path}")
        print(f"     LTX-2.3 needs Gemma 3 12B text encoder for full functionality.")
        print(f"     Download from: google/gemma-3-12b-it")
        print(f"     Or the ComfyUI-LTXVideo node may auto-download it.")

if __name__ == "__main__":
    main()
