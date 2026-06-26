# 🎬 ComfyUI FLUX + LTX Video — AI 图片转视频工作流

> 给 AI 写一段话描述画面，先生成高清图片，再让图片"动起来"变成视频。

8 个 ComfyUI 工作流，覆盖写实（FLUX）、动漫（Animagine XL）、人脸参考（IP-Adapter）。

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-0.25.0-blue)](https://github.com/comfyanonymous/ComfyUI)
[![Models](https://img.shields.io/badge/Models-15-orange)](#-模型清单)
[![Workflows](https://img.shields.io/badge/Workflows-8-purple)](#-工作流)

---

## 目录

- [能做什么](#能做什么)
- [效果预览](#效果预览)
- [快速开始](#快速开始)
- [工作流](#工作流)
- [模型清单](#模型清单)
- [硬件要求](#硬件要求)
- [常见问题](#常见问题)

---

## 能做什么

| 你想要的效果 | 用这个 | 难度 |
|-------------|------|------|
| 文字 → 写实照片 | `flux_text2img` | ⭐ |
| 文字 → 动漫头像 | `anime_portrait` | ⭐ |
| 参考图 + 文字 → 新图 | `flux_img2img` | ⭐ |
| 一张脸 → 各种造型 | `ipadapter_portrait` | ⭐⭐ |
| 文字 → 短视频 | `ltx_txt2vid` | ⭐⭐ |
| 两段文字 → 高质量视频 | `flux_ltx_i2v_v2` ★ | ⭐⭐⭐ |

---

## 效果预览

### FLUX 文生图

![FLUX 文生图](assets/flux_text2img_demo.png)

### FLUX 图生图

![FLUX 图生图](assets/flux_img2img_demo.png)

### FLUX→LTX 图生视频

![首帧](assets/flux_ltx_i2v_frame.png)  🎥 [视频](assets/hypercar_demo.mp4)

### SDXL 动漫头像

![动漫](assets/anime_portrait_demo.png)

### IP-Adapter 人脸参考

![IP-Adapter](assets/ipadapter_portrait_demo.png)

---

## 快速开始

### 1. 下载模型

```bash
cd ~/comfy
bash download_models.sh    # 下载全部 15 个模型 (~85GB)
bash download_extras.sh    # 可选模型 + IP-Adapter 插件
```

### 2. 启动 ComfyUI

```bash
cd ~/comfy/ComfyUI

# 标准启动
venv/bin/python main.py --listen 127.0.0.1

# 24GB 显卡优化启动（RTX 3090 Ti）
# --lowvram: 按需加载模型，用完即卸载到内存
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 \
  venv/bin/python main.py --listen 127.0.0.1 --lowvram
```

### 3. 打开浏览器 → 拖入工作流 → 点击生成

`http://127.0.0.1:8188`

---

## 工作流

| 文件 | 类型 | 节点 | 显存 | 详细文档 |
|------|------|------|------|----------|
| `anime_portrait.json` | 动漫头像 | 8 | 8-12 GB | [📖](docs/anime_portrait.md) |
| `ipadapter_portrait.json` | 人脸参考 | 10 | 10-14 GB | [📖](docs/ipadapter_portrait.md) |
| `flux_text2img.json` | 写实文生图 | 11 | 16-20 GB | [📖](docs/flux_text2img.md) |
| `flux_img2img.json` | 写实图生图 | 14 | 16-20 GB | [📖](docs/flux_img2img.md) |
| `flux_ltx_i2v_v2.json` ★ | 图生视频 | 33 | 22-24 GB | [📖](docs/flux_ltx_i2v_v2.md) |
| `ltx_txt2vid.json` | 文生视频 | 17 | 16-20 GB | [📖](docs/ltx_txt2vid.md) |
| `flux_ltx_i2v.json` | 图生视频(基线) | 32 | 24+ GB | 📦 存档 |
| `flux_ltx_i2v_optimized.json` | 图生视频(实验) | 33 | 24+ GB | ⚠️ 实验 |

> ★ `flux_ltx_i2v_v2.json` 是推荐版本。完整参数文档见 [`flux_ltx_i2v_文档.md`](flux_ltx_i2v_文档.md)。

---

## 项目结构

```
comfy/
├── *.json                 # 8 个工作流文件
├── docs/                  # 工作流详细文档
├── assets/                # 效果预览图和视频
├── download_models.py     # Python 模型下载
├── download_models.sh     # Shell 模型下载
├── download_extras.sh     # 可选模型 + 插件
├── build_optimized_workflow.py
├── fix_workflow_links.py
├── flux_ltx_i2v_文档.md  # FLUX→LTX 完整参数手册
├── README.md
└── LICENSE
```

---

## 模型清单

### FLUX.1-DEV（写实）

| 文件 | 大小 | 位置 |
|------|------|------|
| `flux1-dev.safetensors` | 23 GB | `diffusion_models/` |
| `clip_l.safetensors` | 235 MB | `text_encoders/` |
| `t5xxl_fp16.safetensors` | 9.2 GB | `text_encoders/` |
| `t5xxl_fp8_e4m3fn.safetensors` | 4.6 GB | `text_encoders/` |
| `ae.safetensors` | 320 MB | `vae/` |

### LTX Video 2.3（视频）

| 文件 | 大小 | 位置 |
|------|------|------|
| `ltx-2.3-22b-...mxfp8_block32.safetensors` | 23 GB | `diffusion_models/` |
| `LTX23_video_vae_bf16.safetensors` | 1.4 GB | `vae/` |
| `gemma_3_12B_it_fp8_scaled.safetensors` | 13 GB | `text_encoders/` |
| `ltx-2.3_text_projection_bf16.safetensors` | 2.2 GB | `text_encoders/` + `checkpoints/` |
| `ltx-2.3-22b-distilled-lora-...bf16.safetensors` | 2.5 GB | `loras/` |

### SDXL 动漫

| 文件 | 大小 | 位置 |
|------|------|------|
| `animagine-xl-4.0.safetensors` | 6.5 GB | `checkpoints/` |

### IP-Adapter

| 文件 | 大小 | 位置 |
|------|------|------|
| `ip-adapter-plus_sdxl_vit-h.safetensors` | 809 MB | `ipadapter/` |
| `CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors` | 2.4 GB | `clip_vision/` |

**总计 ~84.5 GB**

---

## 硬件要求

| 显卡 | 显存 | 动漫/IP-Adapter | FLUX 工作流 | 图生视频 |
|------|------|:---:|:---:|:---:|
| RTX 3060/4070 | 12 GB | ✅ | ❌ | ❌ |
| RTX 3090/4090 | 24 GB | ✅ | ✅ | ⚠️ 需优化 |
| A6000 | 48 GB | ✅ | ✅ | ✅ |
| A100/H100 | 80 GB | ✅ | ✅ | ✅ |

---

## 常见问题

### OOM（显存不足）

1. 降分辨率：960×544 → 768×432
2. 启动加 `--novram` 参数
3. 关闭浏览器等吃显存的程序

### 视频有闪烁/变形

- I2V strength 调到 0.85+
- STG 降到 1.0
- VAE Decoder Noise：timestep=0.03, scale=0.015

### IP-Adapter 有波浪纹

- `combine_embeds` 设为 `concat`
- `weight` 降到 0.6-0.7

### 模型找不到

参见 [download_models.sh](download_models.sh) 中的目录映射表。

---

## 更新日志

| 日期 | 变更 |
|------|------|
| 2026-06-26 | 新增动漫头像 + IP-Adapter 人脸参考，完善文档结构和下载脚本 |

## 许可

工作流和脚本使用 [MIT License](LICENSE)。各 AI 模型遵循各自的许可证。
