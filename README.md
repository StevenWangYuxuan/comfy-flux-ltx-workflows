# ComfyUI FLUX + LTX Video 工作流

基于 ComfyUI 搭建的 **FLUX.1-DEV 文生图 → LTX Video 2.3 图生视频** 级联工作流项目。一句话描述：**用自然语言描述一个画面，AI 先生成静态图，再把它"动起来"变成视频。**

---

## 目录

- [项目结构](#-项目结构)
- [工作流详解](#-工作流详解)
- [辅助脚本](#-辅助脚本)
- [模型依赖](#-模型依赖)
- [快速开始](#-快速开始)
- [硬件要求](#-硬件要求)
- [版本演进](#-版本演进)

---

## 📁 项目结构

```
comfy/
│
├── 🎨 工作流文件 (6个)
│   ├── flux_text2img.json            # FLUX 文生图
│   ├── flux_img2img.json             # FLUX 图生图（以图为参考生成新图）
│   ├── flux_ltx_i2v.json             # FLUX→LTX 组合基线版
│   ├── flux_ltx_i2v_optimized.json   # FLUX→LTX 组合优化版（STG时空引导）
│   ├── flux_ltx_i2v_v2.json          # FLUX→LTX 组合 v2 推荐版 ★
│   └── ltx_txt2vid.json              # LTX 纯文生视频（无FLUX阶段，省显存）
│
├── 📖 文档
│   ├── README.md                     # 本文件 — 项目总览
│   └── flux_ltx_i2v_文档.md         # 组合工作流完整中文文档（节点级详参）
│
├── 🔧 工具脚本 (5个)
│   ├── download_models.py            # Python 版模型下载脚本
│   ├── download_models.sh            # Shell 版模型下载脚本
│   ├── download_extras.sh            # LTX 额外模型下载（VAE / LoRA / 投影层）
│   ├── build_optimized_workflow.py   # 从基线版自动生成优化版/STG版工作流
│   └── fix_workflow_links.py         # 修复/验证工作流节点间的连接关系
│
├── ⚙️ 配置
│   ├── .gitignore                    # Git 忽略规则（排除模型/ComfyUI本体/密钥）
│   └── .mcp.json                     # ComfyUI MCP 服务器配置（供 Claude Code 调用）
│
└── 📦 ComfyUI/                       # ComfyUI 主程序（通过 gitignore 排除，不入库）
    ├── models/                       # 模型文件 (~78GB，不入库)
    │   ├── diffusion_models/         # UNet / DiT 扩散模型
    │   ├── text_encoders/            # 文本编码器（CLIP / T5 / Gemma）
    │   ├── vae/                      # VAE 图像编解码器
    │   └── loras/                    # LoRA 微调权重
    └── custom_nodes/                 # 第三方节点插件
        ├── ComfyUI-LTXVideo/         # LTX Video 官方节点
        ├── ComfyUI-GGUF/             # GGUF 量化模型加载
        └── ComfyUI-KJNodes/          # 工具节点集
```

---

## 🎨 工作流详解

### 1. `flux_text2img.json` — FLUX 文生图

> **输入**：文本提示词 → **输出**：1344×768 PNG 图片

纯 FLUX.1-DEV 文生图工作流，11 个节点。适合快速生成高质量单帧图像。

| 参数 | 值 |
|------|-----|
| 模型 | flux1-dev (fp8_e4m3fn_fast) |
| 文本编码 | CLIP-L + T5-XXL FP16, guidance=4.0 |
| 采样器 | dpmpp_2m + sgm_uniform, 28 步 |
| 分辨率 | 1344 × 768 |
| 主题 | 羽毛球运动员扣杀场景 |

### 2. `flux_img2img.json` — FLUX 图生图

> **输入**：参考图片 + 文本提示词 → **输出**：960×544 PNG 图片

在参考图基础上根据提示词重新生成，denoise=0.65 保留原图结构同时允许较大变化。

| 参数 | 值 |
|------|-----|
| 模型 | flux1-dev (fp8_e4m3fn_fast) |
| 文本编码 | CLIP-L + T5-XXL **FP8**（省显存） |
| 采样器 | dpmpp_2m + sgm_uniform, 25 步, denoise=0.65 |
| 分辨率 | 960 × 544 |

### 3. `flux_ltx_i2v.json` — FLUX→LTX 组合基线版 📦

> **输入**：两段文本（画面描述 + 运动描述） → **输出**：960×544 MP4 视频 (~3秒)

32 节点，最早完整版。FLUX 生成首帧 → VAEDecode → LTX 生成 73 帧视频。

| 阶段 | 参数 |
|------|------|
| FLUX | T5 FP16, dpmpp_2m 28步, guidance=4.0 |
| LTX | NAG(12, 0.30, 2.5), CFG引导, 25步, I2V strength=0.80 |

### 4. `flux_ltx_i2v_optimized.json` — 优化版（实验） ⚠️

33 节点。新增 **STG 时空跳跃引导**，参数偏激进。

| 变化 | 基线 → 优化版 |
|------|---------------|
| T5 | FP16 → **FP8** |
| 引导 | CFG → **STG (stg=1.5, rescale=0.7)** |
| NAG | (12, 0.30, 2.5) → **(13, 0.35, 2.0)** |
| I2V strength | 0.80 → **0.72** |

### 5. `flux_ltx_i2v_v2.json` — v2 推荐版 ★

> **这是当前推荐使用的版本。** 在优化版基础上回调到保守参数，输出更稳定。

| 变化 | 优化版 → v2 | 原因 |
|------|-------------|------|
| NAG scale | 13 → **10** | 减弱引导，减少伪影 |
| NAG tau | 2.0 → **3.5** | 增强时间平滑 |
| STG stg | 1.5 → **1.0** | 降低时空锐化，防过曝 |
| I2V strength | 0.72 → **0.85** | 紧锚定首帧，防漂移 |
| 步数 | 25 → **30** | 更多去噪，画质更好 |
| VAE noise | 0.06/0.03 → **0.03/0.015** | 减少残影鬼影 |

### 6. `ltx_txt2vid.json` — LTX 纯文生视频

> **输入**：单段文本 → **输出**：768×512 MP4 (~2秒)

17 节点，无需 FLUX 阶段，显存友好（~16-20GB）。适合快速生成短视频。

| 参数 | 值 |
|------|-----|
| 文本编码 | Gemma 3 12B FP8 |
| UNet | LTX-2.3 22B (mxfp8 block32) |
| 帧数/帧率 | 49 帧 @ 25fps |
| 采样 | Euler, 20 步, CFG=1.0 |
| NAG | scale=11, alpha=0.25, tau=2.5 |

---

## 🔧 辅助脚本

| 脚本 | 用途 | 用法 |
|------|------|------|
| `download_models.py` | 下载 FLUX.1-DEV 全部模型 (~32GB) | `python3 download_models.py` |
| `download_models.sh` | 同上，Shell 版本 | `bash download_models.sh` |
| `download_extras.sh` | 下载 LTX 模型（VAE/LoRA/投影层） | `bash download_extras.sh` |
| `build_optimized_workflow.py` | 基于基线版自动构建优化版工作流，调整 NAG/STG/步数参数 | `python3 build_optimized_workflow.py` |
| `fix_workflow_links.py` | 验证并修复工作流 JSON 中节点间的 link 连接完整性 | `python3 fix_workflow_links.py` |

---

## 🧠 模型依赖

### FLUX.1-DEV（文生图引擎）

| 模型文件 | 大小 | 加载节点 | 用途 |
|----------|------|----------|------|
| `flux1-dev.safetensors` | 23 GB | UNETLoader (fp8) | DiT 扩散主干 |
| `clip_l.safetensors` | 235 MB | DualCLIPLoader | 短文本编码 |
| `t5xxl_fp8_e4m3fn.safetensors` | 4.6 GB | DualCLIPLoader | 长文本编码 (v2 推荐) |
| `t5xxl_fp16.safetensors` | 9.2 GB | DualCLIPLoader | 长文本编码 (基线版备用) |
| `ae.safetensors` | 320 MB | VAELoader | latent ↔ 像素 编解码 |

### LTX Video 2.3（视频生成引擎）

| 模型文件 | 大小 | 加载节点 | 用途 |
|----------|------|----------|------|
| `ltx-2.3-22b-...-mxfp8_block32.safetensors` | 23 GB | UNETLoader | 视频 DiT 主干（分块加载） |
| `gemma_3_12B_it_fp8_scaled.safetensors` | 13 GB | LTXAVTextEncoderLoader | 运动描述编码 |
| `ltx-2.3_text_projection_bf16.safetensors` | 2.2 GB | LTXAVTextEncoderLoader | Gemma→LTX 维度投影 |
| `LTX23_video_vae_bf16.safetensors` | 1.4 GB | VAELoader | 视频 latent ↔ 像素 |
| `ltx-2.3-22b-distilled-lora-..._bf16.safetensors` | 2.5 GB | LTXICLoRALoader | 动态增强 LoRA |

> **总计约 78 GB**（全部已下载并就绪）

---

## 🚀 快速开始

### 1. 启动 ComfyUI

```bash
cd ~/comfy/ComfyUI

# 标准启动
venv/bin/python main.py --listen 127.0.0.1

# 24GB 显存优化启动
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 \
  venv/bin/python main.py --listen 127.0.0.1
```

### 2. 加载并运行

浏览器打开 `http://127.0.0.1:8188`，将 `.json` 工作流文件拖入窗口，点击 Queue Prompt。

### 3. 选型指南

| 你想做什么 | 用这个文件 | 显存需求 |
|-----------|-----------|----------|
| 文生图 | `flux_text2img.json` | 16-20 GB |
| 图生图 | `flux_img2img.json` | 16-20 GB |
| **图片变视频（推荐）** | `flux_ltx_i2v_v2.json` ★ | 22-24 GB |
| 纯文本变视频（省显存） | `ltx_txt2vid.json` | 16-20 GB |

---

## 💻 硬件要求

| GPU | 显存 | FLUX 工作流 | I2V 组合工作流 | 建议 |
|-----|------|------------|---------------|------|
| RTX 3090 Ti | 24 GB | ✅ 流畅 | ⚠️ 极限（需优化参数） | 降分辨率至 768×432 |
| RTX 4090 | 24 GB | ✅ 流畅 | ⚠️ 极限 | 同上 |
| RTX 6000 Ada / A6000 | 48 GB | ✅ 完美 | ✅ 推荐 | 无限制 |
| A100 / H100 | 80 GB | ✅ 完美 | ✅ 完美 | 全工作流随便跑 |

24GB 显卡运行 I2V 工作流的详细优化策略见 `.claude/projects/-home-steven-comfy/memory/vram-troubleshooting.md`。

---

## 📜 版本演进

```
2026-06-22  安装 ComfyUI + FLUX Schnell 文生图
2026-06-23  迁移到 FLUX.1-DEV，下载全部模型 (~32GB)
2026-06-24  创建 FLUX+LTX 组合基线版 (32节点)
           修复 T5→Gemma 文本编码器，NAG 参数调优
           新增纯 LTX 文生视频工作流
2026-06-25  引入 STG 时空跳跃引导 (33节点)
           发布优化版 + v2 保守参数版
           新增 FLUX 图生图工作流，T5 切换至 FP8
2026-06-26  驱动升级 595→580，编写 VRAM 故障排除文档
           清理 40GB 冗余模型，整理文档结构
           项目上传 GitHub
```

---

## 📚 延伸阅读

- [`flux_ltx_i2v_文档.md`](flux_ltx_i2v_文档.md) — 32 节点逐一拆解，包含每个节点的参数含义和调优建议
- 项目 memory 文件位于 `.claude/projects/-home-steven-comfy/memory/`，由 Claude Code 维护
