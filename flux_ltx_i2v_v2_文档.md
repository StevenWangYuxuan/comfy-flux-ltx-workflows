# FLUX + LTX I2V 组合工作流 — 完整节点与参数文档

> **当前推荐版本**: `flux_ltx_i2v_v2.json` ★ (33 节点, STG 时空跳跃引导, 保守参数)
>
> 本文档基于 v2 版本撰写。基线版 (`flux_ltx_i2v.json`) 和优化版 (`flux_ltx_i2v_optimized.json`) 的差异在文末附表中说明。

| 版本 | 文件 | 引导 | 步数 | 状态 |
|------|------|------|------|------|
| 基线版 | `flux_ltx_i2v.json` | CFG | 25 | 📦 存档 |
| 优化版 | `flux_ltx_i2v_optimized.json` | STG 1.5 | 25 | ⚠️ 实验 |
| **v2** ★ | **`flux_ltx_i2v_v2.json`** | **STG 1.0** | **30** | ✅ **推荐** |

---

## 目录

1. [架构总览](#1-架构总览)
2. [数据流图](#2-数据流图)
3. [Group 1: FLUX 模型加载](#3-group-1-flux-模型加载-节点-14)
4. [Group 2: FLUX Prompt 与 Conditioning](#4-group-2-flux-prompt-与-conditioning-节点-58)
5. [Group 3: FLUX 采样](#5-group-3-flux-采样-节点-9)
6. [Group 4: FLUX→LTX 桥接](#6-group-4-fluxltx-桥接-节点-1011)
7. [Group 5: LTX 模型加载](#7-group-5-ltx-模型加载-节点-1219)
8. [Group 6: LTX Prompt 与 I2V](#8-group-6-ltx-prompt-与-i2v-节点-1315)
9. [Group 7: LTX 采样](#9-group-7-ltx-采样-节点-2127)
10. [Group 8: LTX 输出](#10-group-8-ltx-输出-节点-2832)
11. [参数速查表](#11-参数速查表)
12. [版本对比](#12-版本对比)
13. [调试指南](#13-调试指南)

---

## 1. 架构总览

这是一个**两阶段级联工作流**：

```
阶段 1: FLUX.1-DEV 文生图 (节点 1-11)
    └→ 根据文本 Prompt 生成一张高质量的初始帧图片 (960×544)

阶段 2: LTX Video 2.3 图生视频 (节点 12-33)
    └→ 以 FLUX 生成的图片为起点，根据运动 Prompt 生成 73 帧 (≈3秒) 连贯视频
```

**为什么分两阶段**: FLUX.1-DEV 是目前最强的开源文生图模型之一，LTX Video 2.3 是 24GB 显存可运行的最强图生视频模型。两者结合：FLUX 负责"画面质量"，LTX 负责"运动质量"。

**v2 的核心改进**: 相比基线版，v2 加入了 STG (Spatio-Temporal Guidance) 时空跳跃引导，在 transformer 第 19 层施加额外注意力约束；NAG 参数回调到更保守的值；I2V 锚定增强到 0.85；采样步数从 25 提升到 30。

---

## 2. 数据流图

```
┌──────────────────────────────────────────────────────────────────────────┐
│                   FLUX→LTX 图生视频 v2 工作流 (33节点)                      │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ╔══════════════════════════ 阶段1: FLUX 文生图 ═════════════════════════╗ │
│  ║                                                                      ║ │
│  ║  ① UNET Loader ──→ ② Flux 调度 ──→ ⑨ KSampler                       ║ │
│  ║  ③ DualCLIP ──→ ⑤ 正向Prompt ──→ ⑨   ←── ⑥⑦⑧ (负向/归零/画布)     ║ │
│  ║  ④ VAE Loader ──────────────────→ ⑩ VAEDecode                       ║ │
│  ║                                      │                               ║ │
│  ║                                      ▼                               ║ │
│  ║                                  ⑩ IMAGE (960×544 像素)              ║ │
│  ║                                  ⑪ SaveImage (存档首帧)              ║ │
│  ╚══════════════════════════════════════════════════════════════════════╝ │
│                                    │                                      │
│  ╔══════════════════════ 阶段2: FLUX→LTX 桥接 ══════════════════════════╗ │
│  ║                                                                      ║ │
│  ║  ⑩ IMAGE → ⑳ LTXVPreprocess(压缩=38) → ㉒ I2V Condition              ║ │
│  ║                    ⑲ VAE + ㉑ 空视频画布 ──→ ㉒ (strength=0.85)      ║ │
│  ║                                                                      ║ │
│  ║  FLUX VAE 和 LTX VAE 互不兼容，必须走像素中间表示                        ║ │
│  ╚══════════════════════════════════════════════════════════════════════╝ │
│                                    │                                      │
│  ╔══════════════════════ 阶段3: LTX 视频生成 ═══════════════════════════╗ │
│  ║                                                                      ║ │
│  ║  ⑫ Gemma 编码 → ⑬正向 / ⑭负向 → ⑮ LTXV Conditioning(24fps)              ║ │
│  ║                                                                      ║ │
│  ║  模型链: ⑯ UNet → ㉝ STG(block19) → ⑰ NAG(10,0.18,3.5) → ⑱ LoRA    ║ │
│  ║                                                                      ║ │
│  ║  ㉓ STGGuider(stg=1.0,rescale=0.5) ← 正/负Conditioning               ║ │
│  ║                                                                      ║ │
│  ║  ㉗ SamplerCustomAdvanced ← ㉔调度(30步) ㉕采样(euler) ㉖噪声(888)      ║ │
│  ║                                                                      ║ │
│  ║  ㉘ VAE Decoder Noise → ㉙ TiledVAEDecode → ㉚ Video → ㉜ Save        ║ │
│  ╚══════════════════════════════════════════════════════════════════════╝ │
│                                                                          │
│  输出: 960×544 MP4, 73帧, 24fps, ≈3秒                                     │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Group 1: FLUX 模型加载 (节点 1-4)

| 节点 | 类型 | 模型文件 | 参数 |
|------|------|----------|------|
| 1 | UNETLoader | `flux1-dev.safetensors` | weight_dtype=`fp8_e4m3fn_fast` |
| 2 | ModelSamplingFlux | — | max_shift=1.35, base_shift=0.5, 960×544 |
| 3 | DualCLIPLoader | `clip_l.safetensors` + `t5xxl_fp8_e4m3fn.safetensors` | type=`flux` |
| 4 | VAELoader | `ae.safetensors` | — |

**关键设计**:
- FLUX UNet 使用 `fp8_e4m3fn_fast` 量化，23GB → 12GB 显存，损失极小
- v2 将 T5 编码器从 FP16 改为 **FP8** (9.2GB → 4.6GB)
- ModelSamplingFlux 的 shift 值 1.35 是针对 960×544 分辨率优化的

---

## 4. Group 2: FLUX Prompt 与 Conditioning (节点 5-8)

| 节点 | 类型 | 功能 | 关键参数 |
|------|------|------|----------|
| 5 | CLIPTextEncodeFlux | 正向 Prompt | guidance=**4.0** |
| 6 | CLIPTextEncode | 负向 Prompt | — |
| 7 | ConditioningZeroOut | 负向归零 | FLUX 标准要求 |
| 8 | EmptySD3LatentImage | 画布 | 960×544, batch=1 |

**v2 的 FLUX Prompt** (超跑主题):

```
clip_l (简短关键词):
"sleek matte black hypercar parked on wet asphalt road, golden sunset backlight,
cinematic automotive photography, low angle three-quarter front hero shot,
dramatic rim lighting on car body curves, wet pavement reflections"

t5xxl (详细场景):
"A breathtaking cinematic wide shot of a matte black hypercar parked diagonally
across a rain-dampened asphalt road during golden hour..."
```

**Prompt 策略**:
- CLIP-L 写**简短标签**（构图、光线、主体）
- T5-XXL 写**完整描述**（场景、氛围、细节）
- guidance=4.0 是 FLUX.1-DEV 的最佳平衡点：3.5 画面偏自由，4.5 过度锐化

**负向 Prompt**:
```
blurry, low quality, distorted, ugly, bad anatomy, motion blur, moving car,
driving, wheels spinning, smoke from tires, people, driver, person
```
注意负向里排除了"运动/驾驶/车轮转动"——因为我们要的是**静止**超跑。

---

## 5. Group 3: FLUX 采样 (节点 9)

| 参数 | 值 | 说明 |
|------|-----|------|
| seed | 777 | 固定种子确保可复现 |
| steps | **28** | FLUX 性价比最佳 |
| cfg | 1.0 | FLUX 内部用 guidance，CFG 固定 1.0 |
| sampler | `dpmpp_2m` | 收敛稳定 |
| scheduler | `sgm_uniform` | FLUX 推荐搭配 |
| denoise | 1.0 | 文生图全去噪 |

---

## 6. Group 4: FLUX→LTX 桥接 (节点 10-11)

| 节点 | 类型 | 功能 |
|------|------|------|
| 10 | VAEDecode | FLUX latent → 960×544 像素图 |
| 11 | SaveImage | 保存首帧 `flux_frame_*.png` |

**桥接原理**: FLUX 和 LTX 使用**不同的 VAE**，不能直接传递 latent。

```
FLUX Latent (16ch, ae.safetensors) → VAEDecode → 像素
    → LTXVPreprocess(压缩=38) → LTX VAE (128ch, LTX23_video_vae_bf16) → LTX Latent
```

---

## 7. Group 5: LTX 模型加载 (节点 12-19)

| 节点 | 类型 | 模型/参数 | 说明 |
|------|------|-----------|------|
| 12 | LTXAVTextEncoderLoader | Gemma 3 12B fp8 + 投影层 | LTX 2.3 原生编码器 |
| 16 | UNETLoader | LTX-2.3 22B mxfp8 block32 | 23GB, 分块加载 |
| 33 | LTXVApplySTG | block_indices=**"19"** | 时空跳跃引导 |
| 17 | LTX2_NAG | scale=**10**, alpha=**0.18**, tau=**3.5** | 噪声增强引导 |
| 18 | LTXICLoRALoaderModelOnly | LoRA, strength=1.0 | 动态增强 |
| 19 | VAELoader | LTX23_video_vae_bf16 (1.4GB) | LTX 视频 VAE |

### STG 时空跳跃引导 (v2 新增)

`LTXVApplySTG` 对 transformer 第 19 层施加额外的时空注意力约束：

```
block_indices="19"  →  仅第 19 层
```

配合 `STGGuider` (节点 23)：
- `stg=1.0`: 引导强度。1.0=标准，1.5=激进（可能过度锐化）
- `rescale=0.5`: 输出缩放。防止 STG 导致亮度过曝

### NAG 参数 (v2 保守值)

| 参数 | 值 | 范围 | 作用 |
|------|-----|------|------|
| nag_scale | **10** | 8-14 | 运动引导强度。太低不跟 Prompt，太高产生鬼畜 |
| nag_alpha | **0.18** | 0.10-0.30 | 锐度/对比度。高值画面锐利但可能噪点增多 |
| nag_tau | **3.5** | 2.0-4.0 | 时间平滑。高值运动更连贯，但可能"拖影" |

### 投影层存放位置

LTXAVTextEncoderLoader 读取两个文件：
- Gemma 模型: `models/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors` (13GB)
- **投影层: `models/checkpoints/ltx-2.3_text_projection_bf16.safetensors` (2.2GB)** ← 注意!

---

## 8. Group 6: LTX Prompt 与 I2V (节点 13-15)

| 节点 | 类型 | 功能 | 参数 |
|------|------|------|------|
| 13 | CLIPTextEncode | 正向 Prompt (运动描述) | — |
| 14 | CLIPTextEncode | 负向 Prompt (排除伪影) | — |
| 15 | LTXVConditioning | 帧率条件 | **24fps** |

**v2 的 LTX Prompt** (超跑运动):

```
"The hypercar remains perfectly still for a breath... Then the LED headlights
flicker on sharply... The engine ignites with a deep visceral growl... the rear
tires suddenly bite into the wet asphalt and the hypercar launches forward with
brutal acceleration — a violent spray of water mist and gravel erupts from the
rear wheel wells. The vehicle rockets away... taillights streak into crimson
light trails... Continuous coherent motion, no morphing or warping, physically
realistic vehicle dynamics"
```

**Prompt 写作原则**:
- FLUX Prompt: 描述**初始帧画面** (谁、在哪、什么光线、什么构图)
- LTX Prompt: 描述**接下来发生的运动** (如何启动、如何移动、什么速度、什么物理效果)
- 两者**互补不重复**

**负向 Prompt** (视频专用):
```
blurry, jittery, low quality, distorted, bad motion, static image, still frame,
no movement, ugly, stop motion, choppy, jerky, teleporting, morphing, warping,
flickering, inconsistent, ghosting, motion blur, frame skip, car disappearing,
sudden cut, glitch, deformation, rubber tires, floating car, cartoon physics,
unreal engine, CGI look, plastic material, toy car
```

---

## 9. Group 7: LTX 采样 (节点 20-27)

| 节点 | 类型 | 关键参数 |
|------|------|----------|
| 20 | LTXVPreprocess | img_compression=**38** |
| 21 | EmptyLTXVLatentVideo | 960×544, 73帧, batch=1 |
| 22 | LTXVImgToVideoConditionOnly | strength=**0.85**, bypass=false |
| 23 | STGGuider | stg=**1.0**, rescale=**0.5**, cfg=1.0 |
| 24 | LTXVScheduler | steps=**30**, max_shift=2.05, base_shift=0.95, terminal=**0.05**, stretch=true |
| 25 | KSamplerSelect | **euler** |
| 26 | RandomNoise | seed=888 |
| 27 | SamplerCustomAdvanced | — |

### 核心参数详解

**img_compression (38)**:
控制预处理时对 FLUX 输出图的压缩程度。值越高压缩越强，LTX 获得更多"创作自由度"。38 略高于基线版的 35，配合高 I2V strength 使用。

**I2V strength (0.85)**:
首帧锚定强度。范围 0-1：
- 0.0: 完全忽略首帧，LTX 自由生成 → 画面可能偏离
- 0.85: 紧锚定首帧，但保留 15% 自由度给运动 → **v2 推荐**
- 1.0: 死扣首帧 → 运动受限，画面可能僵硬

**LTXVScheduler**:
- `steps=30`: 比基线版多 5 步，更多去噪 → 画面更干净
- `terminal=0.05`: 去噪终点。越低越干净，但太低损失细节。0.05 是保守值
- `stretch=true`: 拉伸噪声调度，改善帧间一致性

**STGGuider**:
- `stg=1.0`: 标准强度。比优化版的 1.5 更保守，减少过度锐化
- `rescale=0.5`: 输出缩放因子，防止 STG 导致画面过曝

---

## 10. Group 8: LTX 输出 (节点 28-32)

| 节点 | 类型 | 关键参数 |
|------|------|----------|
| 28 | Set VAE Decoder Noise | timestep=**0.03**, scale=**0.015**, seed=42 |
| 29 | LTXVTiledVAEDecode | 2×2 tiles, overlap=6, **last_frame_fix=true** |
| 30 | CreateVideo | fps=**24** |
| 31 | PreviewImage | — |
| 32 | SaveVideo | prefix=`hypercar_flux_ltx`, mp4, h264 |

**VAE Decoder Noise (保守值)**:
- `timestep=0.03`: 噪声注入的时间步。比基线版 0.05 更低 → 更干净的输出
- `scale=0.015`: 噪声强度。比基线版 0.025 更低 → 减少残影和鬼影
- 这是一个**微妙的参数**: 太低会出现色带 (banding)，太高会出现残影。0.03/0.015 是经过实验的保守平衡点

**LTXVTiledVAEDecode**:
- 2×2 分块解码：将视频切成 4 块分别解码，大幅降低显存峰值
- `last_frame_fix=true`: I2V 强烈推荐，修复最后一帧的伪影

---

## 11. 参数速查表

### FLUX 阶段

| 参数 | 值 | 可调范围 |
|------|-----|----------|
| UNet 精度 | fp8_e4m3fn_fast | 不动 |
| T5 编码器 | fp8_e4m3fn (4.6GB) | 可换 fp16 (9.2GB) |
| FLUX steps | 28 | 20-35 |
| guidance | 4.0 | 3.5-4.5 |
| 分辨率 | 960×544 | 须被 8 整除 |
| seed | 777 | 任意 |

### 桥接阶段

| 参数 | 值 | 可调范围 |
|------|-----|----------|
| img_compression | 38 | 30-40 |
| I2V strength | 0.85 | 0.70-0.95 |

### LTX 阶段

| 参数 | 值 | 可调范围 |
|------|-----|----------|
| LTX steps | 30 | 20-35 |
| NAG scale | 10 | 8-14 |
| NAG alpha | 0.18 | 0.10-0.30 |
| NAG tau | 3.5 | 2.0-4.0 |
| STG stg | 1.0 | 0.5-1.5 |
| STG rescale | 0.5 | 0.3-0.7 |
| terminal | 0.05 | 0.02-0.10 |
| 帧数 | 73 (≈3秒) | 49-97 |
| fps | 24 | 16-30 |
| VAE noise timestep | 0.03 | 0.02-0.06 |
| VAE noise scale | 0.015 | 0.01-0.03 |
| STG block | "19" | "15"-"25" |

---

## 12. 版本对比

| 参数 | 基线版 | 优化版 | v2 ★ |
|------|--------|--------|------|
| 文件 | `flux_ltx_i2v.json` | `flux_ltx_i2v_optimized.json` | `flux_ltx_i2v_v2.json` |
| 节点数 | 32 | 33 | 33 |
| T5 编码器 | FP16 (9.2GB) | FP8 (4.6GB) | **FP8 (4.6GB)** |
| 引导方式 | CFG (cfg=1.0) | STG (stg=1.5) | **STG (stg=1.0)** |
| STG rescale | — | 0.7 | **0.5** |
| NAG scale | 12 | 13 | **10** |
| NAG alpha | 0.30 | 0.35 | **0.18** |
| NAG tau | 2.5 | 2.0 | **3.5** |
| I2V strength | 0.80 | 0.72 | **0.85** |
| LTX steps | 25 | 25 | **30** |
| terminal | 0.1 | 0.1 | **0.05** |
| VAE noise (t/s) | 0.05/0.025 | 0.06/0.03 | **0.03/0.015** |
| img_compression | 35 | 30 | **38** |
| 主题 | 羽毛球 | 超跑 | **超跑** |

### 参数调优方向

| 问题 | 调整 |
|------|------|
| 视频画面偏离首帧 | I2V strength ↑ (0.85→0.90) |
| 运动不够流畅 | NAG tau ↑ (3.5→4.0), terminal ↓ (0.05→0.03) |
| 画面有闪烁/伪影 | STG stg ↓ (1.0→0.5), VAE noise timestep ↑ |
| 运动太僵硬 | I2V strength ↓ (0.85→0.75), NAG scale ↑ |
| 画面不够清晰 | LTX steps ↑ (30→35), NAG alpha ↑ (0.18→0.25) |
| 显存不足 | 降分辨率 (960×544→768×432), 减帧数 (73→49) |

---

## 13. 调试指南

### 运行前检查

1. **模型文件齐全**: 投影层需在 `checkpoints/` 和 `text_encoders/` 各有一份
2. **ComfyUI 版本**: 0.25.0+, 已安装 `ComfyUI-LTXVideo` 和 `ComfyUI-GGUF`
3. **显存**: 24GB 卡需 `--lowvram` 或优化启动参数

### 常见错误

| 错误 | 原因 | 解决 |
|------|------|------|
| "投影层找不到" | 文件只在 text_encoders/ | 复制一份到 checkpoints/ |
| OOM | 显存不够 | 降分辨率 / `--novram` |
| 画面变色/闪烁 | STG 过高 | stg 降到 0.5 |
| 视频鬼影/残影 | VAE noise 太低 | timestep 调到 0.05 |
| 视频不跟运动 Prompt | NAG scale 太低 | 调到 12-14 |

### 优化启动命令 (24GB)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 \
  venv/bin/python main.py --listen 127.0.0.1 --lowvram
```
