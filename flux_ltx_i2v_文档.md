# FLUX + LTX I2V 组合工作流 — 完整节点与参数文档

> **当前推荐版本**: [`flux_ltx_i2v_v2.json`](docs/flux_ltx_i2v_v2.md) ★ (33 节点, STG 引导, 保守参数)
>
> **本文档基于**: `flux_ltx_i2v.json`（基线版, 32 节点, CFG 引导）— 节点结构和数据流仍适用，参数请以 v2 为准。

| 版本 | 文件 | 引导 | 步数 | 状态 |
|------|------|------|------|------|
| 基线版 | `flux_ltx_i2v.json` | CFG | 25 | 📦 本文档描述的版本 |
| 优化版 | `flux_ltx_i2v_optimized.json` | STG 1.5 | 25 | ⚠️ 实验性, 参数激进 |
| **v2** ★ | **`flux_ltx_i2v_v2.json`** | **STG 1.0** | **30** | ✅ **推荐使用** |

> **输出**: 960×544 MP4 视频 (~3秒 @ 24fps)

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
12. [调试指南](#12-调试指南)

---

## 1. 架构总览

这是一个**两阶段级联工作流**：

```
阶段 1: FLUX.1-DEV 文生图 (节点 1-11)
    └→ 根据文本 Prompt 生成一张高质量的初始帧图片

         ↓ (像素桥接)

阶段 2: LTX Video 2.3 图生视频 (节点 12-32)
    └→ 以初始帧为起点，生成连续的视频序列
```

**为什么这样设计？** FLUX 是目前最强的开源文生图模型之一，生成的单帧质量极高。LTX Video 2.3 是图生视频模型，需要一个起始帧。两者 VAE 互不兼容，所以必须走「FLUX latent → FLUX VAE 解码 → 像素图 → LTX VAE 编码 → LTX latent」的桥接路径。

### 关键数字

| 参数 | 值 | 说明 |
|------|-----|------|
| 分辨率 | 960×544 | 宽高都能被 32 整除，FLUX/LTX 双兼容 |
| FLUX 步数 | 28 | dpmpp_2m 采样器，高质量 |
| LTX 步数 | 25 | euler 采样器，平衡速度与质量 |
| 视频帧数 | 73 | @24fps ≈ 3.04 秒 |
| 总节点 | 32 | 8 个功能组 |

---

## 2. 数据流图

```
┌─────────────────────────────────────────────────────────────────────┐
│  GROUP 1: FLUX 模型加载                                              │
│                                                                     │
│  [1] UNETLoader ──→ [2] ModelSamplingFlux ──────────────┐           │
│       flux1-dev          max_shift=1.35                  │           │
│       fp8_e4m3fn_fast    base_shift=0.5                  │           │
│                                                          │           │
│  [3] DualCLIPLoader ──→ CLIP ──→ [5] [6]               │           │
│       clip_l + t5xxl                                       │           │
│                                                          │           │
│  [4] VAELoader ──→ VAE ──→ [10]                         │           │
│       ae.safetensors                                        │           │
└──────────────────────────────────────────────────────────│──────────┘
                                                          │
┌──────────────────────────────────────────────────────────│──────────┐
│  GROUP 2: FLUX Prompt & Conditioning                    │           │
│                                                          │           │
│  [5] CLIPTextEncodeFlux ──→ positive ──→ [9]           │           │
│       guidance=4.0                                              │           │
│                                                          │           │
│  [6] CLIPTextEncode ──→ [7] ConditioningZeroOut ──→ negative ──→ [9]
│                                                          │           │
│  [8] EmptySD3LatentImage ──→ latent ──→ [9]             │           │
│       960×544                                                │           │
└──────────────────────────────────────────────────────────│──────────┘
                                                          │
┌──────────────────────────────────────────────────────────│──────────┐
│  GROUP 3: FLUX 采样                                       ↓          │
│                                                                     │
│  [9] KSampler (dpmpp_2m + sgm_uniform, 28步, CFG=1.0)              │
│       │                                                             │
│       ↓ latent                                                      │
└─────────────────────────────────────────────────────────────────────┘
         │
┌────────↓────────────────────────────────────────────────────────────┐
│  GROUP 4: FLUX→LTX 桥接                                             │
│                                                                     │
│  [10] VAEDecode ──→ 像素图 ──→ [11] SaveImage (flux_frame_*.png)   │
│       │                            │                                │
│       │                            ↓                                │
│       │                   [20] LTXVPreprocess (img_compression=35)  │
│       │                            │                                │
│       │                            ↓ 预处理后的图                    │
│       │                           [22] LTXVImgToVideoConditionOnly   │
│       │                                strength=0.80                │
│       │                                │                            │
│       │              [21] EmptyLTXVLatentVideo (960×544, 73帧)       │
│       │                                │                            │
│       │                                ↓                            │
│       │                           latent ──→ [27]                   │
└───────┼─────────────────────────────────────────────────────────────┘
        │
┌───────┼─────────────────────────────────────────────────────────────┐
│  GROUP 5: LTX 模型加载                                               │
│                                                                     │
│  [12] LTXAVTextEncoderLoader (Gemma 3 12B + 投影) ──→ CLIP ──→ [13][14]
│                                                                     │
│  [16] UNETLoader ──→ [17] LTX2_NAG ──→ [18] LTXICLoRALoader ──→ MODEL
│        ltx-2.3-22b       scale=12          dynamic LoRA              │
│        mxfp8              alpha=0.30        strength=1.0              │
│                           tau=2.5                                     │
│                                                                     │
│  [19] VAELoader (LTX23_video_vae_bf16) ──→ VAE ──→ [22][28]        │
└─────────────────────────────────────────────────────────────────────┘
         │
┌────────↓────────────────────────────────────────────────────────────┐
│  GROUP 6: LTX Prompt & I2V                                          │
│                                                                     │
│  [13] CLIPTextEncode ──→ positive ──→ [15] LTXVConditioning        │
│                                                       │              │
│  [14] CLIPTextEncode ──→ negative ──→ [15]          │              │
│                                             │         │              │
│                                      positive   negative            │
│                                             │         │              │
│                                             ↓         ↓              │
│                                          [23] CFGGuider (cfg=1.0)   │
└─────────────────────────────────────────────────────────────────────┘
         │
┌────────↓────────────────────────────────────────────────────────────┐
│  GROUP 7: LTX 采样                                                   │
│                                                                     │
│  [24] LTXVScheduler (25步) ──→ SIGMAS ──→ [27]                     │
│  [25] KSamplerSelect (euler) ──→ SAMPLER ──→ [27]                  │
│  [26] RandomNoise (seed=888) ──→ NOISE ──→ [27]                    │
│                                                                     │
│  [27] SamplerCustomAdvanced ──→ latent ──→ [29]                    │
└─────────────────────────────────────────────────────────────────────┘
         │
┌────────↓────────────────────────────────────────────────────────────┐
│  GROUP 8: LTX 输出                                                   │
│                                                                     │
│  [28] Set VAE Decoder Noise ──→ VAE ──→ [29] LTXVTiledVAEDecode    │
│       timestep=0.05                                 2×2 tiles       │
│       scale=0.025                                   overlap=6       │
│                                                     last_frame_fix  │
│       │                                                             │
│       ↓ 像素帧                                                       │
│  [30] CreateVideo (24fps) ──→ [32] SaveVideo (mp4/h264)             │
│  [31] PreviewImage (首帧预览)                                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Group 1: FLUX 模型加载 (节点 1-4)

颜色标签: `📦 FLUX 模型加载 — flux1-dev + clip_l + t5xxl + ae`

这组负责加载 FLUX.1-DEV 所需的全部模型文件。

### 节点 1: UNETLoader — FLUX UNet

| 属性 | 值 |
|------|-----|
| 模型文件 | `flux1-dev.safetensors` |
| 精度 | `fp8_e4m3fn_fast` |
| 输出 | MODEL |

**UNet 是什么？** 扩散模型的核心网络，负责在 latent 空间中去噪。FLUX.1-DEV 的 UNet 是 DiT (Diffusion Transformer) 架构，参数量约 12B。

**参数详解：**

| 参数 | 当前值 | 作用 | 调大/调小的效果 |
|------|--------|------|----------------|
| `unet_name` | `flux1-dev.safetensors` | 选择 FLUX 模型文件 | 可选 `flux1-schnell.safetensors`（4步出图，但质量低）或其他微调版 |
| `weight_dtype` | `fp8_e4m3fn_fast` | 推理精度 | `default`=BF16（质量最高，~23GB 显存）；`fp8_e4m3fn_fast`（质量略降，~12GB 显存，速度快 30%）；`fp8_e5m2`（动态范围更大，但精度更低） |

**调试建议：**
- 遇到 OOM（显存不足）：保持 `fp8_e4m3fn_fast`
- 画面有异常噪点/伪影：切换到 `default` 看看是否是精度问题
- 想更快出图：已经是 fp8 fast，基本是最快选择了

---

### 节点 2: ModelSamplingFlux — Flux 调度器

| 属性 | 值 |
|------|-----|
| 输入 | MODEL (来自节点 1) |
| 输出 | MODEL (含采样参数) |

**这是什么？** FLUX 使用偏移-噪声调度（shift-noise schedule），通过调整信噪比分布来控制生成质量。这个节点将调度参数注入到模型中。

**参数详解：**

| 参数 | 当前值 | 范围 | 作用 |
|------|--------|------|------|
| `max_shift` | **1.35** | 0.5–3.0 | 高噪声偏移量。**影响最大**——决定高分辨率细节的噪声清除力度 |
| `base_shift` | **0.50** | 0.3–1.5 | 低噪声偏移量。影响低分辨率结构/构图 |
| `width` | **960** | — | 目标宽度，须被 32 整除 |
| `height` | **544** | — | 目标高度，须被 32 整除 |

**调参效果：**

| `max_shift` | 效果 |
|-------------|------|
| ↑ 提高到 1.8–2.5 | 更多细节纹理（可能过锐、产生噪点）。适合风景/纹理密集型图像 |
| ↓ 降低到 0.8–1.0 | 更平滑干净（可能略显模糊）。适合简洁画面 |
| **1.35（当前）** | 细节与平滑的均衡点，适合运动场景 |

| `base_shift` | 效果 |
|--------------|------|
| ↑ 提高 0.7–1.0 | 更强烈的整体构图，更大胆的布局 |
| ↓ 降低 0.3–0.4 | 保守的构图，更稳定（但可能平淡） |

**调试建议：**
- 图片细节过度（锐化感/噪点）：降低 `max_shift` 到 1.0
- 图片模糊/细节不足：提高 `max_shift` 到 1.8
- 构图像素混乱：降低 `base_shift` 到 0.4
- 分辨率变更时同步更新 `width`/`height`

---

### 节点 3: DualCLIPLoader — FLUX 双 CLIP 加载

| 属性 | 值 |
|------|-----|
| CLIP 1 | `clip_l.safetensors` (小 CLIP, ~246MB) |
| CLIP 2 | `t5xxl_fp16.safetensors` (T5-XXL, ~9.4GB) |
| type | `flux` |
| device | `default` |
| 输出 | CLIP (联合文本编码器) |

**CLIP 是什么？** 文本编码器，将你的 Prompt 文字转化为模型能理解的数学向量（embedding）。FLUX 使用**双编码器架构**：clip_l 负责简短关键词理解，T5-XXL 负责长文本深层语义。

**参数详解：**

| 参数 | 当前值 | 说明 |
|------|--------|------|
| `clip_name1` | `clip_l.safetensors` | 小 CLIP，处理关键词级语义。不可换 |
| `clip_name2` | `t5xxl_fp16.safetensors` | T5-XXL，处理长句/段落级语义。可换 `t5xxl_fp8_e4m3fn.safetensors`（更低显存） |
| `type` | `flux` | 编码器类型。`flux`=标准双编码器，不可改 |
| `device` | `default` | 设备分配策略。`default`=GPU；`cpu`=CPU 推理（极慢） |

**调试建议：**
- T5-XXL 加载 OOM：换 `t5xxl_fp8_e4m3fn.safetensors`
- 语义理解不够深刻：确保用的是 T5-XXL（不是 T5-small/base），这个是 FLUX 的核心优势
- 不要在 `type` 里选其他值——FLUX 只兼容 `flux` 类型

---

### 节点 4: VAELoader — FLUX VAE

| 属性 | 值 |
|------|-----|
| 模型文件 | `ae.safetensors` |
| 输出 | VAE |

**VAE 是什么？** Variational Autoencoder，负责 latent 空间 ↔ 像素空间的编解码。FLUX 的 VAE 将 960×544 像素图压缩到约 120×68 的 latent。

| 参数 | 说明 |
|------|------|
| `vae_name` | FLUX 专用 `ae.safetensors`。不能用 SD 的 VAE，架构不同 |

---

## 4. Group 2: FLUX Prompt 与 Conditioning (节点 5-8)

颜色标签: `📝 FLUX Prompt & Conditioning — CLIPTextEncodeFlux guidance=4.0`

这组把你的文字 Prompt 转化成采样器能理解的条件信号。

### 节点 5: CLIPTextEncodeFlux — FLUX 正向 Prompt

| 属性 | 值 |
|------|-----|
| 输入 | CLIP (来自节点 3) |
| 输出 | CONDITIONING (正向) |
| guidance | 4.0 |

**这是整个工作流中最重要的节点之一。** Prompt 的质量直接决定输出质量。

**参数详解：**

| 参数 | 当前值 | 说明 |
|------|--------|------|
| `clip_l` | 简短关键词 (1行) | clip_l 擅长处理标签式关键词：主体、动作、场景类型 |
| `t5xxl` | 详细场景描述 (5行段落) | T5-XXL 擅长处理自然语言长文本：光线、构图、氛围、摄影参数 |
| `guidance` | **4.0** | FLUX 引导强度。**非常关键！** |

**Prompt 策略：**
- **clip_l（短）**: `"professional badminton player mid-air smashing shuttlecock, sports action photo, indoor court"`
- **t5xxl（长）**: 描述了完整的场景——运动员蓝色球衣、跳起扣杀、羽毛球的运动轨迹、观众反应、灯光、地板材质、摄影参数

**guidance 调参效果：**

| guidance | 效果 |
|----------|------|
| ↑ 6.0–10.0 | 更强遵循 Prompt，但可能过饱和、对比度过高、伪影增多 |
| ↓ 1.5–3.0 | 更自由的创作，但可能偏离 Prompt 描述 |
| **4.0（当前）** | FLUX 推荐的均衡值 |

**调试建议：**
- 图片内容不符合描述：先检查 Prompt 是否清楚，再适当提高 guidance 到 5-6
- 画面过于"油腻"、色彩失真：降低 guidance 到 2.5-3.5
- clip_l 保持 1 行关键词即可，不要塞长句子——它不擅长
- t5xxl 可以写很长（512 tokens），尽情描述场景细节

---

### 节点 6: CLIPTextEncode — FLUX 负向 Prompt

| 属性 | 值 |
|------|-----|
| 输入 | CLIP (来自节点 3) |
| 输出 | CONDITIONING (负向，未归零) |

**什么是负向 Prompt？** 告诉模型「不要生成这些」。是一组排除项。

当前负向 Prompt:
```
blurry, low quality, distorted, ugly, bad anatomy, extra limbs,
missing fingers, deformed hands, watermark, text, oversaturated,
cartoon, 3d render, painting, disfigured, out of focus, bad lighting
```

**调试建议：**
- 出现手部畸形：增加 `bad hands, fused fingers, extra digits`
- 画面过于卡通：增加 `illustration, anime, digital art`
- 默认负向已经覆盖了大部分常见问题，一般不需要大改

---

### 节点 7: ConditioningZeroOut — 负向归零

| 属性 | 值 |
|------|-----|
| 输入 | 负向 CONDITIONING (来自节点 6) |
| 输出 | 归零后的 CONDITIONING |

**这是什么？** FLUX 使用 CFG=1.0（无负向引导），所有引导力都来自 guidance 参数。这个节点将负向条件信号归零，确保不影响采样。

**为什么需要？** FLUX 的 guidance 机制不同于传统 SD CFG。如果不归零，会干扰 guidance 的工作方式。

**调试：** 这个节点不需要调参。保持不动即可。

---

### 节点 8: EmptySD3LatentImage — FLUX 画布

| 属性 | 值 |
|------|-----|
| width | 960 |
| height | 544 |
| batch_size | 1 |
| 输出 | LATENT (空白画布) |

**这是采样的起点**——一个充满随机噪声的 latent 画布，采样器会逐步将它去噪成图像。

| 参数 | 说明 |
|------|------|
| `width` × `height` | 必须与 ModelSamplingFlux (节点 2) 保持一致，且能被 32 整除 |
| `batch_size` | 一次生成几张图。>1 会消耗更多显存 |

**为什么用 SD3 Latent？** FLUX 的 latent 格式与 SD3 兼容（16 通道，而非 SD1.5 的 4 通道）。

---

## 5. Group 3: FLUX 采样 (节点 9)

颜色标签: `🎨 FLUX 采样 — dpmpp_2m + sgm_uniform, 28步`

### 节点 9: KSampler — FLUX 采样器

| 属性 | 值 |
|------|-----|
| 输入 | MODEL + positive + negative + latent_image |
| 输出 | LATENT (去噪后的图像 latent) |

**KSampler 做什么？** 这是扩散模型的核心执行器——它接收噪声画布，在条件信号的引导下，按照采样算法一步步去噪，最终输出干净 latent。

**参数详解：**

| 参数 | 当前值 | 说明 |
|------|--------|------|
| `seed` | `777 (randomize)` | 随机种子。固定值=可复现，`randomize`=每次不同 |
| `steps` | **28** | 采样步数。越多越精细 |
| `cfg` | **1.0** | CFG scale。FLUX 固定 1.0，用 guidance 代替 |
| `sampler_name` | **`dpmpp_2m`** | 采样算法 |
| `scheduler` | **`sgm_uniform`** | 噪声调度策略 |
| `denoise` | **1.0** | 降噪强度。1.0=从头生成，<1.0=保留部分原图（I2I 用） |

**steps 调参效果：**

| steps | 效果 |
|-------|------|
| 20 | 可接受，速度最快。细节可能不够 |
| 28（当前） | 高质量，dpmpp_2m 的 sweet spot |
| 40–50 | 理论上更精细，但 dpmpp_2m 下与 28 差距极小，不值得 |

**sampler 对比：**

| 采样器 | 特点 |
|--------|------|
| `dpmpp_2m` ⭐ | **推荐**。快速收敛，28 步足够 |
| `euler` | 简单直接，需要更多步数（40+） |
| `dpmpp_3m_sde` | 纹理更细腻，但可能在 FLUX 上不稳定 |
| `dpm++ 2M SDE` | 带随机性，适合创意生成 |

**scheduler 对比：**

| 调度器 | 特点 |
|--------|------|
| `sgm_uniform` ⭐ | **推荐**。均匀分布噪声步长，稳定 |
| `karras` | 低噪声步更多，细节更好但速度慢 |
| `exponential` | 高噪声步更多，构图更大胆 |
| `ddim_uniform` | DDIM 风格的均匀分布 |

**调试建议：**
- 先固定 seed 测试不同 Prompt，满意后再 randomize
- 图片噪点/伪影：增加 steps 到 35，或切换 sampler
- 生成太慢：降到 20-25 步，dpmpp_2m 仍能出不错的结果
- `denoise` 保持 1.0——这是 T2I，需要从头生成

---

## 6. Group 4: FLUX→LTX 桥接 (节点 10-11)

颜色标签: `🔄 FLUX→LTX 桥接 — VAE Decode → Preprocess → ImgToVideo`

这是工作流的**关键枢纽**。两个模型的 VAE 不互通，必须经过像素空间桥接。

### 节点 10: VAEDecode — FLUX VAE 解码

| 属性 | 值 |
|------|-----|
| 输入 | FLUX latent (来自节点 9) + FLUX VAE (来自节点 4) |
| 输出 | IMAGE (像素图, 960×544) |

**做什么？** 将 FLUX 生成的 latent 张量解码为可见的像素图像。这张图将成为 LTX 视频的**起始帧**。

**输出分叉：**
- → 节点 11 (SaveImage)：保存为 `flux_frame_*.png`，方便检查 FLUX 阶段输出
- → 节点 20 (LTXVPreprocess)：送入 LTX 预处理

**调试：** 如果最终视频起始帧不对，先检查这里保存的 `flux_frame_*.png`，确认 FLUX 阶段是否正常。

---

### 节点 11: SaveImage — 保存 FLUX 图片

| 属性 | 值 |
|------|-----|
| 文件前缀 | `flux_frame` |
| 保存路径 | `ComfyUI/output/flux_frame_00001.png` |

**用途：** 诊断标记——保存 FLUX 生成的起始帧，方便对比调试。如果视频有问题，先检查这张图。

---

## 7. Group 5: LTX 模型加载 (节点 12-19)

颜色标签: `📦 LTX 模型加载 — Gemma编码 + LTX UNet + NAG + LoRA + VAE`

### 节点 12: LTXAVTextEncoderLoader — LTX 文本编码器

| 属性 | 值 |
|------|-----|
| 文本编码器 | `gemma_3_12B_it_fp8_scaled.safetensors` (~12.3GB) |
| 投影层 | `ltx-2.3_text_projection_bf16.safetensors` (~2.2GB) |
| device | `default` |
| 输出 | CLIP (Gemma + 投影层) |

**这是什么？** LTX Video 2.3 使用 Google Gemma 3 12B 作为文本编码器（不是 T5！）。投影层将 Gemma 的 3840 维输出映射到 LTX 需要的 4096 维（video）+ 2048 维（audio）。

**为什么必须用这个节点？** 普通的 CLIPLoader 只能加载单文件，缺少投影层，会导致维度不匹配错误（3840 vs 4096）。

**调试：**
- 报错 `size mismatch` 或 `mat1 and mat2 shapes`: 检查投影层路径是否正确
- 不要尝试用 T5 替代 Gemma——LTX 2.3 专属 Gemma

---

### 节点 13: CLIPTextEncode — LTX 正向 Prompt

| 属性 | 值 |
|------|-----|
| 输入 | CLIP (来自节点 12) |
| 输出 | CONDITIONING (正向) |

**LTX Prompt 策略（与 FLUX 互补）：**

FLUX Prompt 描述的是**静态画面**（谁在哪、什么动作瞬间、灯光构图）。
LTX Prompt 描述的是**运动过程**（如何落地、如何移动、镜头如何跟踪）。

当前 LTX Prompt:
```
A badminton player continues the rally after a smash.
The player lands gracefully, then shuffles sideways with rapid footwork...
The opponent hits a clear shot high...
The camera pans to follow the shuttlecock...
```

**关键技巧：**
- 用**动作动词**: lands, shuffles, leaps, backpedaling, tracks
- 描述**时间顺序**: "continues the rally after a smash" → "then shuffles" → "The opponent hits" → "the camera pans"
- 避免静态描述（由 FLUX 覆盖了）：不要重复颜色、服装、场地
- 加入镜头运动：`camera pans, tracks`, 这会影响 LTX 对场景变化的理解

---

### 节点 14: CLIPTextEncode — LTX 负向 Prompt

| 属性 | 值 |
|------|-----|
| 输入 | CLIP (来自节点 12) |
| 输出 | CONDITIONING (负向) |

专门排除视频生成中的常见问题：
```
blurry, jittery, low quality, distorted, bad motion,
static image, still frame, no movement, ugly, watermark,
text, stop motion, choppy, jerky, teleporting, morphing,
warping, flickering, inconsistent, ghosting, motion blur, frame skip
```

**调试：**
- 视频不动（静止画面）：加强 `static image, still frame, no movement`
- 画面闪烁：加强 `flickering, inconsistent`
- 物体变形/蠕动：加强 `morphing, warping, teleporting`
- 残影/鬼影：加强 `ghosting, motion blur`

---

### 节点 15: LTXVConditioning — LTX 条件包装

| 属性 | 值 |
|------|-----|
| frame_rate | 24.0 |
| 输入 | positive + negative CONDITIONING |
| 输出 | 带帧率信息的 positive + negative |

**做什么？** 将帧率信息嵌入条件信号，确保 LTX 生成流畅一致的运动。`frame_rate` 必须与 `CreateVideo` 的 `fps` 一致。

| frame_rate | 效果 |
|------------|------|
| 24 ⭐ | 电影感，当前值 |
| 30 | 更流畅（电视/游戏），但需要更多帧 |
| 60 | 超流畅，但同样时间需要 2.5× 帧数 |

---

### 节点 16: UNETLoader — LTX UNet

| 属性 | 值 |
|------|-----|
| 模型 | `ltx-2.3-22b-distilled-1.1_transformer_only_mxf8_block32.safetensors` |
| 精度 | `default` |
| 输出 | MODEL |

**这是什么？** LTX Video 2.3 的扩散 Transformer，22B 参数，使用蒸馏 + MXFP8 量化。这是整个工作流中最大的模型（~23GB）。

**精度说明：**
- `default` = 自动，会按模型原始精度加载
- MXFP8 是 MicroXNet FP8 格式，块大小 32，专门优化的视频生成精度
- 不要换用 `fp8_e4m3fn_fast`——LTX 对精度敏感

**调试：**
- 如果显存不够（<24GB），这个模型可能会 OOM。考虑降低分辨率或帧数

---

### 节点 17: LTX2_NAG — NAG 引导

| 属性 | 值 |
|------|-----|
| 输入 | MODEL (来自节点 16) |
| 输出 | MODEL (含 NAG 参数) |

**NAG 是什么？** Noise-Aware Guidance，LTX Video 的专属引导机制。与传统的 CFG 不同，NAG 直接在去噪过程中注入 Prompt 信号，更高效且更稳定。当前工作流 CFG=1.0，完全依赖 NAG。

**参数详解：**

| 参数 | 当前值 | 范围 | 作用 |
|------|--------|------|------|
| `nag_scale` | **12.0** | 5–20 | **最重要的参数**。引导强度，越高越跟随 Prompt |
| `nag_alpha` | **0.30** | 0.1–0.8 | 锐度/对比度增强。越高画面越锐，但可能产生噪点 |
| `nag_tau` | **2.5** | 1.0–5.0 | 时间平滑度。越高帧间过渡越平滑，但可能轻微模糊 |

**调参效果：**

| nag_scale | 效果 |
|-----------|------|
| ↑ 15–18 | 更严格遵循 Prompt，但可能产生伪影、运动僵硬 |
| ↓ 6–10 | 更自由/自然的运动，但可能偏离 Prompt 意图 |
| **12（当前）** | 运动自然 + Prompt 遵循的良好平衡 |

| nag_alpha | 效果 |
|-----------|------|
| ↑ 0.5–0.7 | 更锐利的画面，更多纹理。风险：噪点、过度锐化 |
| ↓ 0.1–0.2 | 更柔和，可能轻微模糊 |
| **0.30（当前）** | 适中锐度 |

| nag_tau | 效果 |
|---------|------|
| ↑ 3.5–4.5 | 超平滑的运动，适合慢速镜头。风险：运动模糊 |
| ↓ 1.0–1.5 | 更快的运动响应。风险：画面抖动、闪烁 |
| **2.5（当前）** | 运动平滑与响应性的平衡 |

**调试建议：**
- 视频不动/不符合 Prompt：先提高 `nag_scale` 到 15
- 画面闪烁：提高 `nag_tau` 到 3.5
- 画面太模糊：提高 `nag_alpha` 到 0.4
- 异常伪影/噪点：降低 `nag_scale` 和 `nag_alpha`

---

### 节点 18: LTXICLoRALoaderModelOnly — 动态 LoRA

| 属性 | 值 |
|------|-----|
| LoRA 文件 | `ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors` |
| strength_model | 1.0 |
| 输出 | MODEL (含 LoRA) |

**LoRA 是什么？** Low-Rank Adaptation，一种轻量级模型微调。这个 LoRA 专门增强 LTX 的动态运动质量。

| 参数 | 说明 |
|------|------|
| `lora_name` | 动态增强 LoRA，提升运动自然度 |
| `strength_model` | 1.0 = 完整效果。降低会减弱 LoRA 影响 |

**调试：**
- 运动不自然/僵硬：确保 LoRA 已加载且 strength=1.0
- 可以在 0.5–1.5 之间微调，但不推荐超过 1.2

---

### 节点 19: VAELoader — LTX Video VAE

| 属性 | 值 |
|------|-----|
| 模型 | `LTX23_video_vae_bf16.safetensors` (~1.4GB) |
| 输出 | VAE (×2: 给 I2V 条件 + 给解码器) |

**这是 LTX 的视频 VAE**，与 FLUX 的图像 VAE (节点 4) 完全不兼容。它将 960×544×73 的像素帧压缩/解压为视频 latent。

**关键：** 这个 VAE 输出**两路**：
- → 节点 22 (LTXVImgToVideoConditionOnly)：用于编码起始帧
- → 节点 28 (Set VAE Decoder Noise)：用于解码 latent → 视频帧

---

## 8. Group 6: LTX Prompt 与 I2V (节点 13-15, 20-23)

颜色标签: `📝 LTX Prompt & I2V — Gemma编码 → ImgToVideoConditionOnly`

### 节点 20: LTXVPreprocess — 图片预处理

| 属性 | 值 |
|------|-----|
| 输入 | 像素图 (来自节点 10/VAEDecode) |
| img_compression | 35 |
| 输出 | 预处理后的图 |

**做什么？** 对 FLUX 输出的图片进行 LTX 兼容的预处理。`img_compression` 控制图片在 VAE 编码前的压缩程度，影响 LTX 如何"看到"起始帧。

| img_compression | 效果 |
|-----------------|------|
| ↑ 50–80 | 更多压缩，起始帧更"模糊引导"，给视频更多自由（但可能偏离原图） |
| ↓ 10–25 | 更少压缩，起始帧更精确（但约束太强，运动可能不自然） |
| **35（当前）** | 合理的平衡——保留原图结构 + 允许运动自由度 |

**调试建议：**
- 视频起始几帧与原图差异大：降低到 25
- 视频太僵硬/不动：提高到 50
- 这个值很少需要调整，35 适用于大多数场景

---

### 节点 21: EmptyLTXVLatentVideo — LTX 视频画布

| 属性 | 值 |
|------|-----|
| width | 960 |
| height | 544 |
| length | 73 帧 |
| batch_size | 1 |
| 输出 | LATENT (噪声音视频 latent) |

**length=73 帧 @ 24fps ≈ 3.04 秒**。这是视频的"空白画布"，充满噪声，待采样器去噪。

| 参数 | 说明 |
|------|------|
| `length` | 帧数。更多=更长视频，但也消耗更多显存和时间 |
| 分辨率 | 必须与 FLUX 阶段一致（960×544） |

**帧数计算：**
- 73 帧 / 24fps = 3.04 秒
- 想生成 2 秒视频：length = 48
- 想生成 5 秒视频：length = 121（需要 ~2× 显存和时间）

**调试：**
- 如果显存溢出，优先减少 length（比如降到 49 帧 ≈ 2秒）
- 最后几帧突然变差：尝试减少帧数，或增加 `last_frame_fix`（节点 29）

---

### 节点 22: LTXVImgToVideoConditionOnly — 图生视频条件

| 属性 | 值 |
|------|-----|
| 输入 | VAE + 预处理图 + 视频 latent |
| strength | 0.80 |
| bypass | false |
| 输出 | LATENT (含起始帧条件的视频 latent) |

**这是 I2V 的核心！** 它将起始帧图片编码到视频 latent 的开头部分，约束前几帧必须接近原图。

**参数详解：**

| 参数 | 当前值 | 范围 | 作用 |
|------|--------|------|------|
| `strength` | **0.80** | 0.0–1.0 | **关键参数。** 起始帧的约束强度 |
| `bypass` | **false** | — | true=完全跳过 I2V 条件，变成纯 T2V |

| strength | 效果 |
|----------|------|
| ↑ 0.90–1.00 | 前几帧几乎完全复制原图，过渡更平滑但自由度低 |
| ↓ 0.50–0.70 | 更多运动自由，但可能偏离原图/出现突变 |
| **0.80（当前）** | 保留 20% 噪声自由度，让动作自然展开 |

**调试建议：**
- 视频开头与原图差异太大：提高 strength 到 0.90
- 视频前几帧定格不动：降低 strength 到 0.70
- 想要纯 T2V（无起始图）：设置 bypass=true
- 这是调 I2V 效果最常用的参数

---

### 节点 23: CFGGuider — CFG 引导器

| 属性 | 值 |
|------|-----|
| 输入 | MODEL + positive + negative |
| cfg | 1.0 |
| 输出 | GUIDER |

**CFG=1.0 意味着不使用传统 CFG 引导。** LTX Video 依赖 NAG (节点 17) 而不是 CFG。CFGGuider 仅作为容器将条件信号传给采样器。

| cfg | 效果 |
|-----|------|
| 1.0（当前） | 关闭 CFG，纯 NAG 引导 |
| 1.5–2.0 | 加入轻微 CFG，增加 Prompt 遵循度（但可能引入 CFG 烧灼感） |
| >3.0 | 不推荐——LTX 上 CFG 容易导致颜色偏移和运动异常 |

---

## 9. Group 7: LTX 采样 (节点 21-27)

颜色标签: `🎬 LTX 采样 — Scheduler + Sampler + Noise → SamplerCustomAdvanced`

### 节点 24: LTXVScheduler — LTX 调度器

| 属性 | 值 |
|------|-----|
| steps | 25 |
| max_shift | 2.05 |
| base_shift | 0.95 |
| stretch | true |
| terminal | 0.1 |
| 输出 | SIGMAS (噪声调度序列) |

**这是什么？** LTX Video 的专用噪声调度器，生成 25 个递减的 sigma 值（噪声水平），告诉采样器每一步清除多少噪声。

**参数详解：**

| 参数 | 当前值 | 范围 | 作用 |
|------|--------|------|------|
| `steps` | **25** | 10–50 | 采样步数。越多越精细，越慢 |
| `max_shift` | **2.05** | 1.0–3.0 | 高噪声端偏移。影响视频的全局结构/构图 |
| `base_shift` | **0.95** | 0.5–2.0 | 低噪声端偏移。影响视频细节纹理 |
| `stretch` | **true** | boolean | 拉伸 sigma 分布。true=更多步数在低噪声区（细节打磨） |
| `terminal` | **0.1** | 0.0–1.0 | 终端 sigma 值。控制最后一步的剩余噪声量 |

**steps 调参：**

| steps | 效果 |
|-------|------|
| 15–20 | 快速预览，运动可能不完全 |
| **25（当前）** | 质量/速度最佳平衡点 |
| 30–40 | 理论上更精细，但 LTX 上 >30 收益递减明显 |

**max_shift 调参：**

| max_shift | 效果 |
|-----------|------|
| ↑ 2.5–3.0 | 更强的全局结构，但可能产生幻觉/伪影 |
| ↓ 1.5–1.8 | 更保守的生成 |
| **2.05（当前）** | LTX 推荐值 |

**stretch 的效果：**
- `true`：更多步数集中在低噪声区 → 纹理细节更好，适合 I2V
- `false`：均匀分布 → 适合 T2V 或快速预览

**terminal 的效果：**
- `0.0`：最后一步去噪到 0（理论最干净，但可能不自然）
- **0.1（当前）**：保留微量噪声（更自然的纹理，减少 banding）
- `0.3–0.5`：明显噪声残留（除非特殊效果，否则不推荐）

**调试建议：**
- 视频细节不足：增加 steps 到 30，保持 stretch=true
- 生成太慢：减少 steps 到 20
- 画面有异常纹理：降低 max_shift 到 1.7
- 色带/条带：适当增加 terminal 到 0.15

---

### 节点 25: KSamplerSelect — 采样器选择

| 属性 | 值 |
|------|-----|
| sampler_name | euler |
| 输出 | SAMPLER |

**euler** 是 LTX Video 的推荐采样器——简单、稳定、快速。不需要像 FLUX 那样使用高阶采样器。

| 可选采样器 | 说明 |
|-----------|------|
| `euler` ⭐ | 推荐。简单高效，25 步够用 |
| `euler_ancestral` | 带随机注入，更有创意但更不可控 |
| `dpmpp_2m` | 可尝试，但 LTX 上收益不大 |
| `ddim` | 较旧的算法，可用但不推荐 |

---

### 节点 26: RandomNoise — 随机噪声

| 属性 | 值 |
|------|-----|
| noise_seed | 888 (randomize) |
| 输出 | NOISE |

**做什么？** 生成初始的随机噪声张量。这个噪声 + 调度器 = 视频生成的基础随机性来源。

**seed 的作用：**
- 固定 seed = 同样设定下可复现相同视频
- `randomize` = 每次随机不同结果
- FLUX 用 seed=777，LTX 用 seed=888 —— 两者独立随机，即使固定也能有多样性

**调试：** 如果对某个结果满意想复现——记下 seed 值，下次固定它。

---

### 节点 27: SamplerCustomAdvanced — LTX 高级采样器

| 属性 | 值 |
|------|-----|
| 输入 | NOISE + GUIDER + SAMPLER + SIGMAS + LATENT |
| 输出 | LATENT (去噪后的视频 latent) |

**这是 LTX 视频生成的实际执行器。** 它将所有组件（噪声、引导器、采样器、调度器、条件 latent）组合在一起，执行 25 步去噪过程。

**为什么用 CustomAdvanced 而不是 KSampler？** LTX 使用独立的调度器（节点 24）+ 独立的采样器选择器（节点 25）+ CFGGuider，需要 CustomAdvanced 将它们合在一起。这与 FLUX 阶段用 KSampler（一体化）不同。

**输出分叉：**
- `output` → 节点 29 (VAE 解码) — 主输出
- `denoised_output` → 未使用（可直接解码为最终帧的预测，通常不用于视频）

---

## 10. Group 8: LTX 输出 (节点 28-32)

颜色标签: `🖼️ LTX 输出 — DecoderNoise + TiledDecode → Video → Save`

### 节点 28: Set VAE Decoder Noise — VAE 解码噪声

| 属性 | 值 |
|------|-----|
| 输入 | VAE (来自节点 19) |
| timestep | 0.05 |
| scale | 0.025 |
| seed | 42 (randomize) |

**做什么？** 在 VAE 解码时添加微量噪声，作用是：
1. **减少色带（banding）** —— 大面积纯色区域（如墙壁、地板）容易出现阶梯状的色彩断层
2. **增加微纹理** —— 让画面看起来更自然，不虚假平滑

**参数详解：**

| 参数 | 当前值 | 范围 | 作用 |
|------|--------|------|------|
| `timestep` | **0.05** | 0.0–0.5 | 噪声的时间步。控制噪声的"粗细" |
| `scale` | **0.025** | 0.0–0.1 | 噪声强度。越高纹理越多 |

| 调整 | 效果 |
|------|------|
| timestep↑ / scale↑ | 更多纹理，但可能明显噪点 |
| timestep↓ / scale↓ | 更干净的输出，但可能出现色带 |
| 全部设 0 | 关闭 DecoderNoise（不推荐——容易出现 banding） |
| **当前值** | 刚好在人眼不可感知的水平——减少色带但看不出加了噪声 |

**调试：**
- 视频有可见噪点/雪花：降低 scale 到 0.01
- 墙壁/地板出现色带（渐变阶梯）：提高 scale 到 0.04，timestep 到 0.08
- 一般不需要调整，当前值是经验最优

---

### 节点 29: LTXVTiledVAEDecode — 分块 VAE 解码

| 属性 | 值 |
|------|-----|
| 输入 | VAE + 视频 latent |
| horizontal_tiles | 2 |
| vertical_tiles | 2 |
| overlap | 6 |
| last_frame_fix | true |

**做什么？** 将视频 latent 解码为像素帧。由于视频 latent 很大（960×544×73 帧），一次性解码可能超出显存。分块解码将每帧切成 2×2=4 块分别解码，再拼接。

**参数详解：**

| 参数 | 当前值 | 作用 |
|------|--------|------|
| `horizontal_tiles` | 2 | 水平分块数。2=每块 480×272 |
| `vertical_tiles` | 2 | 垂直分块数。总共 2×2=4 块 |
| `overlap` | 6 | 块之间的重叠像素。越大拼接越无缝，越慢 |
| `last_frame_fix` | **true** | **I2V 推荐开启！** 修复最后一帧的伪影问题 |

**分块策略：**

| tiles | 适用场景 |
|-------|---------|
| 1×1 | 显存足够时（更快，无拼接缝） |
| 2×2（当前） | 标准选择，显存友好 |
| 3×3 | 显存紧张时，但拼接缝风险增加 |

**overlap 的效果：**
- ↑ 8–12：更少的拼接缝，但更慢
- ↓ 2–4：更快的解码，但可能看到块边界
- **6（当前）**：标准值

**last_frame_fix 的重要性：**
- I2V 模式下，最后一帧有时会出现异常（模糊、伪影）
- `true`：应用专门的后处理修复最后一帧
- 对于 I2V 工作流，强烈建议保持 `true`

**输出分叉：**
- → 节点 30 (CreateVideo)：合成视频
- → 节点 31 (PreviewImage)：显示首帧预览

---

### 节点 30: CreateVideo — 帧合成视频

| 属性 | 值 |
|------|-----|
| 输入 | IMAGE (73 帧, 来自节点 29) |
| fps | 24.0 |
| 输出 | VIDEO |

**fps 必须与 LTXVConditioning (节点 15) 的 frame_rate 一致！**

| fps | 效果 |
|-----|------|
| 24 ⭐ | 电影标准帧率，当前值 |
| 30 | 电视/网络视频标准 |
| 60 | 超流畅慢动作（但 73 帧只有 1.2 秒） |

---

### 节点 31: PreviewImage — 首帧预览

**辅组节点**，在 ComfyUI 界面中显示视频的第一帧。方便快速检查生成结果而不需要等待视频编码完成。

---

### 节点 32: SaveVideo — 保存视频

| 属性 | 值 |
|------|-----|
| filename_prefix | `badminton_flux_ltx` |
| format | `mp4` |
| codec | `h264` |

输出路径: `ComfyUI/output/badminton_flux_ltx_00001.mp4`

| 参数 | 说明 |
|------|------|
| `format` | `mp4`（兼容性最好）或 `webm`/`mov` |
| `codec` | `h264`（通用）或 `h265`（体积更小但兼容性稍差） |

---

## 11. 参数速查表

### 最常调参数 Top 10（按影响从大到小）

| 优先级 | 节点 | 参数 | 默认值 | 调大效果 | 调小效果 |
|--------|------|------|--------|---------|---------|
| ⭐⭐⭐ | 5 | guidance | 4.0 | 更遵循 FLUX Prompt（可能过饱和） | 更自由（可能偏离） |
| ⭐⭐⭐ | 22 | strength | 0.80 | 更贴合起始帧（可能僵硬） | 更多运动自由（可能突变） |
| ⭐⭐⭐ | 17 | nag_scale | 12.0 | 更遵循 LTX Prompt（可能伪影） | 更自然（可能偏离意图） |
| ⭐⭐ | 9 | steps (FLUX) | 28 | 更精细（更慢） | 更快（可能粗糙） |
| ⭐⭐ | 24 | steps (LTX) | 25 | 更精细（更慢） | 更快（可能不完整） |
| ⭐⭐ | 2 | max_shift (FLUX) | 1.35 | 更多细节纹理 | 更平滑干净 |
| ⭐⭐ | 17 | nag_alpha | 0.30 | 更锐利（可能噪点） | 更柔和（可能模糊） |
| ⭐ | 20 | img_compression | 35 | 更自由的运动 | 更贴合原图 |
| ⭐ | 17 | nag_tau | 2.5 | 更平滑（可能模糊） | 更快响应（可能抖动） |
| ⭐ | 28 | scale (DecoderNoise) | 0.025 | 更多纹理（可能噪点） | 更干净（可能色带） |

### 完整参数表

| 节点 | 参数 | 值 | 类型 |
|------|------|-----|------|
| 1 | unet_name | `flux1-dev.safetensors` | 模型选择 |
| 1 | weight_dtype | `fp8_e4m3fn_fast` | 精度 |
| 2 | max_shift | 1.35 | 调度 |
| 2 | base_shift | 0.50 | 调度 |
| 2 | width | 960 | 分辨率 |
| 2 | height | 544 | 分辨率 |
| 3 | clip_name1 | `clip_l.safetensors` | 模型选择 |
| 3 | clip_name2 | `t5xxl_fp16.safetensors` | 模型选择 |
| 3 | type | `flux` | 编码器类型 |
| 4 | vae_name | `ae.safetensors` | 模型选择 |
| 5 | clip_l | (短关键词) | Prompt |
| 5 | t5xxl | (长描述段落) | Prompt |
| 5 | guidance | 4.0 | 引导强度 |
| 6 | text | (负向 Prompt) | Prompt |
| 8 | width×height | 960×544 | 分辨率 |
| 9 | seed | 777/randomize | 随机种子 |
| 9 | steps | 28 | 采样步数 |
| 9 | cfg | 1.0 | CFG scale |
| 9 | sampler_name | `dpmpp_2m` | 采样器 |
| 9 | scheduler | `sgm_uniform` | 调度器 |
| 9 | denoise | 1.0 | 降噪强度 |
| 12 | text_encoder | `gemma_3_12B_it_fp8_scaled` | 模型选择 |
| 12 | ckpt_name | `ltx-2.3_text_projection_bf16` | 投影层 |
| 15 | frame_rate | 24.0 | 帧率 |
| 16 | unet_name | `ltx-2.3-22b-distilled-...` | 模型选择 |
| 16 | weight_dtype | `default` | 精度 |
| 17 | nag_scale | 12.0 | NAG |
| 17 | nag_alpha | 0.30 | NAG |
| 17 | nag_tau | 2.5 | NAG |
| 18 | lora_name | `ltx-2.3-22b-distilled-lora-...` | LoRA 选择 |
| 18 | strength_model | 1.0 | LoRA 强度 |
| 19 | vae_name | `LTX23_video_vae_bf16` | 模型选择 |
| 20 | img_compression | 35 | 预处理 |
| 21 | width×height | 960×544 | 分辨率 |
| 21 | length | 73 | 帧数 |
| 22 | strength | 0.80 | I2V 强度 |
| 22 | bypass | false | I2V 开关 |
| 23 | cfg | 1.0 | CFG scale |
| 24 | steps | 25 | 采样步数 |
| 24 | max_shift | 2.05 | 调度 |
| 24 | base_shift | 0.95 | 调度 |
| 24 | stretch | true | 拉伸分布 |
| 24 | terminal | 0.1 | 终端噪声 |
| 25 | sampler_name | `euler` | 采样器 |
| 26 | noise_seed | 888/randomize | 随机种子 |
| 28 | timestep | 0.05 | DecoderNoise |
| 28 | scale | 0.025 | DecoderNoise |
| 28 | seed | 42/randomize | DecoderNoise |
| 29 | horizontal_tiles | 2 | 分块解码 |
| 29 | vertical_tiles | 2 | 分块解码 |
| 29 | overlap | 6 | 分块重叠 |
| 29 | last_frame_fix | true | 末帧修复 |
| 30 | fps | 24.0 | 输出帧率 |
| 32 | filename_prefix | `badminton_flux_ltx` | 文件名 |
| 32 | format/codec | `mp4`/`h264` | 视频格式 |

---

## 12. 调试指南

### 常见问题诊断流程

```
                    最终视频有问题
                          │
           ┌──────────────┼──────────────┐
           ↓              ↓              ↓
    画面质量差      运动不自然      视频开头不对
           │              │              │
    检查 flux_frame   检查 LTX       检查 I2V
    _*.png (节点11)   Prompt (节点13)  strength (节点22)
           │              │              │
    ┌─────┴─────┐   调整 nag_scale    ↑ 如果突变
    ↓           ↓   调整 nag_tau       ↓ 如果僵硬
  FLUX图片    FLUX图片
  也有问题    正常
    │           │
 调整FLUX    问题在
 Prompt/      LTX阶段
 guidance
```

### 问题 → 解决方案速查

| 症状 | 可能原因 | 解决 |
|------|---------|------|
| OOM (显存不足) | UNet FP16 过大 | FLUX: 保持 fp8; LTX: 降低帧数到 49 |
| 图片模糊 | max_shift 太低 | 提高 FLUX max_shift 到 1.6-1.8 |
| 图片有噪点 | max_shift 太高/guidance 太高 | 降低 max_shift 到 1.0; 降低 guidance 到 3.0 |
| 图片不符合描述 | guidance 太低/Prompt 问题 | 提高 guidance 到 5-6; 重写 Prompt |
| 视频不动(静止) | 负向不够/NAG 太低 | 加强负向"static image,still frame"; 提高 nag_scale |
| 视频闪烁 | nag_tau 太低 | 提高 nag_tau 到 3.5 |
| 物体变形/蠕动 | 负向不够 | 加强"morphing,warping,teleporting" |
| 画面过锐/伪影 | nag_alpha 太高 | 降低 nag_alpha 到 0.2 |
| 视频开头与原图不符 | I2V strength 太低 | 提高 strength 到 0.90 |
| 视频前几帧定格 | I2V strength 太高 | 降低 strength 到 0.70 |
| 色带/banding | DecoderNoise 不够 | 提高 scale 到 0.04 |
| 可见噪点/雪花 | DecoderNoise 太多 | 降低 scale 到 0.01 |
| 拼接接缝 | overlap 太小 | 提高 overlap 到 8-10, 或减少 tiles |
| 最后一帧异常 | last_frame_fix 关闭 | 设置 last_frame_fix=true |
| 生成太慢 | steps 过多 | FLUX: 降 steps 到 20; LTX: 降 steps 到 20 |

### 模型文件清单

| 文件 | 节点 | 大小(约) |
|------|------|---------|
| `flux1-dev.safetensors` | 1 | ~12 GB |
| `clip_l.safetensors` | 3 | ~246 MB |
| `t5xxl_fp16.safetensors` | 3 | ~9.4 GB |
| `ae.safetensors` | 4 | ~320 MB |
| `gemma_3_12B_it_fp8_scaled.safetensors` | 12 | ~12.3 GB |
| `ltx-2.3_text_projection_bf16.safetensors` | 12 | ~2.2 GB |
| `ltx-2.3-22b-distilled-1.1_transformer_only_mxf8_block32.safetensors` | 16 | ~23 GB |
| `ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors` | 18 | ~500 MB |
| `LTX23_video_vae_bf16.safetensors` | 19 | ~1.4 GB |

**总计约 61 GB 模型文件。**

### 最佳实践

1. **固定 seed 调试**: 先用固定 seed（FLUX=777, LTX=888）反复测试参数，找到最佳组合后再 randomize
2. **先调 Prompt，再调参数**: 80% 的问题可以通过优化 Prompt 解决
3. **逐阶段验证**: 先跑 FLUX 部分，确认 `flux_frame_*.png` 满意后，再跑完整流程
4. **NAG 优先于 steps**: 视频运动问题先调 NAG，而不是盲目加步数
5. **保持双端分辨率一致**: FLUX 和 LTX 的分辨率必须相同（统一 960×544）
6. **fps 必须一致**: 节点 15 的 frame_rate 和节点 30 的 fps 必须匹配

---

> **相关文档**: [LTX Workflow Fixes](ltx-workflow-fixes-2025-06-24.md) | [FLUX.1-DEV Migration](flux1-dev-migration.md) | [Workflow Files Index](comfyui-workflow-files-index.md)
