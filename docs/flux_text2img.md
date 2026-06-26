# FLUX 文生图 (`flux_text2img.json`)

> 一段文字 → 一张写实照片。11 节点，FLUX.1-DEV 模型。

## 效果预览

![FLUX 文生图](../assets/flux_text2img_demo.png)

## 怎么用

1. 拖入 `flux_text2img.json`
2. 节点 5（CLIPTextEncodeFlux）：用**自然语言**写 Prompt
3. 节点 6：负向提示词
4. 节点 8：分辨率（默认 1344×768）
5. 点击 Queue Prompt

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| steps | 28 | 性价比最高 |
| guidance | 4.0 | FLUX 推荐值 |
| CFG | 1.0 | FLUX 不需要调 |
| width/height | 1344/768 | 别超过 1600 |

## 模型依赖

- `flux1-dev.safetensors`（23 GB）→ `models/diffusion_models/`
- `clip_l.safetensors`（235 MB）→ `models/text_encoders/`
- `t5xxl_fp16.safetensors`（9.2 GB）→ `models/text_encoders/`
- `ae.safetensors`（320 MB）→ `models/vae/`
