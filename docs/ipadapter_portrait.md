# IP-Adapter 人脸参考 (`ipadapter_portrait.json`)

> 给 AI 一张人脸照片 + 一段风格描述，生成**同一个人的不同造型**。10 节点。

## 效果预览

![IP-Adapter](../assets/ipadapter_portrait_demo.png)

## 核心原理

```
参考脸照片 → LoadImage ──────┐
                              ├─→ IPAdapterAdvanced → KSampler → 输出
Animagine XL → UnifiedLoader ─┘        ↑
                                Prompt (只写服装/场景)
```

## 怎么用

1. 拖入 `ipadapter_portrait.json`
2. 节点 2：选一张**正脸清晰**的参考照片
3. 节点 5：写服装/场景/风格，**不要写脸**
4. 点击 Queue Prompt

## Prompt 规则

```
❌ 不要写：头发颜色、眼型、脸型 — 由参考图决定
✅ 只写：  服装、表情、姿势、场景、画风、光线
```

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| weight (节点4) | 0.75 | 人脸相似度。0.5=微调, 0.85=很像, 1.2=可能变形 |
| weight_type | ease in-out | 渐进式注入，比 standard 更平滑 |
| combine_embeds | concat | 🔑 消除波浪纹的关键 |
| embeds_scaling | K+V | 平滑注意力影响 |
| preset (节点3) | PLUS | SDXL 配套，不动 |

## 常见问题

| 问题 | 解决 |
|------|------|
| 波浪纹/网格 | combine_embeds 设为 `concat` |
| 脸不够像 | weight 从 0.75 调到 0.9-1.0 |
| 脸太僵硬 | weight 从 0.75 降到 0.5-0.6 |
| 服装不跟 Prompt 走 | weight_type 改为 `prompt is more important` |

## 模型依赖（需额外安装）

| 模型 | 大小 | 位置 |
|------|------|------|
| IP-Adapter SDXL Plus | 809 MB | `models/ipadapter/` |
| CLIP Vision ViT-H | 2.4 GB | `models/clip_vision/` |

**插件**: `ComfyUI_IPAdapter_plus`（通过 ComfyUI Manager 或 git clone 安装）
