# SDXL 动漫头像 (`anime_portrait.json`)

> 用 Danbooru 标签描述角色，AI 生成纯正动漫风格头像。8 节点，仅需 8-12GB 显存。

## 效果预览

![动漫头像](../assets/anime_portrait_demo.png)

## 怎么用

1. 拖入 `anime_portrait.json`
2. 节点 2：用 **Danbooru 标签** 写角色描述
3. 节点 3：负向提示词（排除写实/3D/低质量）
4. 节点 4：调整分辨率（默认 896×1152 竖版头像）
5. 点击 Queue Prompt

## Prompt 格式

Danbooru 标签，逗号分隔：

```
✅ "1girl, solo, silver hair, long hair, purple eyes, portrait, white dress, soft smile, anime coloring, masterpiece, best quality"

❌ "请画一个银发紫瞳的动漫少女"  ← SDXL 动漫模型不理解自然语言
```

## 标签速查

| 类别 | 常用标签 |
|------|---------|
| 角色 | `1girl`, `1boy`, `solo`, `looking at viewer` |
| 发型 | `silver hair`, `long hair`, `short hair`, `bangs`, `ahoge` |
| 眼睛 | `purple eyes`, `blue eyes`, `red eyes`, `detailed eyes` |
| 表情 | `smile`, `soft smile`, `cool expression` |
| 服装 | `white dress`, `school uniform`, `hoodie`, `lace trim` |
| 画质 | `masterpiece`, `best quality`, `absurdres`, `highres` |
| 风格 | `anime coloring`, `cel shading`, `line art`, `depth of field` |
| 背景 | `pastel background`, `gradient sky`, `bokeh`, `simple background` |

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| width/height | 896/1152 | 竖版头像。可改为 1024×1024 方形 |
| steps | 25 | 20-28 最佳 |
| CFG | 7 | SDXL 动漫推荐 5-8 |
| sampler | euler | 动漫首选 |
| scheduler | normal | normal 或 karras |

## 与 FLUX 对比

| | FLUX 文生图 | SDXL 动漫 |
|------|-------------|-----------|
| 模型 | 23 GB | **6.5 GB** |
| 显存 | 16-24 GB | **8-12 GB** |
| 风格 | 写实/电影 | **纯正动漫** |
| 速度 | 慢 | **快 ~3 倍** |

## 模型依赖

- `animagine-xl-4.0.safetensors`（6.5 GB）→ `models/checkpoints/`
