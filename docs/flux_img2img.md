# FLUX 图生图 (`flux_img2img.json`)

> 参考图 + 文字 → 新图。14 节点，denoise=0.65 保留原图结构。

## 效果预览

![FLUX 图生图](../assets/flux_img2img_demo.png)

## 怎么用

1. 拖入 `flux_img2img.json`
2. 节点 12（LoadImage）：上传参考图
3. 节点 5（CLIPTextEncodeFlux）：写 Prompt 描述想要的效果
4. 点击 Queue Prompt

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| denoise | 0.65 | 0=原图不变, 1=完全重画。0.55-0.75 最佳 |
| steps | 25 | 有原图打底，不需要 28 步 |
| T5 编码器 | FP8 | 比文生图的 FP16 省 4.6GB |

## 模型依赖

- 同 FLUX 文生图，但 T5 用 FP8 版（`t5xxl_fp8_e4m3fn.safetensors`）
