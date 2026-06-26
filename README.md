# 🎬 ComfyUI FLUX + LTX Video — AI 图片转视频工作流

> **一句话说清楚**：给 AI 写一段话描述你想要的画面，它先生成一张高清图片，再让这张图片"动起来"变成视频。

这个项目包含 8 个 ComfyUI 工作流，覆盖写实风格（FLUX）、动漫风格（Animagine XL）、人脸参考生成（IP-Adapter），从最简单的"文字→图片"到最复杂的"文字→图片→视频"都有。

---

## 📖 目录

- [能做什么？](#-能做什么)
- [效果预览](#-效果预览)
- [项目文件说明](#-项目文件说明)
- [新手入门（5 分钟上手）](#-新手入门5-分钟上手)
- [工作流详细介绍](#-工作流详细介绍)
- [常见问题排查](#-常见问题排查)
- [模型文件清单](#-模型文件清单)
- [硬件要求](#-硬件要求)

---

## 🎯 能做什么？

| 你想要的效果 | 用这个工作流 | 难度 |
|-------------|-------------|------|
| 输入一段文字，输出一张写实照片 | `flux_text2img` | ⭐ 入门 |
| 输入一段文字，输出一张动漫头像 | `anime_portrait` | ⭐ 入门 |
| 输入一张脸 + 文字，生成同一个人不同造型 | `ipadapter_portrait` | ⭐⭐ 进阶 |
| 输入一张图 + 一段文字，输出变体图 | `flux_img2img` | ⭐ 入门 |
| 输入文字描述，输出 3 秒短视频 | `ltx_txt2vid` | ⭐⭐ 进阶 |
| 输入两段文字，先生成首帧再变成视频（画质最高） | `flux_ltx_i2v_v2` ★ | ⭐⭐⭐ 高级 |

---

## 🖼️ 效果预览

### FLUX 文生图 — "羽毛球运动员跳起扣杀"

![FLUX 文生图示例](assets/flux_text2img_demo.png)

*工作流: `flux_text2img.json` | 分辨率: 1344×768 | 28 步采样*

### FLUX 图生图 — 参考图 + 电影风格重绘

![FLUX 图生图示例](assets/flux_img2img_demo.png)

*工作流: `flux_img2img.json` | 分辨率: 960×544 | denoise=0.65*

### FLUX→LTX 图生视频 — 超跑静止→启动→弹射

![FLUX 生成的首帧画面](assets/flux_ltx_i2v_frame.png)

*工作流: `flux_ltx_i2v_v2.json` | 首帧生成后的画面*

🎥 **生成的视频**：[点击下载观看](assets/hypercar_demo.mp4) — 超跑从静止到弹射起步的 3 秒连贯视频

### SDXL 动漫头像 — 银发少女

![动漫头像示例](assets/anime_portrait_demo.png)

*工作流: `anime_portrait.json` | 模型: Animagine XL 4.0 | 分辨率: 896×1152 | Euler 25步*

---

## 📁 项目文件说明

```
comfy/                                    # 🏠 项目根目录
│
├── 📖 README.md                          # 👈 你正在看的这个文件
├── 📖 flux_ltx_i2v_文档.md              # 组合工作流完整参数文档（进阶阅读）
│
├── 🎨 工作流文件（直接拖进 ComfyUI 就能用）
│   ├── anime_portrait.json              # SDXL 动漫头像 — "标签 → 动漫角色"（省显存）
│   ├── ipadapter_portrait.json          # IP-Adapter 人脸参考 — "一张脸 → 各种造型"
│   ├── flux_text2img.json               # FLUX 文生图 — "一段话 → 一张图"
│   ├── flux_img2img.json                # FLUX 图生图 — "参考图 + 一段话 → 新图"
│   ├── flux_ltx_i2v.json                # FLUX→LTX 基线版 — "两段话 → 3秒视频"
│   ├── flux_ltx_i2v_optimized.json      # FLUX→LTX 优化版 — 加入 STG 增强（实验性）
│   ├── flux_ltx_i2v_v2.json             # FLUX→LTX v2 ★ 推荐 — 最稳定的视频生成
│   └── ltx_txt2vid.json                 # LTX 文生视频 — "一段话 → 短视频"（省显存）
│
├── 🔧 工具脚本（一般不需要手动运行）
│   ├── download_models.py               # 下载 FLUX 模型（Python 版）
│   ├── download_models.sh               # 下载 FLUX 模型（Shell 版）
│   ├── download_extras.sh               # 下载 LTX 额外模型
│   ├── build_optimized_workflow.py      # 自动构建优化版工作流
│   └── fix_workflow_links.py            # 修复工作流节点连接
│
├── 🖼️ assets/                           # 示例图片和视频
│   ├── anime_portrait_demo.png          # 动漫头像效果展示
│   ├── ipadapter_demo.png               # IP-Adapter 人脸参考效果
│   ├── flux_text2img_demo.png           # 文生图效果展示
│   ├── flux_img2img_demo.png            # 图生图效果展示
│   ├── flux_ltx_i2v_frame.png           # 图生视频首帧展示
│   └── hypercar_demo.mp4                # 最终视频效果
│
└── ⚙️ 配置文件
    ├── .gitignore                        # Git 忽略规则（不追踪模型文件/密钥）
    └── .mcp.json                         # ComfyUI 自动化接口配置
```

---

## 🚀 新手入门（5 分钟上手）

### 第一步：确认模型已下载

打开终端，检查模型文件是否存在：

```bash
ls ~/comfy/ComfyUI/models/diffusion_models/
# 应该看到: flux1-dev.safetensors, ltx-2.3-22b-...mxfp8_block32.safetensors

ls ~/comfy/ComfyUI/models/text_encoders/
# 应该看到: clip_l.safetensors, t5xxl_fp8_e4m3fn.safetensors, gemma_3_12B_it_fp8_scaled.safetensors
```

> 如果文件不存在，运行 `python3 download_models.py` 下载（约 78GB，需要代理）。

### 第二步：启动 ComfyUI

```bash
cd ~/comfy/ComfyUI

# 标准启动（推荐 48GB 以上显卡）
venv/bin/python main.py --listen 127.0.0.1

# 24GB 显卡优化启动（RTX 3090/4090）
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 \
  venv/bin/python main.py --listen 127.0.0.1
```

看到 `Starting server` 就说明启动成功了。

### 第三步：打开浏览器

在浏览器地址栏输入：**`http://127.0.0.1:8188`**

你会看到 ComfyUI 的图形界面。

### 第四步：加载并运行工作流

1. 把 `.json` 工作流文件**直接拖入**浏览器窗口
2. 节点图会自动加载，检查一下参数是否符合预期
3. 点击右侧面板的 **`Queue Prompt`** 按钮
4. 等待生成完成，图片/视频会自动保存到 `ComfyUI/output/` 目录

> 💡 **第一次用推荐从简单的开始**：先试试 `flux_text2img.json`，确认能正常出图后再试 `flux_ltx_i2v_v2.json`。

---

## 📘 工作流详细介绍

### 1. SDXL 动漫头像 (`anime_portrait.json`) 🆕

> **这是什么**：用 Danbooru 标签（如 `1girl, silver hair, purple eyes`）描述角色，AI 生成纯正动漫风格的头像/立绘。
>
> **什么时候用**：你想要动漫角色图、头像、漫画风格插画，而且不想折腾复杂的 FLUX 参数。

**怎么用：**

1. 拖入 `anime_portrait.json`
2. 节点 2（正向提示词）：用 **Danbooru 标签** 写角色描述
3. 节点 3（负向提示词）：排除写实、3D、低质量
4. 节点 4：调整分辨率（默认 896×1152 竖版）
5. 点击 Queue Prompt

**Prompt 格式 — Danbooru 标签：**

```
✅ 正确写法（标签式，逗号分隔）：
   "1girl, solo, silver hair, long hair, purple eyes, portrait, 
    white dress, soft smile, looking at viewer, anime coloring,
    masterpiece, best quality, absurdres"

❌ 错误写法（自然语言，SDXL 动漫模型不理解）：
   "请画一个银发紫瞳的动漫少女，穿着白色连衣裙，微笑看向镜头"
```

**Prompt 标签速查：**

| 类别 | 常用标签 |
|------|---------|
| 角色 | `1girl`, `1boy`, `solo`, `looking at viewer` |
| 发型 | `silver hair`, `long hair`, `short hair`, `bangs`, `ahoge` |
| 眼睛 | `purple eyes`, `blue eyes`, `red eyes`, `detailed eyes` |
| 表情 | `smile`, `soft smile`, `cool expression`, `gentle expression` |
| 服装 | `white dress`, `school uniform`, `hoodie`, `lace trim` |
| 画质 | `masterpiece`, `best quality`, `absurdres`, `highres` |
| 风格 | `anime coloring`, `cel shading`, `line art`, `depth of field` |
| 背景 | `pastel background`, `gradient sky`, `bokeh`, `simple background` |

**和 FLUX 文生图的对比：**

| | FLUX 文生图 | SDXL 动漫头像 |
|------|-------------|---------------|
| 模型 | FLUX.1-dev (23GB) | Animagine XL 4.0 (6.5GB) |
| 显存 | 16-24 GB | **8-12 GB** |
| 风格 | 写实/电影感 | **纯正动漫/漫画** |
| Prompt | 自然语言 | Danbooru 标签 |
| 出图速度 | 较慢 | **快 ~3 倍** |

**关键参数：**

| 参数 | 默认值 | 是什么意思 |
|------|--------|-----------|
| width/height | 896/1152 | 竖版头像。可改为 1024×1024 方形 |
| steps | 25 | 20-28 最佳。太低粗糙，太高浪费 |
| CFG | 7 | SDXL 动漫推荐 5-8。越低越自由，越高越贴 Prompt |
| sampler | euler | 动漫首选。euler_ancestral 风格更随机 |
| scheduler | normal | normal 或 karras 都可以 |

---

### 2. IP-Adapter 人脸参考 (`ipadapter_portrait.json`) 🆕

> **这是什么**：给 AI 一张人脸照片 + 一段风格描述，它生成**同一个人的不同造型**——换服装、换场景、换表情，但脸保持一致。
>
> **什么时候用**：你想把自己或朋友"动漫化"，或者给同一个角色做不同造型的设定图。

**怎么用：**

1. 拖入 `ipadapter_portrait.json`
2. 节点 2（LoadImage）：选一张**正脸清晰**的参考照片
3. 节点 5（正向 Prompt）：写服装/场景/风格，**不要写脸**（发色眼型由参考图决定）
4. 点击 Queue Prompt

**关键参数：**

| 参数 | 默认值 | 是什么意思 |
|------|--------|-----------|
| weight (节点4) | 0.75 | 人脸相似度。0.5=微调，0.85=很像，1.2=死扣但可能变形 |
| weight_type | ease in-out | 渐进式注入，比 standard 更平滑 |
| combine_embeds | concat | 消除波浪纹的关键参数 |
| embeds_scaling | K+V | 同时在 Key 和 Value 上施加影响 |
| preset (节点3) | PLUS (high strength) | SDXL 动漫模型配套，不动 |

**Prompt 写作核心规则：**

```
❌ 不要写脸：头发颜色、眼型、脸型 — 这些由参考图决定
✅ 只写这些：服装、表情、姿势、场景、画风、光线
```

**模型依赖（额外下载）：**
- IP-Adapter 插件：`ComfyUI_IPAdapter_plus`（需通过 ComfyUI Manager 或 git clone 安装）
- IP-Adapter 权重：`ip-adapter-plus_sdxl_vit-h.safetensors`（809 MB，放 `models/ipadapter/`）
- CLIP Vision：`CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors`（2.4 GB，放 `models/clip_vision/`）

| 对比 | 纯动漫头像 | IP-Adapter |
|------|-----------|------------|
| 人脸来源 | AI 随机生成 | **你指定的参考图** |
| 角色一致性 | 每次不同 | **同一个人** |
| 适合 | 探索各种角色 | 固定角色+不同造型 |

---

### 3. FLUX 文生图 (`flux_text2img.json`) — 写实风格

> **这是什么**：给 AI 一段描述文字，它给你画出来。
>
> **什么时候用**：你想要一张高质量的图，脑子里有画面但不会画。

**怎么用：**

1. 拖入 `flux_text2img.json`
2. 找到节点 5（`CLIPTextEncodeFlux`），在文本框里写上你的 Prompt
3. 节点 6 是负向提示词（告诉 AI **不要**画什么）
4. 节点 8 设置图片尺寸（默认 1344×768）
5. 点击 Queue Prompt

**使用了什么模型：**
- 扩散模型：`flux1-dev.safetensors`（23GB，FP8 量化）
- 文本理解：`clip_l.safetensors`（理解短词）+ `t5xxl_fp16.safetensors`（理解长句）
- 图像编解码：`ae.safetensors`（320MB）

**关键参数解释：**

| 参数 | 默认值 | 是什么意思 |
|------|--------|-----------|
| steps | 28 | 采样步数，越高画质越好但越慢。28 是性价比最高的点 |
| guidance | 4.0 | Prompt 遵循度，越高 AI 越听话。4.0 是 FLUX 的推荐值 |
| CFG | 1.0 | FLUX 不需要 CFG，保持 1.0 |
| width / height | 1344 / 768 | 输出尺寸，越大越吃显存。建议别超过 1600 |

---

### 4. FLUX 图生图 (`flux_img2img.json`)

> **这是什么**：给 AI 一张参考图 + 一段描述，它在原图基础上按你的要求修改。
>
> **什么时候用**：你有一张图想"换风格"或"微调"，而不是从零画。

**怎么用：**

1. 拖入 `flux_img2img.json`
2. 节点 12（`LoadImage`）：点 `choose file` 上传你的参考图
3. 节点 13（`ImageScale`）：会自动把图片缩放到 960×544
4. 节点 5（`CLIPTextEncodeFlux`）：写 Prompt 描述你想要的效果
5. 点击 Queue Prompt

**和文生图的区别：**
- 多了 `LoadImage` → `ImageScale` → `VAEEncode` 这个"图片输入链"
- `denoise` 参数 = 0.65，意思是保留 35% 原图信息、65% AI 重新创作
- 使用 T5 **FP8** 版编码器（比文生图的 FP16 省 4.6GB 显存）

| 参数 | 默认值 | 是什么意思 |
|------|--------|-----------|
| denoise | 0.65 | 重绘程度。0=原图不变，1=完全重画。0.55-0.75 是最佳范围 |
| steps | 25 | 比文生图少 3 步，因为有原图打底不需要那么多次 |

---

### 5. FLUX→LTX 图生视频 v2 ★ (`flux_ltx_i2v_v2.json`)

> **这是什么**：项目最核心的工作流。分两步走——
> 1. **FLUX 先生成一张静态图**（首帧画面）
> 2. **LTX 把这张图变成 3 秒视频**（添加运动）
>
> **什么时候用**：你想要最高质量的 AI 视频。适合产品展示、概念动画、短视频素材。

**怎么用：**

1. 拖入 `flux_ltx_i2v_v2.json`
2. **节点 5**（FLUX 正向 Prompt）：描述**首帧画面**——谁、在哪、什么姿势、什么光线
3. **节点 13**（LTX 正向 Prompt）：描述**接下来发生什么运动**——怎么动、往哪走、多快
4. 节点 8 设置分辨率（默认 960×544）
5. 节点 21 设置帧数（73 帧 ≈ 3 秒 @ 24fps）
6. 点击 Queue Prompt

**Prompt 写作技巧：**

```
❌ 错误写法（两个 Prompt 写一样的内容）：
  FLUX: "一辆超跑在公路上飞驰"  
  LTX:  "一辆超跑在公路上飞驰"     ← 重复了，没用

✅ 正确写法（各司其职）：
  FLUX: "一辆哑光黑超跑斜停在湿漉漉的沿海公路上，金色夕阳逆光，
        低角度三/四前侧拍摄，水面反射琥珀色光芒，无人，静止"
         ↑ 只描述"这一帧"的画面

  LTX:  "超跑停留片刻后 LED 大灯亮起，引擎发出深沉的咆哮，
        排气管开始发光，后轮突然咬住湿沥青，车辆猛烈弹射起步，
        水雾和碎石从后轮拱喷出，尾灯拉成红色光带消失在海滨公路尽头"
         ↑ 只描述"接下来发生的运动"
```

**v2 版为什么最推荐：**

| 对比项 | 基线版 | 优化版 | v2 ★ |
|--------|--------|--------|------|
| 稳定性 | 中等 | 差（容易出伪影） | **最好** |
| 运动幅度 | 适中 | 大（但可能变形） | **适中偏保守** |
| 引导方式 | CFG | STG 1.5 | **STG 1.0** |
| 采样步数 | 25 | 25 | **30**（画质更好） |
| 首帧锚定 | strength 0.80 | 0.72 | **0.85**（更忠实首帧） |

**关键参数解释：**

| 参数 | 默认值 | 是什么意思 |
|------|--------|-----------|
| FLUX steps | 28 | 首帧采样步数，不需要改 |
| LTX steps | 30 | 视频采样步数，越高运动越细腻 |
| I2V strength | 0.85 | 首帧锚定强度。0=完全不管首帧，1=死扣首帧。0.80-0.90 最佳 |
| NAG scale | 10 | 运动引导强度。太低=不动，太高=鬼畜。8-14 是安全范围 |
| NAG tau | 3.5 | 时间平滑度。越高运动越连贯。2.0-4.0 |
| STG stg | 1.0 | 时空引导。1.0=标准，1.5=激进（可能过度锐化） |
| terminal | 0.05 | 去噪终点。越低越干净，但太低会失去细节 |
| frame_rate | 24 | 帧率，和 CreateVideo 里的 fps 保持一致 |
| 帧数 | 73 | 73÷24≈3 秒。减少可加速但视频变短 |

---

### 6. LTX 文生视频 (`ltx_txt2vid.json`)

> **这是什么**：跳过 FLUX，直接从文字生成视频。
>
> **什么时候用**：显存不够（16-20GB 就能跑），或不需要特别高质量的首帧。

**怎么用：**

1. 拖入 `ltx_txt2vid.json`
2. 节点 13：写 Prompt（同时描述画面+运动）
3. 节点 21：设置分辨率和帧数（默认 768×512, 49 帧 ≈ 2 秒）
4. 点击 Queue Prompt

**和 FLUX→LTX 的区别：**
- 少了整个 FLUX 阶段（11 个节点），更简单更快
- 没有参考首帧，画面一致性不如 I2V
- 显存需求低 ~8GB

---

## 🔧 常见问题排查

### Q: 点击 Queue Prompt 后报错 "CUDA Out of Memory"

**原因**：显存不够。

**解决办法（按优先级）：**

1. **降分辨率**：把 960×544 改成 768×432（FLUX token 减少 ~30%，显存降到 18-20GB）
2. **加启动参数**：
   ```bash
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 \
     venv/bin/python main.py --listen 127.0.0.1 --novram
   ```
3. **关闭其他程序**：关掉浏览器（吃 1-2GB 显存）、IDE 等

### Q: 生成的视频画面闪烁/变形/有鬼影

**可能原因和解决：**
- **I2V strength 太低**：调到 0.85 或更高
- **STG 太高**：降到 1.0 或关掉 STGGuider
- **VAE Decoder Noise 不够**：timestep 调到 0.05，scale 调到 0.025
- **Prompt 里有矛盾描述**：比如"静止"+"运动"同时出现，AI 会困惑

### Q: 视频第一帧和最后几帧有奇怪残影

- 确保 `last_frame_fix` = **true**（节点 29 LTXVTiledVAEDecode）
- 减少帧数（73→49）试试

### Q: 画面不够清晰

- 增加 LTX steps（30→35），但会更慢
- NAG alpha 调高到 0.25（增加锐度）
- 检查分辨率可被 32 整除（LTX 的硬性要求）

### Q: 启动 ComfyUI 后网页打不开

```bash
# 检查是否真的在监听
curl -s http://127.0.0.1:8188 | head -5

# 检查进程
ps aux | grep main.py
```

---

## 🧠 模型文件清单

> 这些文件都在 `ComfyUI/models/` 下，**不需要手动管理**。仅作参考。

### FLUX 系列（负责生成图片）

| 文件 | 大小 | 做什么的 |
|------|------|---------|
| `diffusion_models/flux1-dev.safetensors` | 23 GB | 🧠 扩散模型大脑，负责"画" |
| `text_encoders/clip_l.safetensors` | 235 MB | 📖 理解短词和标签 |
| `text_encoders/t5xxl_fp8_e4m3fn.safetensors` | 4.6 GB | 📖 理解长句描述（v2 推荐） |
| `text_encoders/t5xxl_fp16.safetensors` | 9.2 GB | 📖 同上，更精确但更吃显存 |
| `vae/ae.safetensors` | 320 MB | 🖼️ 图像压缩/解压 |

### SDXL 动漫系列（负责生成动漫角色）

| 文件 | 大小 | 做什么的 |
|------|------|---------|
| `checkpoints/animagine-xl-4.0.safetensors` | 6.5 GB | 🎨 SDXL 动漫模型，840万张动漫图训练 |

> 💡 SDXL 模型自带 CLIP 和 VAE，不需要额外下载文本编码器和 VAE。

### LTX 系列（负责生成视频）

| 文件 | 大小 | 做什么的 |
|------|------|---------|
| `diffusion_models/ltx-2.3-22b-...mxfp8_block32.safetensors` | 23 GB | 🧠 视频扩散模型大脑 |
| `text_encoders/gemma_3_12B_it_fp8_scaled.safetensors` | 13 GB | 📖 理解运动描述 |
| `text_encoders/ltx-2.3_text_projection_bf16.safetensors` | 2.2 GB | 🔗 文字到视频的"翻译层" |
| `vae/LTX23_video_vae_bf16.safetensors` | 1.4 GB | 🎞️ 视频压缩/解压 |
| `loras/ltx-2.3-22b-distilled-lora-...bf16.safetensors` | 2.5 GB | 💪 动态增强插件 |

**总计：~84.5 GB**（FLUX ~37GB + SDXL ~6.5GB + LTX ~41GB）

---

## 💻 硬件要求

| 显卡 | 显存 | 动漫头像 | 文生图 | 图生图 | 图生视频 | 纯文生视频 |
|------|------|----------|--------|--------|----------|-----------|
| RTX 3060 (12GB) | 12 GB | ✅ 流畅 | ❌ | ❌ | ❌ | ❌ |
| RTX 4070 (12GB) | 12 GB | ✅ 流畅 | ❌ | ❌ | ❌ | ❌ |
| RTX 3090 (24GB) | 24 GB | ✅ 秒出 | ✅ | ✅ | ⚠️ 需优化 | ✅ |
| RTX 4090 (24GB) | 24 GB | ✅ 秒出 | ✅ | ✅ | ⚠️ 需优化 | ✅ |
| A6000 (48GB) | 48 GB | ✅ | ✅ | ✅ | ✅ | ✅ |
| A100 (80GB) | 80 GB | ✅ | ✅ | ✅ | ✅ | ✅ |

> 🎉 **好消息**：动漫头像工作流只需 8-12GB 显存，连 12GB 入门卡都能流畅运行！

**24GB 显卡跑图生视频的优化方法：**
1. 分辨率降到 768×432
2. 启动时加 `--novram` 参数
3. 减少帧数（73→49）
4. 关掉浏览器等吃显存的程序

---

## 📜 更新日志

| 日期 | 做了什么 |
|------|---------|
| 2026-06-22 | 搭建 ComfyUI 环境，FLUX Schnell 文生图 |
| 2026-06-23 | 迁移到 FLUX.1-DEV，下载全部模型 |
| 2026-06-24 | 创造 FLUX+LTX 组合工作流，Gemma 替代 T5 编码器 |
| 2026-06-25 | 引入 STG 时空引导，发布 v2 稳定版，FP8 优化 |
| 2026-06-26 | 新增动漫头像 (Animagine XL 4.0) + IP-Adapter 人脸参考，清理 40GB 冗余模型，上传 GitHub |

---

> 💡 **有疑问？** 先看 `flux_ltx_i2v_文档.md`（完整参数手册），或者在这个仓库提 Issue。
