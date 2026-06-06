# ComfyUI-FaceFusion

基于 FaceFusion 引擎的 ComfyUI 换脸插件，GPU 加速。

## 功能

- **FaceFusion Face Swap** — 图片换脸节点（源图 + 目标图 → 结果）
- **FaceFusion Video Face Swap** — 视频换脸节点（源图 + 目标视频 → 结果视频）
- 完整参数控制：换脸模型、增强器、检测器、关键点、遮罩、选择器
- 使用 ComfyUI 原生 `LoadImage`、`Load Video`、`SaveImage`、`SaveVideo` 节点
- CUDA 加速（通过 onnxruntime-gpu）

## 安装

### 1. 下载插件

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/mxcoo/comfyui_facefusion.git
```

或下载 ZIP 解压到 `ComfyUI/custom_nodes/`。

### 2. 安装 onnxruntime-gpu（必须）

**关键：必须安装 onnxruntime-gpu 并删除 CPU 版，否则 CUDA 无效。**

```bash
cd ComfyUI/python312/    # 根据你的环境选择 python\ 或 python312\

# 卸载 CPU 版（如果已安装）
python.exe -m pip uninstall onnxruntime -y

# 安装 GPU 版
python.exe -m pip install "onnxruntime-gpu[cuda]==1.26.0"
```

### 3. 验证 CUDA

```bash
python.exe -c "import onnxruntime; print(onnxruntime.get_available_providers())"
```

**预期输出：** `['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`

如果只有 `['AzureExecutionProvider', 'CPUExecutionProvider']`，说明 CPU 版覆盖了 GPU — 回到步骤 2。

### 4. 重启 ComfyUI

## 节点说明

### FaceFusion Face Swap（图片换脸）

工作流：`LoadImage` → `FaceFusion Face Swap` → `SaveImage`

| 输入 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| source_image | IMAGE | — | 源人脸图片 |
| target_image | IMAGE | — | 目标换脸图片 |
| face_swapper_model | list | `hyperswap_1c_256` | 换脸模型 |
| face_swapper_pixel_boost | list | `256x256` | 像素提升级别 |
| face_swapper_weight | float (0–1) | `0.5` | 换脸权重 |
| face_enhancer_model | list | `gfpgan_1.4` | 增强模型 |
| face_enhancer_blend | int (0–100) | `80` | 增强混合比 |
| face_enhancer_weight | float (0–1) | `0.5` | 增强权重 |
| face_detector_model | list | `yolo_face` | 检测模型 |
| face_detector_size | list | `640x640` | 检测输入尺寸 |
| face_detector_angles | list | `0,90,180,270` | 检测角度 |
| face_detector_score | float | `0.5` | 检测阈值 |
| face_selector_order | list | `large-small` | 人脸选择顺序 |
| face_selector_gender | list | `none` | 性别过滤 |
| face_occluder_model | list | `xseg_1` | 遮挡模型 |
| face_parser_model | list | `bisenet_resnet_34` | 分割模型 |
| face_mask_types | list | `box,occlusion,area,region` | 遮罩类型 |
| face_mask_blur | float | `0.3` | 遮罩模糊 |
| execution_providers | list | `cuda` | ONNX 执行提供者 |

### FaceFusion Video Face Swap（视频换脸）

工作流：`LoadImage` + `Load Video` → `FaceFusion Video Face Swap` → `SaveVideo`

参数与图片节点相同。使用 ComfyUI 原生的 `SaveVideo` 节点保存，无需额外视频保存节点。

## 模型

模型在首次使用时自动下载。该插件使用 facefusion 的模型下载系统。如果你已有 facefusion 模型，可以复制到插件的模型目录。

**模型路径检测顺序：**
1. 插件目录 `.assets/models/`
2. FaceFusion 标准模型位置
3. 从 GitHub/HuggingFace 按需下载

## CUDA 故障排查

### 症状：日志中有 CPUExecutionProvider 但无 CUDAExecutionProvider

**原因：** CPU 版 `onnxruntime` 和 `onnxruntime-gpu` 被同时安装，CPU 版 DLL 优先加载。

**修复：**
```bash
# 1. 检查已安装的包
python.exe -m pip list | findstr onnxruntime

# 2. 如果同时看到 onnxruntime 和 onnxruntime-gpu，卸载 CPU 版
python.exe -m pip uninstall onnxruntime -y

# 3. 重新安装 GPU 版
python.exe -m pip install "onnxruntime-gpu[cuda]==1.26.0"

# 4. 重启 ComfyUI 并验证
python.exe -c "import onnxruntime; print(onnxruntime.get_available_providers())"
```

### ComfyUI 更新后又变回 CPU

ComfyUI 更新或 ComfyUI Manager 的依赖检查可能重新引入 CPU 版 `onnxruntime`。重复上述步骤即可。

## 架构

```
ComfyUI-FaceFusion/
├── __init__.py              # 节点注册 + CUDA DLL 初始化
├── facefusion_wrapper.py    # 图片换脸节点 + facefusion 核心逻辑
├── facefusion_video.py      # 视频换脸节点
├── requirements.txt
├── .gitignore
├── README.md
├── README-zh.md
└── facefusion/              # FaceFusion 引擎完整代码
    ├── face_detector.py
    ├── face_landmarker.py
    ├── face_masker.py
    ├── face_analyser.py
    ├── inference_manager.py
    ├── execution.py
    ├── state_manager.py
    ├── download.py
    └── processors/
        ├── face_swapper/
        └── face_enhancer/
```

使用 [Codex](https://codex.ai) 完成（基于 GPT-5 的 AI 编程助手）。

## 鸣谢

- [FaceFusion](https://github.com/facefusion/facefusion) — FaceFusion 引擎
- 由 [Codex](https://codex.ai) 完成
