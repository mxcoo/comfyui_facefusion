# ComfyUI-FaceFusion

FaceFusion integration nodes for ComfyUI. Provides GPU-accelerated face swapping and face enhancement via FaceFusion engine.

基于 FaceFusion 引擎的 ComfyUI 换脸插件，支持 GPU 加速。

## Features / 功能

- **FaceFusion Face Swap** — 图片换脸节点（原图 + 目标图 → 换脸结果）
- **FaceFusion Video Face Swap** — 视频换脸节点（原图 + 目标视频 → 换脸视频）
- Supports swapper, enhancer, detector, landmarker, masker, selector full parameter control
- 支持换脸模型、增强模型、检测、遮罩、选择器等完整参数

## Installation / 安装

### 1. 下载插件

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/mxcoo/comfyui_facefusion.git
```

Or download the ZIP and extract to `ComfyUI/custom_nodes/ComfyUI-FaceFusion/`.

### 2. 安装 onnxruntime-gpu (必须)

**⚠️ 关键步骤：必须安装 onnxruntime-gpu 并卸载 CPU 版，否则 CUDA 不可用**

```bash
# 进入 ComfyUI 的 Python 环境
cd ComfyUI/python312/

# 卸载 CPU 版（如果已安装）
python.exe -m pip uninstall onnxruntime -y

# 安装 GPU 版（带 CUDA 依赖）
python.exe -m pip install "onnxruntime-gpu[cuda]==1.26.0"
```

### 3. 验证 CUDA 是否可用

```bash
python.exe -c "import onnxruntime; print(onnxruntime.get_available_providers())"
```

**预期输出：** `['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`

如果只看到 `['AzureExecutionProvider', 'CPUExecutionProvider']`，说明 CPU 版 `onnxruntime` 覆盖了 GPU 文件 —— 回到步骤 2 重新安装。

### 4. 重启 ComfyUI

## Nodes / 节点

### FaceFusion Face Swap

| Input | Type | Description |
|-------|------|-------------|
| source_image | IMAGE | 人脸来源图片 |
| target_image | IMAGE | 目标换脸图片 |
| face_swapper_model | list | 换脸模型 |
| face_swapper_pixel_boost | list | 像素提升级别 |
| face_swapper_weight | float | 换脸权重 (0-1) |
| face_enhancer_model | list | 面部增强模型 |
| face_enhancer_blend | int | 增强混合比 (0-100) |
| face_enhancer_weight | float | 增强权重 (0-1) |
| ... | ... | 检测器/遮罩/选择器等 |

Workflow: `LoadImage → FaceFusion Face Swap → SaveImage`

### FaceFusion Video Face Swap

| Input | Type | Description |
|-------|------|-------------|
| source_images | IMAGE | 人脸来源图片 |
| target_video | VIDEO | 目标换脸视频 |
| ... | ... | 同图片节点 |

Workflow: `LoadImage + Load Video → FaceFusion Video Face Swap → SaveVideo`

Uses ComfyUI native `SaveVideo` node (no extra save nodes needed).

## CUDA Troubleshooting

### 症状：日志显示 CPUExecutionProvider 但无 CUDAExecutionProvider

**根因：** CPU 版 `onnxruntime` 被安装后覆盖了 `onnxruntime-gpu` 的核心文件。

**修复：**

```bash
# 1. 确认两个包都装了
python.exe -m pip list | findstr onnxruntime

# 如果看到 onnxruntime 和 onnxruntime-gpu 同时存在
# 2. 卸载 CPU 版
python.exe -m pip uninstall onnxruntime -y

# 3. 重装 GPU 版（带 CUDA 依赖）
python.exe -m pip install "onnxruntime-gpu[cuda]==1.26.0"

# 4. 验证
python.exe -c "import onnxruntime; onnxruntime.get_available_providers()"
# 应该显示: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

### ComfyUI 更新后又变回 CPU

ComfyUI 更新可能重新引入了 CPU-only 的 `onnxruntime` 依赖。请重复上述步骤。

### 插件自动修复

插件在 `__init__.py` 中会自动：
1. 将 `torch/lib/` 中的 CUDA DLL 路径添加到系统 PATH
2. 使用 `os.add_dll_directory()` 注册 DLL 搜索路径（Windows）
3. 加载 `nvidia-cuda-runtime-cu12` 等包的 DLL 目录
4. 启动时打印 CUDA 状态日志，方便排查

## Architecture / 架构

This plugin was built with [Codex](https://codex.ai), an AI coding agent based on GPT-5.

由 Codex 完成。

```
ComfyUI-FaceFusion/
├── __init__.py              # 节点注册 + CUDA DLL 路径初始化
├── facefusion_wrapper.py    # 图片换脸节点 + facefusion 核心逻辑
├── facefusion_video.py      # 视频换脸节点
├── requirements.txt
├── README.md
└── facefusion/              # FaceFusion 引擎完整代码
    ├── face_detector.py     # 人脸检测（YOLO / RetinaFace / SCRFD）
    ├── face_landmarker.py   # 关键点检测
    ├── face_masker.py       # 遮罩生成
    ├── face_analyser.py     # 人脸分析
    ├── inference_manager.py # ONNX Runtime 推理管理
    ├── execution.py         # 执行提供者管理
    ├── state_manager.py     # 全局状态管理
    ├── download.py          # 模型下载（curl fix applied）
    └── processors/          # 处理器模块
        ├── face_swapper/    # 换脸处理器
        └── face_enhancer/   # 面部增强器
```

## Credits

- [facefusion](https://github.com/facefusion/facefusion) — FaceFusion engine
- Built with [Codex](https://codex.ai)
