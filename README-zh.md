# ComfyUI-FaceFusion

基于 FaceFusion 引擎的 ComfyUI 换脸插件，GPU 加速。

## 功能

- **FaceFusion Face Swap** — 图片换脸节点（原图 + 目标图 → 换脸结果）
- **FaceFusion Video Face Swap** — 视频换脸节点（原图 + 目标视频 → 换脸视频）
- 完整参数控制：换脸模型、增强模型、检测器、关键点、遮罩、选择器
- 使用 ComfyUI 原生 `LoadImage`、`Load Video`、`SaveImage`、`SaveVideo` 节点

## 安装

### 1. 下载插件

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/mxcoo/comfyui_facefusion.git
```

或下载 ZIP 解压到 `ComfyUI/custom_nodes/`。

### 2. 安装 onnxruntime-gpu（必须）

**⚠️ 关键步骤：必须安装 onnxruntime-gpu 并卸载 CPU 版，否则 CUDA 不可用**

```bash
# 进入 ComfyUI 的 Python 环境
cd ComfyUI/python312/    # 或 python/，取决于你的目录结构

# 卸载 CPU 版（如果已安装）
python.exe -m pip uninstall onnxruntime -y

# 安装 GPU 版
python.exe -m pip install onnxruntime-gpu==1.26.0
```

### 3. 验证 CUDA 是否可用

```bash
python.exe -c "import onnxruntime; print(onnxruntime.get_available_providers())"
```

**预期输出：** `['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`

如果只看到 `['AzureExecutionProvider', 'CPUExecutionProvider']`，说明 CPU 版 `onnxruntime` 覆盖了 GPU 文件 —— 回到步骤 2 重新安装。

### 4. 重启 ComfyUI

## 节点

### FaceFusion Face Swap（图片换脸）

| 输入 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| source_image | IMAGE | 人脸来源图片 | — |
| target_image | IMAGE | 目标换脸图片 | — |
| face_swapper_model | list | 换脸模型 | `hyperswap_1c_256` |
| face_swapper_pixel_boost | list | 像素提升级别 | `256x256` |
| face_swapper_weight | float | 换脸权重 (0–1) | `0.5` |
| face_enhancer_model | list | 面部增强模型 | `gfpgan_1.4` |
| face_enhancer_blend | int | 增强混合比 (0–100) | `80` |
| face_enhancer_weight | float | 增强权重 (0–1) | `0.5` |
| face_detector_model | list | 检测器模型 | `yolo_face` |
| face_detector_size | list | 检测器输入尺寸 | `640x640` |
| face_detector_angles | list | 检测角度 | `0,90,180,270` |
| face_detector_score | float | 检测阈值 | `0.5` |
| face_selector_order | list | 人脸选择顺序 | `large-small` |
| face_selector_gender | list | 性别过滤 | `none` |
| face_occluder_model | list | 遮挡模型 | `xseg_1` |
| face_parser_model | list | 解析模型 | `bisenet_resnet_34` |
| face_mask_types | list | 遮罩类型 | `box,occlusion,area,region` |
| face_mask_blur | float | 遮罩模糊 | `0.3` |
| execution_providers | list | ONNX 执行提供者 | `cuda` |

工作流：`LoadImage` → `FaceFusion Face Swap` → `SaveImage`

### FaceFusion Video Face Swap（视频换脸）

| 输入 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| source_images | IMAGE | 人脸来源图片 | — |
| target_video | VIDEO | 目标换脸视频 | — |
| ... | ... | 其余参数同图片节点 | — |

工作流：`LoadImage` + `Load Video` → `FaceFusion Video Face Swap` → `SaveVideo`

使用 ComfyUI 原生 `SaveVideo` 节点，无需额外的视频保存节点。

## CUDA 故障排查

### 症状：日志有 CPUExecutionProvider 但无 CUDAExecutionProvider

**根因：** CPU 版 `onnxruntime` 和 `onnxruntime-gpu` 被同时安装，CPU 版文件优先加载，导致 CUDA provider 被隐藏。

**修复：**

```bash
# 1. 检查已安装的包
python.exe -m pip list | findstr onnxruntime

# 2. 如果同时看到 onnxruntime 和 onnxruntime-gpu，卸载 CPU 版
python.exe -m pip uninstall onnxruntime -y

# 3. 重装 GPU 版
python.exe -m pip install onnxruntime-gpu==1.26.0

# 4. 验证
python.exe -c "import onnxruntime; onnxruntime.get_available_providers()"
# 应该显示: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

### ComfyUI 更新后又变回 CPU

ComfyUI 更新可能重新引入了 CPU 版 `onnxruntime`。请重复上述步骤。

### 插件自动修复

插件 `__init__.py` 会自动：
1. 将 `torch/lib/` 中的 CUDA DLL 路径添加到系统 `PATH`
2. 使用 `os.add_dll_directory()` 注册 DLL 搜索路径（Windows）
3. 加载 `nvidia-cuda-runtime-cu12` 等包的 DLL 目录
4. 启动时打印 CUDA 状态日志，方便排查

## 架构

此插件由 [Codex](https://codex.ai)（基于 GPT-5 的 AI 编程助手）完成。

```
ComfyUI-FaceFusion/
├── __init__.py              # 节点注册 + CUDA DLL 路径初始化
├── facefusion_wrapper.py    # 图片换脸节点 + facefusion 核心逻辑
├── facefusion_video.py      # 视频换脸节点
├── requirements.txt
├── README.md
├── README-zh.md
└── facefusion/              # FaceFusion 引擎完整代码
    ├── face_detector.py     # 人脸检测（YOLO / RetinaFace / SCRFD）
    ├── face_landmarker.py   # 关键点检测
    ├── face_masker.py       # 遮罩生成
    ├── face_analyser.py     # 人脸分析
    ├── inference_manager.py # ONNX Runtime 推理管理
    ├── execution.py         # 执行提供者管理
    ├── state_manager.py     # 全局状态管理
    ├── download.py          # 模型下载（已修复 curl 依赖）
    └── processors/          # 处理器模块
        ├── face_swapper/    # 换脸处理器
        └── face_enhancer/   # 面部增强器
```

## 致谢

- [FaceFusion](https://github.com/facefusion/facefusion) — FaceFusion 引擎
- 由 [Codex](https://codex.ai) 完成
