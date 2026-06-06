# ComfyUI-FaceFusion

FaceFusion integration nodes for ComfyUI — GPU-accelerated face swapping and face enhancement.

## Features

- **FaceFusion Face Swap** — image face swap node (source + target → result)
- **FaceFusion Video Face Swap** — video face swap node (source + target video → result video)
- Full parameter control: swapper, enhancer, detector, landmarker, masker, selector
- Uses ComfyUI native `LoadImage`, `Load Video`, `SaveImage`, and `SaveVideo` nodes

## Installation

### 1. Clone the plugin

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/mxcoo/comfyui_facefusion.git
```

Or download the ZIP and extract to `ComfyUI/custom_nodes/`.

### 2. Install onnxruntime-gpu (required)

**⚠️ Critical: you must install onnxruntime-gpu and remove the CPU-only version, or CUDA will not work.**

```bash
# Navigate to ComfyUI Python environment
cd ComfyUI/python312/    # or python/ depending on your setup

# Remove CPU-only version if installed
python.exe -m pip uninstall onnxruntime -y

# Install GPU version
python.exe -m pip install onnxruntime-gpu==1.26.0
```

### 3. Verify CUDA is available

```bash
python.exe -c "import onnxruntime; print(onnxruntime.get_available_providers())"
```

**Expected output:** `['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`

If you see only `['AzureExecutionProvider', 'CPUExecutionProvider']`, the CPU-only `onnxruntime` is overriding the GPU files — go back to step 2.

### 4. Restart ComfyUI

## Nodes

### FaceFusion Face Swap

| Input | Type | Default |
|-------|------|---------|
| source_image | IMAGE | — |
| target_image | IMAGE | — |
| face_swapper_model | list | `hyperswap_1c_256` |
| face_swapper_pixel_boost | list | `256x256` |
| face_swapper_weight | float (0–1) | `0.5` |
| face_enhancer_model | list | `gfpgan_1.4` |
| face_enhancer_blend | int (0–100) | `80` |
| face_enhancer_weight | float (0–1) | `0.5` |
| face_detector_model | list | `yolo_face` |
| face_detector_size | list | `640x640` |
| face_detector_angles | list | `0,90,180,270` |
| face_detector_score | float | `0.5` |
| face_selector_order | list | `large-small` |
| face_selector_gender | list | `none` |
| face_occluder_model | list | `xseg_1` |
| face_parser_model | list | `bisenet_resnet_34` |
| face_mask_types | list | `box,occlusion,area,region` |
| face_mask_blur | float | `0.3` |
| execution_providers | list | `cuda` |

Workflow: `LoadImage` → `FaceFusion Face Swap` → `SaveImage`

### FaceFusion Video Face Swap

| Input | Type | Default |
|-------|------|---------|
| source_images | IMAGE | — |
| target_video | VIDEO | — |
| ... | ... | Same params as image node |

Workflow: `LoadImage` + `Load Video` → `FaceFusion Video Face Swap` → `SaveVideo`

Uses ComfyUI's native `SaveVideo` node — no custom video writer needed.

## CUDA Troubleshooting

### Symptom: CPUExecutionProvider but no CUDAExecutionProvider

**Root cause:** The CPU-only `onnxruntime` package was installed alongside `onnxruntime-gpu`, and its files take precedence, hiding the CUDA provider DLLs.

**Fix:**

```bash
# 1. Check installed packages
python.exe -m pip list | findstr onnxruntime

# 2. If both onnxruntime and onnxruntime-gpu appear, remove the CPU version
python.exe -m pip uninstall onnxruntime -y

# 3. Reinstall GPU version
python.exe -m pip install onnxruntime-gpu==1.26.0

# 4. Verify
python.exe -c "import onnxruntime; onnxruntime.get_available_providers()"
# Should show: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

### After ComfyUI update, back to CPU

ComfyUI updates may reintroduce the CPU-only `onnxruntime`. Repeat the fix above.

### Automatic CUDA path setup

The plugin `__init__.py` automatically:
1. Adds `torch/lib/` CUDA DLL paths to system `PATH`
2. Registers DLL search directories via `os.add_dll_directory()` (Windows)
3. Loads DLL directories from `nvidia-cuda-runtime-cu12` and related packages
4. Logs CUDA status at startup for easy troubleshooting

## Architecture

This plugin was built with [Codex](https://codex.ai), an AI coding agent based on GPT-5.

```
ComfyUI-FaceFusion/
├── __init__.py              # Node registration + CUDA DLL path init
├── facefusion_wrapper.py    # Image swap node + facefusion core logic
├── facefusion_video.py      # Video swap node
├── requirements.txt
├── README.md
└── facefusion/              # Full FaceFusion engine
    ├── face_detector.py     # Face detection (YOLO / RetinaFace / SCRFD)
    ├── face_landmarker.py   # Landmark detection
    ├── face_masker.py       # Mask generation
    ├── face_analyser.py     # Face analysis
    ├── inference_manager.py # ONNX Runtime inference
    ├── execution.py         # Execution provider management
    ├── state_manager.py     # Global state management
    ├── download.py          # Model download (curl fallback applied)
    └── processors/          # Processor modules
        ├── face_swapper/    # Face swapper
        └── face_enhancer/   # Face enhancer
```

## Credits

- [FaceFusion](https://github.com/facefusion/facefusion) — FaceFusion engine
- Built with [Codex](https://codex.ai)
