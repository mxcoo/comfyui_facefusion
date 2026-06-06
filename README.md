# ComfyUI-FaceFusion

FaceFusion integration nodes for ComfyUI — GPU accelerated face swapping and face enhancement.

## Features

- **FaceFusion Face Swap** — image face swap (source + target → result)
- **FaceFusion Video Face Swap** — video face swap (source + target video → result video)
- Full parameter control: swapper, enhancer, detector, landmarker, masker, selector
- Works with ComfyUI native `LoadImage`, `Load Video`, `SaveImage`, `SaveVideo` nodes
- CUDA accelerated via onnxruntime-gpu

## Installation

### 1. Install the plugin

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/mxcoo/comfyui_facefusion.git
```

Or download the ZIP and extract to `ComfyUI/custom_nodes/`.

### 2. Install onnxruntime-gpu (required)

**Critical: you must install onnxruntime-gpu and remove the CPU-only version, or CUDA will not work.**

```bash
cd ComfyUI/python312/    # or python/ depending on your setup

# Remove CPU-only version if installed
python.exe -m pip uninstall onnxruntime -y

# Install GPU version
python.exe -m pip install "onnxruntime-gpu[cuda]==1.26.0"
```

### 3. Verify CUDA

```bash
python.exe -c "import onnxruntime; print(onnxruntime.get_available_providers())"
```

**Expected:** `['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`

If you see only `['AzureExecutionProvider', 'CPUExecutionProvider']`, the CPU-only
`onnxruntime` is overriding the GPU version — go back to step 2.

### 4. Restart ComfyUI

## Nodes

### FaceFusion Face Swap (image)

Workflow: `LoadImage` → `FaceFusion Face Swap` → `SaveImage`

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| source_image | IMAGE | — | Source face image |
| target_image | IMAGE | — | Target face to swap |
| face_swapper_model | list | `hyperswap_1c_256` | Swapper model |
| face_swapper_pixel_boost | list | `256x256` | Pixel boost resolution |
| face_swapper_weight | float (0–1) | `0.5` | Swap influence weight |
| face_enhancer_model | list | `gfpgan_1.4` | Enhancer model |
| face_enhancer_blend | int (0–100) | `80` | Enhancement blend |
| face_enhancer_weight | float (0–1) | `0.5` | Enhancement weight |
| face_detector_model | list | `yolo_face` | Detection model |
| face_detector_size | list | `640x640` | Detection input size |
| face_detector_angles | list | `0,90,180,270` | Detection angles |
| face_detector_score | float | `0.5` | Detection threshold |
| face_selector_order | list | `large-small` | Face selection order |
| face_selector_gender | list | `none` | Gender filter |
| face_occluder_model | list | `xseg_1` | Occlusion model |
| face_parser_model | list | `bisenet_resnet_34` | Segmentation model |
| face_mask_types | list | `box,occlusion,area,region` | Mask types |
| face_mask_blur | float | `0.3` | Mask blur amount |
| execution_providers | list | `cuda` | ONNX execution provider |

### FaceFusion Video Face Swap

Workflow: `LoadImage` + `Load Video` → `FaceFusion Video Face Swap` → `SaveVideo`

Same parameters as the image node. Uses ComfyUI's native `SaveVideo` — no custom video writer needed.

## Models

Models are downloaded automatically on first use. The plugin uses facefusion's model download system. If you have existing facefusion models, you can copy them to the plugin's models directory.

**Model path detection order:**
1. Plugin's `.assets/models/`
2. FaceFusion standard model location
3. Downloaded from GitHub/HuggingFace on demand

## CUDA Troubleshooting

### Symptom: CUDAExecutionProvider not found in logs

**Root cause:** The CPU-only `onnxruntime` package was installed alongside
`onnxruntime-gpu`, and its DLLs take precedence.

**Fix:**
```bash
# 1. Check installed packages
python.exe -m pip list | findstr onnxruntime

# 2. If both onnxruntime and onnxruntime-gpu appear, remove CPU version
python.exe -m pip uninstall onnxruntime -y

# 3. Reinstall GPU version
python.exe -m pip install "onnxruntime-gpu[cuda]==1.26.0"

# 4. Restart ComfyUI and verify
python.exe -c "import onnxruntime; print(onnxruntime.get_available_providers())"
```

### After ComfyUI update

ComfyUI updates or ComfyUI Manager dependency checks may reintroduce the CPU-only
`onnxruntime`. Repeat the fix above if needed.

## Architecture

```
ComfyUI-FaceFusion/
├── __init__.py              # Node registration + CUDA DLL init
├── facefusion_wrapper.py    # Image swap node + facefusion logic
├── facefusion_video.py      # Video swap node
├── requirements.txt
├── .gitignore
├── README.md
├── README-zh.md
└── facefusion/              # Vendored facefusion engine
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

Built with [Codex](https://codex.ai), an AI coding agent based on GPT-5.

## Credits

- [FaceFusion](https://github.com/facefusion/facefusion) — FaceFusion engine
- Built with [Codex](https://codex.ai)
