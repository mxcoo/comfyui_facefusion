# ComfyUI-FaceFusion

ComfyUI custom nodes for face swapping using FaceFusion.

## Nodes

| Node | Input | Output |
|------|-------|--------|
| FaceFusion Face Swap | source_image (IMAGE), target_image (IMAGE) | IMAGE |
| FaceFusion Video Face Swap | source_images (IMAGE), target_video (VIDEO) | VIDEO |

## Parameters

| Parameter | Options | Default |
|-----------|---------|---------|
| face_swapper_model | hyperswap_1c_256, inswapper_128, ghost_*, simswap_* | hyperswap_1c_256 |
| face_swapper_pixel_boost | 256x256, 512x512, 768x768, 1024x1024 | 256x256 |
| face_swapper_weight | 0.0 - 1.0 | 0.5 |
| face_enhancer_model | gfpgan_1.4, codeformer, gpen_bfr_* | gfpgan_1.4 |
| face_enhancer_blend | 0 - 100 | 80 |
| face_enhancer_weight | 0.0 - 1.0 | 0.5 |
| face_detector_model | yolo_face, retinaface, scrfd, yunet, many | yolo_face |
| face_detector_size | 640x640, 320x320, 480x480, 512x512, 160x160 | 640x640 |
| face_detector_angles | comma-separated: 0,90,180,270 | 0,90,180,270 |
| face_detector_score | 0.0 - 1.0 | 0.5 |
| face_landmarker_model | 2dfan4, peppapig | 2dfan4 |
| face_landmarker_score | 0.0 - 1.0 | 0.5 |
| face_selector_order | large-small, small-large, left-right, ... | large-small |
| face_selector_gender | none, female, male | none |
| face_occluder_model | xseg_1, xseg_2, xseg_3, many | xseg_1 |
| face_parser_model | bisenet_resnet_34, bisenet_resnet_18 | bisenet_resnet_34 |
| face_mask_types | box, occlusion, area, region | box,occlusion,area,region |
| face_mask_areas | upper-face, lower-face, mouth | upper-face,lower-face,mouth |
| face_mask_regions | skin, left-eyebrow, right-eye, ... | all regions |
| face_mask_blur | 0.0 - 1.0 | 0.3 |
| execution_providers | cuda, cpu, cuda,tensorrt | cuda |

## Workflow

### Image Face Swap
```
LoadImage --> FaceFusion Face Swap --> SaveImage
```

### Video Face Swap
```
LoadImage ----+
              +--> FaceFusion Video Face Swap --> SaveVideo
LoadVideo ----+
```

## Requirements
- ComfyUI v0.22+
- Python 3.12+
- onnxruntime-gpu (CUDA)
- NVIDIA GPU

## Installation
Copy to ComfyUI/custom_nodes/ComfyUI-FaceFusion/ and restart ComfyUI.

## Models
Model files in .assets/models/. Downloaded automatically on first use, or copy from existing FaceFusion installation.

## Credits
- [FaceFusion](https://github.com/facefusion/facefusion) - Core face swap engine
- Built with [Codex](https://codex.ai) - AI-powered coding agent
