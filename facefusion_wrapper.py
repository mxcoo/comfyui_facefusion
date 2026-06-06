import os
import sys
import logging
import threading

import numpy as np
import cv2
import torch

log = logging.getLogger(__name__)

# Global lock to prevent concurrent state corruption when multiple
# FaceFusion nodes run in parallel via ComfyUI's async execution.
_state_lock = threading.Lock()

FF_DIR = os.path.dirname(__file__)
sys.path.insert(0, FF_DIR)

import facefusion
from facefusion import state_manager


def _ensure_cuda_provider():
    """Verify onnxruntime can see CUDAExecutionProvider and log result."""
    import onnxruntime as _ort
    available = _ort.get_available_providers()
    has_cuda = "CUDAExecutionProvider" in available
    log.info("onnxruntime %s available providers: %s | CUDA available: %s",
             _ort.__version__, available, has_cuda)
    if not has_cuda:
        log.warning(
            "CUDAExecutionProvider NOT found in onnxruntime providers.\n"
            "  This means inference will fall back to CPU.\n"
            "  If torch sees CUDA (%s), you may need:\n"
            "    1. `pip uninstall onnxruntime -y` (remove CPU-only version)\n"
            "    2. `pip install onnxruntime-gpu[cuda]==1.26.0`\n"
            "    3. Restart ComfyUI",
            __import__("torch").cuda.is_available())

def tensor_to_vision_frame(tensor):
    """Convert image tensor (B, H, W, 3) or (1, H, W, 3) to OpenCV BGR frame (H, W, 3). Uses first image in batch."""
    if tensor.dim() == 4:
        img = tensor[0]
    else:
        img = tensor
    img = img.cpu().numpy()
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255.0).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def vision_frame_to_tensor(frame):
    """Convert single OpenCV BGR frame (H, W, 3) to image tensor (1, H, W, 3)."""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).unsqueeze(0)

def tensors_to_vision_frames(tensor):
    """Convert batch tensor (B, H, W, 3) to list of OpenCV BGR frames (H, W, 3)."""
    imgs = tensor.cpu().numpy()
    imgs = np.clip(imgs, 0.0, 1.0)
    imgs = (imgs * 255.0).astype(np.uint8)
    results = []
    for img in imgs:
        results.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return results

def vision_frames_to_tensor(frames):
    """Convert list of OpenCV BGR frames (H, W, 3) to batch tensor (B, H, W, 3)."""
    imgs = []
    for frame in frames:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        imgs.append(torch.from_numpy(img))
    return torch.stack(imgs, dim=0)

def init_facefusion_state():
    _ensure_cuda_provider()

    state_manager.init_item("execution_providers", ["cuda"])
    state_manager.init_item("execution_device_ids", [0])
    state_manager.init_item("execution_thread_count", 4)
    state_manager.init_item("execution_queue_count", 1)
    state_manager.init_item("download_providers", ["github", "huggingface"])
    state_manager.init_item("face_detector_model", "yolo_face")
    state_manager.init_item("face_detector_size", "640x640")
    state_manager.init_item("face_detector_angles", [0, 90, 180, 270])
    state_manager.init_item("face_detector_score", 0.5)
    state_manager.init_item("face_detector_margin", (0, 0, 0, 0))
    state_manager.init_item("face_landmarker_model", "2dfan4")
    state_manager.init_item("face_landmarker_score", 0.5)
    state_manager.init_item("face_selector_mode", "one")
    state_manager.init_item("face_selector_order", "large-small")
    state_manager.init_item("face_selector_gender", None)
    state_manager.init_item("face_selector_race", None)
    state_manager.init_item("face_occluder_model", "xseg_1")
    state_manager.init_item("face_parser_model", "bisenet_resnet_34")
    state_manager.init_item("face_mask_types", ["box", "occlusion", "area", "region"])
    state_manager.init_item("face_mask_areas", ["upper-face", "lower-face", "mouth"])
    state_manager.init_item("face_mask_regions", ["skin", "left-eyebrow", "right-eyebrow", "left-eye", "right-eye", "glasses", "nose", "mouth", "upper-lip", "lower-lip"])
    state_manager.init_item("face_mask_blur", 0.3)
    state_manager.init_item("face_mask_padding", (0, 0, 0, 0))
    state_manager.init_item("face_swapper_model", "hyperswap_1c_256")
    state_manager.init_item("face_swapper_pixel_boost", "256x256")
    state_manager.init_item("face_swapper_weight", 0.5)
    state_manager.init_item("face_enhancer_model", "gfpgan_1.4")
    state_manager.init_item("face_enhancer_blend", 80)
    state_manager.init_item("face_enhancer_weight", 0.5)
    state_manager.init_item("video_memory_strategy", "moderate")
    state_manager.init_item("source_paths", [])
    state_manager.init_item("target_path", None)
    state_manager.init_item("output_path", None)
    state_manager.init_item("log_level", "info")
    state_manager.init_item("processors", ["face_swapper", "face_enhancer"])
    log.info("facefusion state initialized")

def configure_state(**kwargs):
    """Set FaceFusion state from keyword arguments. Thread-safe."""
    with _state_lock:
        init_facefusion_state()
        for key, val in kwargs.items():
            if val is None:
                continue
            if key == "execution_providers":
                state_manager.set_item("execution_providers", [p.strip() for p in val.split(",") if p.strip()])
            elif key == "face_detector_model":
                state_manager.set_item("face_detector_model", val)
            elif key == "face_detector_size":
                state_manager.set_item("face_detector_size", val)
            elif key == "face_detector_angles":
                # Accept both list-of-int and comma-separated string
                if isinstance(val, (list, tuple)):
                    state_manager.set_item("face_detector_angles", list(val))
                else:
                    state_manager.set_item("face_detector_angles", [int(a.strip()) for a in val.split(",") if a.strip()])
            elif key == "face_detector_score":
                state_manager.set_item("face_detector_score", float(val))
            elif key == "face_landmarker_model":
                state_manager.set_item("face_landmarker_model", val)
            elif key == "face_landmarker_score":
                state_manager.set_item("face_landmarker_score", float(val))
            elif key == "face_selector_mode":
                state_manager.set_item("face_selector_mode", val)
            elif key == "face_selector_order":
                state_manager.set_item("face_selector_order", val)
            elif key == "face_selector_gender":
                state_manager.set_item("face_selector_gender", None if val == "none" else val)
            elif key == "reference_face_position":
                state_manager.set_item("reference_face_position", int(val))
            elif key == "reference_face_distance":
                state_manager.set_item("reference_face_distance", float(val))
            elif key == "face_occluder_model":
                state_manager.set_item("face_occluder_model", val)
            elif key == "face_parser_model":
                state_manager.set_item("face_parser_model", val)
            elif key == "face_mask_types":
                state_manager.set_item("face_mask_types", [s.strip() for s in val.split(",") if s.strip()])
            elif key == "face_mask_areas":
                state_manager.set_item("face_mask_areas", [s.strip() for s in val.split(",") if s.strip()])
            elif key == "face_mask_regions":
                state_manager.set_item("face_mask_regions", [s.strip() for s in val.split(",") if s.strip()])
            elif key == "face_mask_blur":
                state_manager.set_item("face_mask_blur", float(val))
            elif key == "face_mask_padding":
                state_manager.set_item("face_mask_padding", val)
            elif key == "face_swapper_model":
                state_manager.set_item("face_swapper_model", val)
            elif key == "face_swapper_pixel_boost":
                state_manager.set_item("face_swapper_pixel_boost", val)
            elif key == "face_swapper_weight":
                state_manager.set_item("face_swapper_weight", float(val))
            elif key == "face_enhancer_model":
                state_manager.set_item("face_enhancer_model", val)
            elif key == "face_enhancer_blend":
                state_manager.set_item("face_enhancer_blend", int(val))
            elif key == "face_enhancer_weight":
                state_manager.set_item("face_enhancer_weight", float(val))

        proc_list = kwargs.get("processors", ["face_swapper", "face_enhancer"])
        if isinstance(proc_list, str):
            proc_list = [s.strip() for s in proc_list.split(",") if s.strip()]
        state_manager.set_item("processors", proc_list)

        ep_str = kwargs.get("execution_providers", "cuda")
        if isinstance(ep_str, (list, tuple)):
            providers = list(ep_str)
        else:
            providers = [p.strip() for p in ep_str.split(",") if p.strip()]
        state_manager.set_item("execution_providers", providers)
        state_manager.set_item("execution_device_ids", [0])
    log.info("configure_state done | execution_providers=%s | onnxruntime available: %s",
             providers,
             __import__("onnxruntime").get_available_providers())

def download_all_models():
    import facefusion.face_detector
    import facefusion.face_landmarker
    import facefusion.face_masker
    from facefusion.processors.modules.face_swapper.core import pre_check as swapper_check
    from facefusion.processors.modules.face_enhancer.core import pre_check as enhancer_check
    log.info("Checking face detection models...")
    facefusion.face_detector.pre_check()
    log.info("Checking face landmarker models...")
    facefusion.face_landmarker.pre_check()
    log.info("Checking face masker models...")
    facefusion.face_masker.pre_check()
    log.info("Checking face swapper model...")
    swapper_check()
    log.info("Checking face enhancer model...")
    enhancer_check()
    log.info("All models checked")

def apply_face_swapper(source_frame, target_frame, reference_frame=None):
    from facefusion.processors.modules.face_swapper.core import process_frame as swapper_process
    mask = np.zeros(target_frame.shape[:2], dtype=np.uint8)
    ref = reference_frame if reference_frame is not None else target_frame
    inputs = {
        "reference_vision_frame": ref,
        "source_vision_frames": [source_frame],
        "target_vision_frame": target_frame,
        "temp_vision_frame": target_frame.copy(),
        "temp_vision_mask": mask
    }
    result, _ = swapper_process(inputs)
    return result

def apply_face_enhancer(target_frame, reference_frame=None):
    from facefusion.processors.modules.face_enhancer.core import process_frame as enhancer_process
    mask = np.zeros(target_frame.shape[:2], dtype=np.uint8)
    ref = reference_frame if reference_frame is not None else target_frame
    inputs = {
        "reference_vision_frame": ref,
        "target_vision_frame": target_frame,
        "temp_vision_frame": target_frame.copy(),
        "temp_vision_mask": mask
    }
    result, _ = enhancer_process(inputs)
    return result

def process_faces(source_tensor, target_tensor, reference_tensor=None, face_enhancer_enabled=True, **kwargs):
    configure_state(**kwargs)

    # Take first image of source batch as reference face
    source_frame = tensor_to_vision_frame(source_tensor)
    log.info("Source face extracted from image %d/%d", 1, source_tensor.shape[0] if source_tensor.dim() == 4 else 1)

    # Use explicit reference image if provided, otherwise target acts as reference
    reference_frame = None
    if reference_tensor is not None:
        reference_frame = tensor_to_vision_frame(reference_tensor)
        log.info("Reference face provided externally")

    # Batch process all target images
    batch_size = target_tensor.shape[0]
    log.info("Processing batch of %d target images...", batch_size)
    results = []

    for i in range(batch_size):
        target_frame = tensor_to_vision_frame(target_tensor[i:i+1])
        current_frame = target_frame.copy()
        ref_frame = reference_frame if reference_frame is not None else target_frame.copy()

        if "face_swapper" in state_manager.get_item("processors"):
            current_frame = apply_face_swapper(source_frame, current_frame, ref_frame)
        if face_enhancer_enabled and "face_enhancer" in state_manager.get_item("processors"):
            current_frame = apply_face_enhancer(current_frame, ref_frame)

        results.append(vision_frame_to_tensor(current_frame))

    return torch.cat(results, dim=0)


class FaceFusionSwapNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "target_image": ("IMAGE",),
                "face_swapper_model": (["hyperswap_1c_256", "hyperswap_1a_256", "hyperswap_1b_256", "inswapper_128", "inswapper_128_fp16", "ghost_1_256", "ghost_2_256", "ghost_3_256", "blendswap_256", "hififace_unofficial_256", "simswap_256", "simswap_unofficial_512", "uniface_256"], {"default": "hyperswap_1c_256"}),
                "face_swapper_pixel_boost": (["256x256", "512x512", "768x768", "1024x1024"], {"default": "256x256"}),
                "face_swapper_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "face_enhancer_model": (["gfpgan_1.4", "gfpgan_1.2", "gfpgan_1.3", "codeformer", "gpen_bfr_256", "gpen_bfr_512", "gpen_bfr_1024", "gpen_bfr_2048", "restoreformer_plus_plus"], {"default": "gfpgan_1.4"}),
                "face_enhancer_blend": ("INT", {"default": 80, "min": 0, "max": 100, "step": 1}),
                "face_enhancer_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "face_enhancer_enabled": ("BOOLEAN", {"default": True}),
                "face_detector_model": (["yolo_face", "retinaface", "scrfd", "yunet", "many"], {"default": "yolo_face"}),
                "face_detector_size": (["640x640", "320x320", "480x480", "512x512", "160x160"], {"default": "640x640"}),
                "face_detector_angles": (["0,90,180,270", "0", "0,90", "0,180", "0,90,180"], {"default": "0,90,180,270"}),
                "face_detector_score": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "face_landmarker_model": (["2dfan4", "peppapig"], {"default": "2dfan4"}),
                "face_landmarker_score": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "face_selector_order": (["large-small", "small-large", "left-right", "right-left", "top-bottom", "bottom-top", "best-worst", "worst-best"], {"default": "large-small"}),
                "face_selector_gender": (["none", "female", "male"], {"default": "none"}),
                "reference_face_distance": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
                "face_occluder_model": (["xseg_1", "xseg_2", "xseg_3", "many"], {"default": "xseg_1"}),
                "face_parser_model": (["bisenet_resnet_34", "bisenet_resnet_18"], {"default": "bisenet_resnet_34"}),
                "face_mask_types": (["box,occlusion,area,region", "box,occlusion", "box,area,region", "box,occlusion,area", "box", "occlusion"], {"default": "box,occlusion,area,region"}),
                "face_mask_areas": (["upper-face,lower-face,mouth", "upper-face,lower-face", "upper-face,mouth", "lower-face"], {"default": "upper-face,lower-face,mouth"}),
                "face_mask_regions": (["skin,left-eyebrow,right-eyebrow,left-eye,right-eye,glasses,nose,mouth,upper-lip,lower-lip", "skin,left-eye,right-eye,nose,mouth", "skin,nose,mouth", "skin,mouth"], {"default": "skin,left-eyebrow,right-eyebrow,left-eye,right-eye,glasses,nose,mouth,upper-lip,lower-lip"}),
                "face_mask_blur": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "execution_providers": (["cuda", "cpu", "cuda,tensorrt"], {"default": "cuda"}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "swap"
    CATEGORY = "FaceFusion"

    def swap(self, source_image, target_image, reference_image=None, face_enhancer_enabled=True, **kwargs):
        return (process_faces(source_image, target_image,
                              reference_tensor=reference_image,
                              face_enhancer_enabled=face_enhancer_enabled,
                              **kwargs),)
