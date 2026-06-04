import os
import sys
import logging
import numpy as np
import cv2
import torch

log = logging.getLogger(__name__)

FF_DIR = os.path.dirname(__file__)
sys.path.insert(0, FF_DIR)

import facefusion
from facefusion import state_manager

def tensor_to_vision_frame(tensor):
    img = tensor.cpu().numpy().squeeze(0)
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255.0).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def vision_frame_to_tensor(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).unsqueeze(0)

def init_facefusion_state():
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
    from facefusion.processors.modules.face_swapper.types import FaceSwapperInputs
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
    from facefusion.processors.modules.face_enhancer.types import FaceEnhancerInputs
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

def process_faces(source_tensor, target_tensor, **kwargs):
    init_facefusion_state()
    for key, val in kwargs.items():
        if val is None:
            continue
        if key == "execution_providers":
            ep = [p.strip() for p in val.split(",") if p.strip()]
            state_manager.set_item("execution_providers", ep)
        elif key == "face_detector_model":
            state_manager.set_item("face_detector_model", val)
        elif key == "face_detector_size":
            state_manager.set_item("face_detector_size", val)
        elif key == "face_detector_angles":
            angles = [int(a.strip()) for a in val.split(",") if a.strip()]
            state_manager.set_item("face_detector_angles", angles)
        elif key == "face_detector_score":
            state_manager.set_item("face_detector_score", float(val))
        elif key == "face_landmarker_model":
            state_manager.set_item("face_landmarker_model", val)
        elif key == "face_landmarker_score":
            state_manager.set_item("face_landmarker_score", float(val))
        elif key == "face_selector_order":
            state_manager.set_item("face_selector_order", val)
        elif key == "face_selector_gender":
            state_manager.set_item("face_selector_gender", None if val == "none" else val)
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

    proc_list = ["face_swapper", "face_enhancer"]
    state_manager.set_item("processors", proc_list)

    source_frame = tensor_to_vision_frame(source_tensor)
    target_frame = tensor_to_vision_frame(target_tensor)
    current_frame = target_frame.copy()
    reference_frame = target_frame.copy()

    if "face_swapper" in proc_list:
        log.info("Running face_swapper...")
        current_frame = apply_face_swapper(source_frame, current_frame, reference_frame)
    if "face_enhancer" in proc_list:
        log.info("Running face_enhancer...")
        current_frame = apply_face_enhancer(current_frame, reference_frame)

    return vision_frame_to_tensor(current_frame)


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
                "face_detector_model": (["yolo_face", "retinaface", "scrfd", "yunet", "many"], {"default": "yolo_face"}),
                "face_detector_size": (["640x640", "320x320", "480x480", "512x512", "160x160"], {"default": "640x640"}),
                "face_detector_angles": (["0,90,180,270", "0", "0,90", "0,180", "0,90,180"], {"default": "0,90,180,270"}),
                "face_detector_score": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "face_landmarker_model": (["2dfan4", "peppapig"], {"default": "2dfan4"}),
                "face_landmarker_score": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "face_selector_order": (["large-small", "small-large", "left-right", "right-left", "top-bottom", "bottom-top", "best-worst", "worst-best"], {"default": "large-small"}),
                "face_selector_gender": (["none", "female", "male"], {"default": "none"}),
                "face_occluder_model": (["xseg_1", "xseg_2", "xseg_3", "many"], {"default": "xseg_1"}),
                "face_parser_model": (["bisenet_resnet_34", "bisenet_resnet_18"], {"default": "bisenet_resnet_34"}),
                "face_mask_types": (["box,occlusion,area,region", "box,occlusion", "box,area,region", "box,occlusion,area", "box", "occlusion"], {"default": "box,occlusion,area,region"}),
                "face_mask_areas": (["upper-face,lower-face,mouth", "upper-face,lower-face", "upper-face,mouth", "lower-face"], {"default": "upper-face,lower-face,mouth"}),
                "face_mask_regions": (["skin,left-eyebrow,right-eyebrow,left-eye,right-eye,glasses,nose,mouth,upper-lip,lower-lip", "skin,left-eye,right-eye,nose,mouth", "skin,nose,mouth", "skin,mouth"], {"default": "skin,left-eyebrow,right-eyebrow,left-eye,right-eye,glasses,nose,mouth,upper-lip,lower-lip"}),
                "face_mask_blur": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "execution_providers": (["cuda", "cpu", "cuda,tensorrt"], {"default": "cuda"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "swap"
    CATEGORY = "FaceFusion"

    def swap(self, source_image, target_image, **kwargs):
        return (process_faces(source_image, target_image, **kwargs),)
