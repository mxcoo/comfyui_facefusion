import time
import logging
import os
import torch
import numpy as np
from fractions import Fraction
from functools import partial
from concurrent.futures import ThreadPoolExecutor

log = logging.getLogger(__name__)

from facefusion_wrapper import configure_state, tensor_to_vision_frame, vision_frame_to_tensor, apply_face_swapper, apply_face_enhancer


def _one_frame(frame_bgr, src_bgr, ref_bgr, enh_enabled, expression_enabled):
    """Process a single frame — picklable for ThreadPoolExecutor."""
    c = frame_bgr.copy()
    r = ref_bgr if ref_bgr is not None else frame_bgr.copy()
    c = apply_face_swapper(src_bgr, c, r)
    if enh_enabled:
        c = apply_face_enhancer(c, r)
    if expression_enabled:
        from facefusion_wrapper import apply_expression_restorer as _apply_er
        c = _apply_er(c, r)
    return c


class FaceFusionVideoSwapNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_images": ("IMAGE",),
                "target_video": ("VIDEO",),
                "face_swapper_model": (["hyperswap_1c_256","hyperswap_1a_256","hyperswap_1b_256","inswapper_128","inswapper_128_fp16","ghost_1_256","ghost_2_256","ghost_3_256","blendswap_256","hififace_unofficial_256","simswap_256","simswap_unofficial_512","uniface_256"], {"default":"hyperswap_1c_256"}),
                "face_swapper_pixel_boost": (["256x256","512x512","768x768","1024x1024"], {"default":"256x256"}),
                "face_swapper_weight": ("FLOAT", {"default":0.5,"min":0.0,"max":1.0,"step":0.05}),
                "face_enhancer_model": (["gfpgan_1.4","gfpgan_1.2","gfpgan_1.3","codeformer","gpen_bfr_256","gpen_bfr_512","gpen_bfr_1024","gpen_bfr_2048","restoreformer_plus_plus"], {"default":"gfpgan_1.4"}),
                "face_enhancer_blend": ("INT", {"default":80,"min":0,"max":100,"step":1}),
                "face_enhancer_weight": ("FLOAT", {"default":0.5,"min":0.0,"max":1.0,"step":0.05}),
                "face_enhancer_enabled": ("BOOLEAN", {"default": False}),
                "face_detector_model": (["yolo_face","retinaface","scrfd","yunet","many"], {"default":"yolo_face"}),
                "face_detector_size": (["640x640","320x320","480x480","512x512","160x160"], {"default":"640x640"}),
                "face_detector_angles": (["0,90,180,270","0","0,90","0,180","0,90,180"], {"default":"0,90,180,270"}),
                "face_detector_score": ("FLOAT", {"default":0.5,"min":0.0,"max":1.0,"step":0.05}),
                "face_landmarker_model": (["2dfan4","peppapig"], {"default":"2dfan4"}),
                "face_landmarker_score": ("FLOAT", {"default":0.5,"min":0.0,"max":1.0,"step":0.05}),
                "face_selector_order": (["large-small","small-large","left-right","right-left","top-bottom","bottom-top","best-worst","worst-best"], {"default":"large-small"}),
                "face_selector_gender": (["none","female","male"], {"default":"none"}),
                "reference_face_distance": ("FLOAT", {"default":0.6,"min":0.0,"max":1.0,"step":0.05}),
                "face_occluder_model": (["xseg_1","xseg_2","xseg_3","many"], {"default":"xseg_1"}),
                "face_parser_model": (["bisenet_resnet_34","bisenet_resnet_18"], {"default":"bisenet_resnet_34"}),
                "face_mask_types": (["box,occlusion,area,region","box,occlusion","box,area,region","box,occlusion,area","box","occlusion"], {"default":"box,occlusion,area,region"}),
                "face_mask_areas": (["upper-face,lower-face,mouth","upper-face,lower-face","upper-face,mouth","lower-face"], {"default":"upper-face,lower-face,mouth"}),
                "face_mask_regions": (["skin,left-eyebrow,right-eyebrow,left-eye,right-eye,glasses,nose,mouth,upper-lip,lower-lip","skin,left-eye,right-eye,nose,mouth","skin,nose,mouth","skin,mouth"], {"default":"skin,left-eyebrow,right-eyebrow,left-eye,right-eye,glasses,nose,mouth,upper-lip,lower-lip"}),
                "face_mask_blur": ("FLOAT", {"default":0.3,"min":0.0,"max":1.0,"step":0.05}),
                "expression_restorer_enabled": ("BOOLEAN", {"default": False}),
                "expression_restorer_model": (["live_portrait"], {"default":"live_portrait"}),
                "expression_restorer_factor": ("INT", {"default":80,"min":0,"max":100,"step":1}),
                "expression_restorer_areas": (["upper-face,lower-face","upper-face","lower-face"], {"default":"upper-face,lower-face"}),
                "execution_providers": (["cuda","cpu","cuda,tensorrt"], {"default":"cuda"}),
                "thread_count": ("INT", {"default": 2, "min": 1, "max": 4}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
            }
        }
    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "process"
    CATEGORY = "FaceFusion"

    def process(self, source_images, target_video, reference_image=None,
                face_enhancer_enabled=False, expression_restorer_enabled=False,
                thread_count=2, **kwargs):
        from comfy_api.latest import Types, InputImpl

        # Get frames from ComfyUI video object
        vc = target_video.get_components() if hasattr(target_video, "get_components") else target_video
        frames = vc.images if hasattr(vc, "images") else target_video
        audio = vc.audio if hasattr(vc, "audio") else None
        fr_val = float(vc.frame_rate) if hasattr(vc, "frame_rate") and vc.frame_rate else 30.0

        # Build processors list based on enabled features
        from facefusion_wrapper import configure_state
        proc_list = ['face_swapper']
        if face_enhancer_enabled:
            proc_list.append('face_enhancer')
        if expression_restorer_enabled:
            proc_list.append('expression_restorer')
        kwargs['processors'] = proc_list
        configure_state(**kwargs)

        src = source_images[0:1] if source_images.dim() == 4 and source_images.shape[0] > 1 else source_images
        src_bgr = tensor_to_vision_frame(src)

        # Resolve reference frame
        ref_bgr = None
        if reference_image is not None:
            ref_bgr = tensor_to_vision_frame(reference_image)
            log.info("Using external reference image for face selection")
        else:
            if frames.shape[0] > 0:
                ref_bgr = tensor_to_vision_frame(frames[0].unsqueeze(0))

        batch = frames.shape[0]
        # Limit max frames to prevent memory overflow
        max_frames = 300
        if batch > max_frames:
            log.warning("Video has %d frames — limiting to first %d to prevent OOM", batch, max_frames)
            batch = max_frames

        log.info("Processing %d frames (%.2f fps, threads=%d, enh=%s)...",
                 batch, fr_val, thread_count, face_enhancer_enabled)
        start = time.time()

        if thread_count <= 1 or batch <= 1:
            # Single-threaded path
            results = []
            for i in range(batch):
                f_bgr = tensor_to_vision_frame(frames[i].unsqueeze(0))
                c = _one_frame(f_bgr, src_bgr, ref_bgr, face_enhancer_enabled, expression_restorer_enabled)
                results.append(vision_frame_to_tensor(c))
                if (i + 1) % 30 == 0 or i == batch - 1:
                    e = time.time() - start
                    fps = (i + 1) / e if e > 0 else 0
                    eta = (batch - i - 1) / fps if fps > 0 else 0
                    log.info("  [%d/%d] %.1f fps, ETA: %.0fs", i + 1, batch, fps, eta)
        else:
            # Multi-threaded path
            frame_bgrs = [tensor_to_vision_frame(frames[i].unsqueeze(0)) for i in range(batch)]
            worker = partial(_one_frame, src_bgr=src_bgr, ref_bgr=ref_bgr, enh_enabled=face_enhancer_enabled, expression_enabled=expression_restorer_enabled)
            with ThreadPoolExecutor(max_workers=thread_count) as ex:
                out_frames = list(ex.map(worker, frame_bgrs))
            results = [vision_frame_to_tensor(f) for f in out_frames]

        log.info("Done: %d frames in %.1fs", batch, time.time() - start)

        output_vc = Types.VideoComponents(
            images=torch.cat(results, dim=0),
            audio=audio,
            frame_rate=Fraction(fr_val).limit_denominator()
        )
        return (InputImpl.VideoFromComponents(output_vc),)
