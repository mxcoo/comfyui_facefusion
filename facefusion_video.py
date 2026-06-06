import time
import logging
import os
import torch
from fractions import Fraction

log = logging.getLogger(__name__)


def configure_state(**kwargs):
    from facefusion_wrapper import init_facefusion_state
    from facefusion import state_manager
    init_facefusion_state()
    for key, val in kwargs.items():
        if val is None: continue
        if key == "execution_providers":
            state_manager.set_item("execution_providers", [p.strip() for p in val.split(",") if p.strip()])
        elif key == "face_detector_model": state_manager.set_item("face_detector_model", val)
        elif key == "face_detector_size": state_manager.set_item("face_detector_size", val)
        elif key == "face_detector_angles": state_manager.set_item("face_detector_angles", [int(a.strip()) for a in val.split(",") if a.strip()])
        elif key == "face_detector_score": state_manager.set_item("face_detector_score", float(val))
        elif key == "face_landmarker_model": state_manager.set_item("face_landmarker_model", val)
        elif key == "face_landmarker_score": state_manager.set_item("face_landmarker_score", float(val))
        elif key == "face_selector_order": state_manager.set_item("face_selector_order", val)
        elif key == "face_selector_gender": state_manager.set_item("face_selector_gender", None if val == "none" else val)
        elif key == "face_occluder_model": state_manager.set_item("face_occluder_model", val)
        elif key == "face_parser_model": state_manager.set_item("face_parser_model", val)
        elif key == "face_mask_types": state_manager.set_item("face_mask_types", [s.strip() for s in val.split(",") if s.strip()])
        elif key == "face_mask_areas": state_manager.set_item("face_mask_areas", [s.strip() for s in val.split(",") if s.strip()])
        elif key == "face_mask_regions": state_manager.set_item("face_mask_regions", [s.strip() for s in val.split(",") if s.strip()])
        elif key == "face_mask_blur": state_manager.set_item("face_mask_blur", float(val))
        elif key == "face_swapper_model": state_manager.set_item("face_swapper_model", val)
        elif key == "face_swapper_pixel_boost": state_manager.set_item("face_swapper_pixel_boost", val)
        elif key == "face_swapper_weight": state_manager.set_item("face_swapper_weight", float(val))
        elif key == "face_enhancer_model": state_manager.set_item("face_enhancer_model", val)
        elif key == "face_enhancer_blend": state_manager.set_item("face_enhancer_blend", int(val))
        elif key == "face_enhancer_weight": state_manager.set_item("face_enhancer_weight", float(val))
    state_manager.set_item("processors", ["face_swapper", "face_enhancer"])
    log.info("configure_state done | execution_providers=%s | onnxruntime available: %s",
             state_manager.get_item("execution_providers"),
             __import__("onnxruntime").get_available_providers())


class FaceFusionVideoSwapNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "source_images": ("IMAGE",),
            "target_video": ("VIDEO",),
            "face_swapper_model": (["hyperswap_1c_256","hyperswap_1a_256","hyperswap_1b_256","inswapper_128","inswapper_128_fp16","ghost_1_256","ghost_2_256","ghost_3_256","blendswap_256","hififace_unofficial_256","simswap_256","simswap_unofficial_512","uniface_256"], {"default":"hyperswap_1c_256"}),
            "face_swapper_pixel_boost": (["256x256","512x512","768x768","1024x1024"], {"default":"256x256"}),
            "face_swapper_weight": ("FLOAT", {"default":0.5,"min":0.0,"max":1.0,"step":0.05}),
            "face_enhancer_model": (["gfpgan_1.4","gfpgan_1.2","gfpgan_1.3","codeformer","gpen_bfr_256","gpen_bfr_512","gpen_bfr_1024","gpen_bfr_2048","restoreformer_plus_plus"], {"default":"gfpgan_1.4"}),
            "face_enhancer_blend": ("INT", {"default":80,"min":0,"max":100,"step":1}),
            "face_enhancer_weight": ("FLOAT", {"default":0.5,"min":0.0,"max":1.0,"step":0.05}),
            "face_detector_model": (["yolo_face","retinaface","scrfd","yunet","many"], {"default":"yolo_face"}),
            "face_detector_size": (["640x640","320x320","480x480","512x512","160x160"], {"default":"640x640"}),
            "face_detector_angles": (["0,90,180,270","0","0,90","0,180","0,90,180"], {"default":"0,90,180,270"}),
            "face_detector_score": ("FLOAT", {"default":0.5,"min":0.0,"max":1.0,"step":0.05}),
            "face_landmarker_model": (["2dfan4","peppapig"], {"default":"2dfan4"}),
            "face_landmarker_score": ("FLOAT", {"default":0.5,"min":0.0,"max":1.0,"step":0.05}),
            "face_selector_order": (["large-small","small-large","left-right","right-left","top-bottom","bottom-top","best-worst","worst-best"], {"default":"large-small"}),
            "face_selector_gender": (["none","female","male"], {"default":"none"}),
            "face_occluder_model": (["xseg_1","xseg_2","xseg_3","many"], {"default":"xseg_1"}),
            "face_parser_model": (["bisenet_resnet_34","bisenet_resnet_18"], {"default":"bisenet_resnet_34"}),
            "face_mask_types": (["box,occlusion,area,region","box,occlusion","box,area,region","box,occlusion,area","box","occlusion"], {"default":"box,occlusion,area,region"}),
            "face_mask_areas": (["upper-face,lower-face,mouth","upper-face,lower-face","upper-face,mouth","lower-face"], {"default":"upper-face,lower-face,mouth"}),
            "face_mask_regions": (["skin,left-eyebrow,right-eyebrow,left-eye,right-eye,glasses,nose,mouth,upper-lip,lower-lip","skin,left-eye,right-eye,nose,mouth","skin,nose,mouth","skin,mouth"], {"default":"skin,left-eyebrow,right-eyebrow,left-eye,right-eye,glasses,nose,mouth,upper-lip,lower-lip"}),
            "face_mask_blur": ("FLOAT", {"default":0.3,"min":0.0,"max":1.0,"step":0.05}),
            "execution_providers": (["cuda","cpu","cuda,tensorrt"], {"default":"cuda"}),
        }}
    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "process"
    CATEGORY = "FaceFusion"

    def process(self, source_images, target_video, **kwargs):
        from facefusion_wrapper import tensor_to_vision_frame, vision_frame_to_tensor, apply_face_swapper, apply_face_enhancer
        from comfy_api.latest import Types, InputImpl

        # Get frames from ComfyUI video object
        vc = target_video.get_components() if hasattr(target_video, "get_components") else target_video
        frames = vc.images if hasattr(vc, "images") else target_video
        audio = vc.audio if hasattr(vc, "audio") else None
        fr_val = float(vc.frame_rate) if hasattr(vc, "frame_rate") and vc.frame_rate else 30.0

        configure_state(**kwargs)

        src = source_images[0:1] if source_images.dim() == 4 and source_images.shape[0] > 1 else source_images
        src_bgr = tensor_to_vision_frame(src)
        batch = frames.shape[0]

        log.info("Processing %d frames (%.2f fps)...", batch, fr_val)
        results = []
        start = time.time()

        for i in range(batch):
            f_bgr = tensor_to_vision_frame(frames[i:i+1])
            c = f_bgr.copy()
            r = f_bgr.copy()
            c = apply_face_swapper(src_bgr, c, r)
            c = apply_face_enhancer(c, r)
            results.append(vision_frame_to_tensor(c))
            if (i+1) % 30 == 0 or i == batch-1:
                e = time.time() - start
                s = (i+1)/e if e > 0 else 0
                rem = (batch-i-1)/s if s > 0 else 0
                log.info("  [%d/%d] %.1f fps, ETA: %.0fs", i+1, batch, s, rem)

        log.info("Done: %d frames in %.1fs", batch, time.time()-start)

        output_vc = Types.VideoComponents(
            images=torch.cat(results, dim=0),
            audio=audio,
            frame_rate=Fraction(fr_val).limit_denominator()
        )
        return (InputImpl.VideoFromComponents(output_vc),)
