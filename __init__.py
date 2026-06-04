import sys
import os
import logging
log = logging.getLogger(__name__)
sys.path.insert(0, os.path.dirname(__file__))
from .facefusion_wrapper import FaceFusionSwapNode
from .facefusion_video import FaceFusionVideoSwapNode
NODE_CLASS_MAPPINGS = {"FaceFusionSwapNode": FaceFusionSwapNode, "FaceFusionVideoSwapNode": FaceFusionVideoSwapNode}
NODE_DISPLAY_NAME_MAPPINGS = {"FaceFusionSwapNode": "FaceFusion Face Swap", "FaceFusionVideoSwapNode": "FaceFusion Video Face Swap"}
log.info("ComfyUI-FaceFusion loaded!")
