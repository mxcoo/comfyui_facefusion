import sys
import os
import logging

log = logging.getLogger(__name__)

# === CUDA DLL path setup — must run BEFORE any onnxruntime import ===
# PyTorch ships CUDA runtime DLLs (cudart64_12.dll, cublas64_12.dll, etc.)
# in torch/lib/.  onnxruntime-gpu's CUDA provider DLL needs those DLLs on
# PATH or added via os.add_dll_directory().  We do both here so that
# CUDAExecutionProvider is available even if onnxruntime was imported
# earlier by ComfyUI or another node.

import torch
_torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
if os.path.isdir(_torch_lib):
    os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")
    try:
        os.add_dll_directory(_torch_lib)
    except AttributeError:
        pass  # Python < 3.8

# Add NVIDIA CUDA DLL paths from torch site-packages
_torch_sp = os.path.dirname(os.path.dirname(_torch_lib))
if os.path.isdir(_torch_sp):
    _nvidia_dirs = [
        os.path.join(_torch_sp, "nvidia", "cuda_runtime", "bin"),
        os.path.join(_torch_sp, "nvidia", "cublas", "bin"),
        os.path.join(_torch_sp, "nvidia", "cufft", "bin"),
        os.path.join(_torch_sp, "nvidia", "curand", "bin"),
        os.path.join(_torch_sp, "nvidia", "cuda_nvrtc", "bin"),
        os.path.join(_torch_sp, "nvidia", "nvjitlink", "bin"),
    ]
    for _d in _nvidia_dirs:
        if os.path.isdir(_d):
            os.environ["PATH"] = _d + os.pathsep + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(_d)
            except AttributeError:
                pass

# CUDA check (torch already imported above)
log.info("PyTorch CUDA available: %s (version: %s)", torch.cuda.is_available(), torch.version.cuda if hasattr(torch.version, 'cuda') else '?')

# Log onnxruntime CUDA availability early
try:
    import onnxruntime
    _providers = onnxruntime.get_available_providers()
    log.info("onnxruntime %s providers: %s | CUDA: %s", onnxruntime.__version__, _providers, "CUDAExecutionProvider" in _providers)
except Exception:
    log.warning("onnxruntime not available yet")

# === Import our node classes ===
sys.path.insert(0, os.path.dirname(__file__))
from .facefusion_wrapper import FaceFusionSwapNode
from .facefusion_video import FaceFusionVideoSwapNode

NODE_CLASS_MAPPINGS = {
    "FaceFusionSwapNode": FaceFusionSwapNode,
    "FaceFusionVideoSwapNode": FaceFusionVideoSwapNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceFusionSwapNode": "FaceFusion Face Swap",
    "FaceFusionVideoSwapNode": "FaceFusion Video Face Swap",
}

log.info("ComfyUI-FaceFusion loaded!")
