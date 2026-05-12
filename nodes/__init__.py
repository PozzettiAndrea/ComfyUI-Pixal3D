"""ComfyUI-Pixal3D nodes — aggregated NODE_CLASS_MAPPINGS for register_nodes()."""

# --- cuda-wheels variant aliases ------------------------------------------------
# We use the _vb_ap / _ap variants from cuda-wheels (nvdiffrast-free, MIT-aligned).
# The vendored pixal3d code (and our stages.py) imports the bare module names.
# Install sys.modules aliases BEFORE any pixal3d / stages import.
import sys as _sys

for _real, _aliases in [
    ("cumesh_vb", ["cumesh"]),
    ("o_voxel_vb_ap", ["o_voxel"]),
    ("flex_gemm_ap", ["flex_gemm"]),
]:
    try:
        _mod = __import__(_real)
        for _alias in _aliases:
            _sys.modules[_alias] = _mod
    except ImportError:
        pass
del _sys, _real, _aliases, _mod, _alias

from .nodes_loader import (
    NODE_CLASS_MAPPINGS as loader_mappings,
    NODE_DISPLAY_NAME_MAPPINGS as loader_display,
)
from .nodes_inference import (
    NODE_CLASS_MAPPINGS as inference_mappings,
    NODE_DISPLAY_NAME_MAPPINGS as inference_display,
)

NODE_CLASS_MAPPINGS = {
    **loader_mappings,
    **inference_mappings,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **loader_display,
    **inference_display,
}
