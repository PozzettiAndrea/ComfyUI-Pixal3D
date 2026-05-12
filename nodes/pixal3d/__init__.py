# cuda-wheels variant aliases — pixal3d code imports bare `cumesh` / `o_voxel` / `flex_gemm`,
# but the wrapper installs the *_vb / *_vb_ap / *_ap forks. Install aliases on first import.
import sys as _sys

for _real, _aliases in [
    ("cumesh_vb", ["cumesh"]),
    ("o_voxel_vb_ap", ["o_voxel"]),
    ("flex_gemm_ap", ["flex_gemm"]),
]:
    if _real in _sys.modules:
        _mod = _sys.modules[_real]
    else:
        try:
            _mod = __import__(_real)
        except ImportError:
            continue
    for _alias in _aliases:
        _sys.modules.setdefault(_alias, _mod)
del _sys

from . import models
from . import modules
from . import pipelines
from . import representations
from . import utils
