"""Pixal3D pipeline stages.

Module-level caches: first call loads from disk, subsequent calls reuse from RAM.
All heavy work happens inside the isolated comfy-env subprocess.
"""

import gc
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

# Pixal3D's image-cond models call `torch.hub.load("valeoai/NAF", "naf", ...)` which
# hits api.github.com for repo validation. If a bad GITHUB_TOKEN is in the env
# (common with multi-account dev setups), the GitHub API returns 401 and torch.hub
# refuses to download. Strip the token here — public repo validation does not
# require auth.
os.environ.pop("GITHUB_TOKEN", None)

import numpy as np
import torch
from PIL import Image

import comfy.model_management
import comfy.utils
import folder_paths

log = logging.getLogger("pixal3d")


# ============================================================================
# Per-phase timing print (visibility during the slow ~150 s cold-boot)
# ============================================================================

class _phase:
    """Context manager that prints `[pixal3d] <label> ... <elapsed>s` on exit.

    Writes to stderr with flush=True so ComfyUI's worker console surfaces it in
    real time (its stdout is line-buffered for tqdm). Reports failures too.
    """

    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        import sys, time
        self._t0 = time.perf_counter()
        print(f"[pixal3d] >>> {self.label} ...", file=sys.stderr, flush=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        import sys, time
        dt = time.perf_counter() - self._t0
        if exc_type is None:
            print(f"[pixal3d] <<< {self.label}  ({dt:.1f}s)", file=sys.stderr, flush=True)
        else:
            print(f"[pixal3d] !!! {self.label} FAILED after {dt:.1f}s: {exc_type.__name__}: {exc}", file=sys.stderr, flush=True)
        return False  # don't swallow exceptions

# ============================================================================
# HuggingFace progress shim
# ============================================================================

def _comfy_tqdm():
    """tqdm that pumps download progress into ComfyUI's UI ProgressBar."""
    try:
        import tqdm as _tqdm_mod
    except ImportError:
        return None
    holder = {"pbar": None, "total": 0, "done": 0}

    class _T(_tqdm_mod.tqdm):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            if self.total and self.total > 0 and holder["pbar"] is None:
                holder["total"] = self.total
                holder["done"] = 0
                holder["pbar"] = comfy.utils.ProgressBar(self.total)

        def update(self, n=1):
            ret = super().update(n)
            if n and holder["pbar"] and holder["total"] > 0:
                holder["done"] = min(holder["done"] + n, holder["total"])
                holder["pbar"].update_absolute(holder["done"], holder["total"])
            return ret

    return _T


# ============================================================================
# Constants from upstream inference.py
# ============================================================================

PIXAL3D_REPO = "TencentARC/Pixal3D"
MOGE_REPO = "Ruicheng/moge-2-vitl"
DINOV3_REPO = "camenduru/dinov3-vitl16-pretrain-lvd1689m"
NAF_REPO = "https://github.com/valeoai/NAF.git"
NAF_CHECKPOINT_URL = "https://github.com/valeoai/NAF/releases/download/model/naf_release.pth"

IMAGE_COND_CONFIGS = {
    "ss": {
        "model_name": DINOV3_REPO,
        "image_size": 512,
        "grid_resolution": 16,
    },
    "shape_512": {
        "model_name": DINOV3_REPO,
        "image_size": 512,
        "grid_resolution": 32,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "shape_1024": {
        "model_name": DINOV3_REPO,
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "tex_1024": {
        "model_name": DINOV3_REPO,
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 1024,
    },
}

# Files we expect in the local pixal3d models dir, mirroring hf_models/TencentARC_Pixal3D.json.
_REQUIRED_FILES = [
    "pipeline.json",
    "ckpts/shape_dec_next_dc_f16c32_fp16.json",
    "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
    "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.json",
    "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors",
    "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.json",
    "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
    "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.json",
    "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors",
    "ckpts/ss_dec_conv3d_16l8_fp16.json",
    "ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
    "ckpts/ss_flow_img_dit_1_3B_64_bf16.json",
    "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
    "ckpts/tex_dec_next_dc_f16c32_fp16.json",
    "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors",
]


# ============================================================================
# Model folder registration
# ============================================================================

_pixal3d_models_dir = os.path.join(folder_paths.models_dir, "pixal3d")
os.makedirs(_pixal3d_models_dir, exist_ok=True)
folder_paths.add_model_folder_path("pixal3d", _pixal3d_models_dir)


def _local_pixal3d_dir() -> Path:
    return Path(_pixal3d_models_dir)


# ============================================================================
# Module-level caches
# ============================================================================

_pipeline = None
_moge_model = None
# id(model_instance) -> ModelPatcher. Used by _wrap_with_comfy_patcher to route
# pixal3d's per-stage .to(device) / .cpu() calls through ComfyUI's memory
# manager (load_models_gpu auto-offloads competing models in the workflow).
_model_patchers: dict = {}
# One nn.Module reused by all 3 cond extractors that need NAF upsampling.
_naf = None


# ============================================================================
# Hardware probe
# ============================================================================

def _check_gpu_or_raise():
    if not torch.cuda.is_available():
        raise RuntimeError("Pixal3D requires a CUDA GPU. No CUDA device detected.")
    cap = torch.cuda.get_device_capability()
    if cap < (8, 0):
        raise RuntimeError(
            f"Pixal3D requires SM >= 8.0 (Ampere/Ada/Hopper/Blackwell). "
            f"Detected SM {cap[0]}.{cap[1]} on {torch.cuda.get_device_name()}. "
            f"flash-attn-3 has no fallback for older GPUs."
        )


# ============================================================================
# Download
# ============================================================================

def _download_pixal3d_weights():
    """Ensure Pixal3D ckpts exist in ComfyUI/models/pixal3d/."""
    from huggingface_hub import hf_hub_download

    local_dir = _local_pixal3d_dir()
    log.info(f"[pixal3d] Ensuring weights present in {local_dir}")

    tqdm_cls = _comfy_tqdm()
    for rel_path in _REQUIRED_FILES:
        target = local_dir / rel_path
        if target.exists():
            continue
        log.info(f"  downloading {rel_path}")
        hf_hub_download(
            repo_id=PIXAL3D_REPO,
            filename=rel_path,
            local_dir=str(local_dir),
            tqdm_class=tqdm_cls,
        )
    return local_dir


def _download_naf() -> Tuple[Path, Path]:
    """Ensure NAF source + checkpoint exist in ComfyUI/models/naf/.

    Replaces torch.hub.load("valeoai/NAF", ...) which (a) hits api.github.com
    on every cold boot, (b) caches under ~/.cache/torch/hub (off the
    ComfyUI/models/ convention), (c) used to break under bad GITHUB_TOKEN env.

    Returns (source_dir, ckpt_path). source_dir is added to sys.path so the
    `NAF` class becomes importable as `src.model.naf.NAF`.
    """
    naf_dir = Path(folder_paths.models_dir) / "naf"
    src_dir = naf_dir / "source"
    ckpt = naf_dir / "naf_release.pth"
    naf_dir.mkdir(parents=True, exist_ok=True)

    if not (src_dir / "hubconf.py").exists():
        import subprocess
        env = {k: v for k, v in os.environ.items() if k != "GITHUB_TOKEN"}
        subprocess.check_call(
            ["git", "clone", "--depth", "1", NAF_REPO, str(src_dir)],
            env=env,
        )
    if not ckpt.exists():
        import urllib.request
        urllib.request.urlretrieve(NAF_CHECKPOINT_URL, str(ckpt))
    return src_dir, ckpt


def _patch_naf_to_local_model():
    """Monkey-patch DinoV3ProjFeatureExtractor._load_naf to (a) load NAF from
    ComfyUI/models/naf/ instead of torch.hub, and (b) reuse ONE shared
    nn.Module across all cond extractors that need it (currently 3 of 4:
    shape_512, shape_1024, tex_1024). NAF has frozen weights and no
    per-instance state, so sharing is safe.

    Idempotent.
    """
    from .pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
        DinoV3ProjFeatureExtractor,
    )
    if getattr(DinoV3ProjFeatureExtractor, "_pixal3d_naf_patched", False):
        return

    src_dir, ckpt = _download_naf()
    # Make `from src.model.naf import NAF` resolve from our local clone.
    import sys
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    def _patched_load_naf(self):
        global _naf
        if _naf is None:
            with _phase("NAF model build + ckpt load"):
                from src.model.naf import NAF  # noqa: F401  (resolved via sys.path)
                m = NAF()
                m.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
                m.eval()
                m.requires_grad_(False)
                _naf = m
        # Bypass nn.Module __setattr__ so PyTorch doesn't register NAF as a
        # child of every cond extractor (which would have each ModelPatcher
        # try to manage the same weights). The forward path just calls
        # `self.naf_model(...)`; attribute access still works.
        self.__dict__["naf_model"] = _naf

    DinoV3ProjFeatureExtractor._load_naf = _patched_load_naf
    DinoV3ProjFeatureExtractor._pixal3d_naf_patched = True


# ============================================================================
# Pipeline init
# ============================================================================

def _stub_rembg():
    """Replace BiRefNet.__init__ with a no-op stub.

    Pixal3D's pipeline.json pins briaai/RMBG-2.0 (gated HF repo). We never call
    rembg from our wrapper anymore -- preprocess_image() below is pure PIL and
    pipeline.run(preprocess_image=False) bypasses pixal3d's own rembg call.
    Stubbing the constructor avoids:
      1. HF 401 on the gated repo at from_pretrained time, and
      2. ~1 GB of unused BiRefNet weights sitting in RAM.

    Background removal is now the user's responsibility -- LoadImage's MASK
    output, or any community rembg node feeding Pixal3DPreprocessImage's mask.
    Idempotent.
    """
    from .pixal3d.pipelines import rembg as _rembg
    if getattr(_rembg.BiRefNet, "_pixal3d_stubbed", False):
        return

    def _stub_init(self, model_name=None, **kwargs):
        self.model = None
        self.transform_image = None

    _rembg.BiRefNet.__init__ = _stub_init
    _rembg.BiRefNet._pixal3d_stubbed = True


def _resolve_attn_backend(backend: str) -> str:
    """Map 'auto' to the fastest installed backend in pixal3d's native dispatch.

    Probe order matches what ComfyUI does internally: flash_attn_3 > flash_attn >
    xformers > sdpa. (sageattention is not in pixal3d's native dispatch and would
    require an upstream patch.) Returns one of the pixal3d-recognized names.
    """
    if backend != "auto":
        return backend
    import importlib
    for name, mod in [
        ("flash_attn_3", "flash_attn_interface"),
        ("flash_attn", "flash_attn"),
        ("xformers", "xformers.ops"),
    ]:
        try:
            importlib.import_module(mod)
            log.info(f"[attn] auto-detect: probed {mod} -> using '{name}'")
            return name
        except ImportError:
            continue
    log.info("[attn] auto-detect: falling back to sdpa")
    return "sdpa"


def _set_attention_backends(backend: str):
    """Wire pixal3d's native dense + sparse attention dispatch.
    Resolves 'auto' by probing installed packages.
    """
    resolved = _resolve_attn_backend(backend)
    from .pixal3d.modules.attention.config import set_backend as set_dense
    from .pixal3d.modules.sparse.config import set_attn_backend as set_sparse
    set_dense(resolved)
    set_sparse(resolved)
    log.info(f"[attn] dense + sparse backend = {resolved} (requested: {backend})")


def _wrap_with_comfy_patcher(model):
    """Wrap an nn.Module in a ComfyUI ModelPatcher and reroute its `.to()` / `.cpu()`
    so pixal3d's per-stage swap goes through `load_models_gpu` / `unpatch_model`.

    The pixal3d pipeline.run() already calls `m.to(device)` before each stage and
    `m.cpu()` after; we just intercept those to inform ComfyUI's memory manager.
    Net effect: when ComfyUI loads another model (e.g. an upstream SDXL UNet),
    it knows our cascade models are evict-able, and vice versa.

    Idempotent per-instance.
    """
    import comfy.model_patcher

    if id(model) in _model_patchers:
        return _model_patchers[id(model)]

    load_device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()

    # Build the patcher; the model stays on whatever device it's currently on
    # (typically CPU after from_pretrained). Pixal3D's model classes now accept
    # ComfyUI's `instance.device = X` bookkeeping (via the @device.setter we
    # added in the vendored model files).
    patcher = comfy.model_patcher.ModelPatcher(
        model, load_device=load_device, offload_device=offload_device,
    )
    _model_patchers[id(model)] = patcher

    _orig_to = model.to
    _orig_cpu = model.cpu

    # Re-entry guard: ComfyUI's ModelPatcher.patch_model internally calls
    # model.to(device) -- which is OUR _patched_to. Without this guard we'd
    # recurse into load_models_gpu forever. With it, the inner call falls
    # through to _orig_to so the actual nn.Module move happens.
    import threading
    _reentry = threading.local()

    def _patched_to(*args, **kwargs):
        if getattr(_reentry, "inside", False):
            return _orig_to(*args, **kwargs)
        tgt = args[0] if args else kwargs.get("device")
        is_cuda_target = False
        if isinstance(tgt, torch.device):
            is_cuda_target = tgt.type == "cuda"
        elif isinstance(tgt, str):
            is_cuda_target = tgt.startswith("cuda")
        if is_cuda_target:
            _reentry.inside = True
            try:
                # Inform ComfyUI's memory manager (auto-offloads competing models).
                comfy.model_management.load_models_gpu([patcher])
                # ComfyUI's ModelPatcher.load doesn't always physically relocate
                # the wrapped module -- it manages cast/lowvram patches for its
                # own forward path. Pixal3D accesses .weight directly, so we
                # also call the real .to() to actually move tensors.
                _orig_to(*args, **kwargs)
            finally:
                _reentry.inside = False
            return model
        return _orig_to(*args, **kwargs)

    def _patched_cpu():
        if getattr(_reentry, "inside", False):
            return _orig_cpu()
        _reentry.inside = True
        try:
            patcher.unpatch_model(device_to=offload_device)
            # Same reason as above: physically move the module's tensors back.
            _orig_cpu()
            comfy.model_management.soft_empty_cache()
        finally:
            _reentry.inside = False
        return model

    model.to = _patched_to
    model.cpu = _patched_cpu
    return patcher


def _wrap_pipeline_models_with_patchers(pipeline):
    """Wrap every nn.Module the cascade swaps in/out: 8 cascade models +
    4 DinoV3 cond models + rembg. All start on CPU; load_models_gpu moves
    them to GPU on first `.to(device)` and offloads on `.cpu()`."""
    for key, m in pipeline.models.items():
        _wrap_with_comfy_patcher(m)
        log.info(f"[patcher] wrapped pipeline.models['{key}']")
    for attr in (
        "image_cond_model_ss",
        "image_cond_model_shape_512",
        "image_cond_model_shape_1024",
        "image_cond_model_tex_1024",
    ):
        m = getattr(pipeline, attr, None)
        if m is not None:
            _wrap_with_comfy_patcher(m)
            log.info(f"[patcher] wrapped pipeline.{attr}")
    # NOTE: pipeline.rembg_model is intentionally NOT wrapped. We never call it
    # (pipeline.run(preprocess_image=False) bypasses its preprocess_image), and
    # wrapping it caused a load_models_gpu <-> patched .to() recursion via the
    # BiRefNet wrapper. Background removal is the user's responsibility (LoadImage
    # MASK, or a community rembg node feeding Pixal3DPreprocessImage's mask input).


def init_pipeline(attn_backend: str = "auto") -> "object":
    """Load + cache Pixal3D pipeline + 4 DinoV3 cond models. Idempotent.

    The cascade always runs in per-stage swap mode (pixal3d's `low_vram=True`).
    This is the only mode that fits a 24 GB GPU; off-stage models are held on
    CPU which is the ComfyUI-native expectation. We don't expose a knob.
    """
    global _pipeline
    if _pipeline is not None:
        _set_attention_backends(attn_backend)
        return _pipeline

    _check_gpu_or_raise()

    with _phase("init_pipeline TOTAL"):
        with _phase("download Pixal3D weights"):
            local_dir = _download_pixal3d_weights()

        _stub_rembg()
        _patch_naf_to_local_model()
        _set_attention_backends(attn_backend)

        from .pixal3d.pipelines import Pixal3DImageTo3DPipeline
        from .pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
            DinoV3ProjFeatureExtractor,
        )

        with _phase("from_pretrained: 8 cascade safetensors -> CPU"):
            pipeline = Pixal3DImageTo3DPipeline.from_pretrained(str(local_dir))

        for key in ("ss", "shape_512", "shape_1024", "tex_1024"):
            with _phase(f"build DinoV3 cond '{key}'"):
                setattr(pipeline, f"image_cond_model_{key}", _build_cond(key))

        # Per-stage swap routed through ComfyUI's ModelPatcher / load_models_gpu.
        # pixal3d's pipeline.run() already calls `model.to(device)` / `model.cpu()`
        # between stages; we wrap each model so those calls go through ComfyUI's
        # memory manager (auto-offloads competing models, plays nice across nodes).
        pipeline.low_vram = True
        # CRITICAL: tell the pipeline its target device. Pixal3DImageTo3DPipeline.to()
        # with low_vram=True only sets `self._device` (no model movement -- good,
        # we want the per-stage swap to handle moves). Without this, self.device
        # stays 'cpu' from from_pretrained, get_proj_cond_ss reads it, and the
        # subsequent `image_cond_model.to(self.device)` is .to('cpu') -- a no-op.
        # The conv2d then fails with cuda-input vs cpu-weight.
        pipeline.to(comfy.model_management.get_torch_device())
        with _phase("ModelPatcher wrap: 13 models"):
            _wrap_pipeline_models_with_patchers(pipeline)

        with _phase("NAF: build singleton + attach to 3 cond models"):
            for attr in (
                "image_cond_model_ss",
                "image_cond_model_shape_512",
                "image_cond_model_shape_1024",
                "image_cond_model_tex_1024",
            ):
                m = getattr(pipeline, attr, None)
                if m is not None and getattr(m, "use_naf_upsample", False):
                    m._load_naf()

    _pipeline = pipeline
    return pipeline


def _build_cond(key: str):
    from .pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
        DinoV3ProjFeatureExtractor,
    )
    model = DinoV3ProjFeatureExtractor(**IMAGE_COND_CONFIGS[key])
    model.eval()
    # transformers >=5.0 moved DINOv3's transformer-layer ModuleList from
    # `DINOv3ViTModel.layer` to `DINOv3ViTModel.model.layer`. Pixal3D's
    # extract_features iterates `self.model.layer` directly — alias it back.
    inner = model.model
    if not hasattr(inner, "layer") and hasattr(inner, "model") and hasattr(inner.model, "layer"):
        inner.layer = inner.model.layer
    return model


# ============================================================================
# MoGe
# ============================================================================

def init_moge():
    global _moge_model
    if _moge_model is not None:
        return _moge_model

    _check_gpu_or_raise()
    with _phase("init_moge: MoGeModel.from_pretrained"):
        from moge.model.v2 import MoGeModel
        moge = MoGeModel.from_pretrained(MOGE_REPO).to(comfy.model_management.get_torch_device())
        moge.eval()
    _moge_model = moge
    return moge


# ============================================================================
# Image utils — ComfyUI IMAGE <-> PIL
# ============================================================================

def comfy_image_to_pil(image: torch.Tensor) -> Image.Image:
    """ComfyUI IMAGE is [B,H,W,C] float in [0,1]. Take batch 0, return RGB PIL."""
    if image.ndim == 4:
        image = image[0]
    arr = (image.detach().cpu().numpy().clip(0, 1) * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def pil_to_comfy_image(img: Image.Image) -> torch.Tensor:
    """PIL -> ComfyUI IMAGE [1,H,W,C] float [0,1]."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


# ============================================================================
# Preprocess
# ============================================================================

def preprocess_image(
    image: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    bg_color=(0, 0, 0),
) -> torch.Tensor:
    """Pure-PIL preprocess: alpha bbox crop + 1024-max resize + bg fill.

    Mirrors `pixal3d.pipelines.pixal3d_image_to_3d.Pixal3DImageTo3DPipeline.preprocess_image`
    minus the rembg call. Background removal is the user's responsibility:
      - If `mask` is provided (ComfyUI MASK, shape [B,H,W] or [H,W], 1.0=opaque,
        0.0=transparent), it's used as the alpha channel for bbox cropping.
      - If `mask` is absent, the image is assumed to already have a clean
        background (e.g. PNG loaded via LoadImage with a transparent BG, where
        ComfyUI's MASK output represents the alpha and would normally be wired in).
        We treat the entire image as the subject (no crop on alpha) and just resize.

    Args:
        image: ComfyUI IMAGE tensor, [B,H,W,3] float in [0,1].
        mask:  ComfyUI MASK tensor, [B,H,W] or [H,W] float in [0,1]. Optional.
        bg_color: RGB tuple in 0-255.

    Returns:
        ComfyUI IMAGE tensor, [1,H',W',3], H'==W'<=1024, subject centered.
    """
    # Image -> PIL RGB
    if image.ndim == 4:
        image = image[0]
    img_np = (image.detach().cpu().numpy().clip(0, 1) * 255.0).astype(np.uint8)
    pil_rgb = Image.fromarray(img_np, mode="RGB")

    # Optional mask -> PIL L
    pil_alpha = None
    if mask is not None:
        m = mask
        if m.ndim == 3:
            m = m[0]
        m_np = (m.detach().cpu().numpy().clip(0, 1) * 255.0).astype(np.uint8)
        pil_alpha = Image.fromarray(m_np, mode="L")
        # Sanity: mask spatial dims must match image. If not, resize mask to image.
        if pil_alpha.size != pil_rgb.size:
            pil_alpha = pil_alpha.resize(pil_rgb.size, Image.Resampling.NEAREST)

    # Downscale longest side to 1024 (same as upstream pixal3d preprocess_image).
    max_size = max(pil_rgb.size)
    scale = min(1.0, 1024 / max_size)
    if scale < 1.0:
        new_size = (int(pil_rgb.width * scale), int(pil_rgb.height * scale))
        pil_rgb = pil_rgb.resize(new_size, Image.Resampling.LANCZOS)
        if pil_alpha is not None:
            pil_alpha = pil_alpha.resize(new_size, Image.Resampling.LANCZOS)

    # If we have a meaningful mask (not all-white), crop to alpha bbox.
    if pil_alpha is not None:
        a = np.asarray(pil_alpha)
        if not np.all(a >= int(0.8 * 255)):
            bbox = np.argwhere(a > 0.8 * 255)
            if bbox.size > 0:
                x0, y0 = int(np.min(bbox[:, 1])), int(np.min(bbox[:, 0]))
                x1, y1 = int(np.max(bbox[:, 1])), int(np.max(bbox[:, 0]))
                cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                side = max(x1 - x0, y1 - y0)
                side = int(side * 1.1)
                half = side // 2
                bbox = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
                # Combine alpha into RGBA before cropping so out-of-frame pixels
                # become transparent (PIL .crop pads with implicit transparent).
                rgba = pil_rgb.convert("RGBA")
                rgba.putalpha(pil_alpha)
                rgba = rgba.crop(bbox)
                # Composite onto solid bg.
                arr = np.asarray(rgba).astype(np.float32) / 255.0
                rgb = arr[:, :, :3]
                a01 = arr[:, :, 3:4]
                bg = np.array(bg_color, dtype=np.float32) / 255.0
                composed = rgb * a01 + bg * (1.0 - a01)
                pil_rgb = Image.fromarray((np.clip(composed, 0, 1) * 255).astype(np.uint8))

    out_t = torch.from_numpy(np.asarray(pil_rgb.convert("RGB"), dtype=np.float32) / 255.0).unsqueeze(0)
    return out_t


# ============================================================================
# Camera estimation (refactored from inference.py:get_camera_params_wild_moge
# to take a tensor directly, skipping the temp-file roundtrip).
# ============================================================================

def _compute_f_pixels(camera_angle_x: float, resolution: int) -> float:
    focal = 16.0 / math.tan(camera_angle_x / 2.0)
    return float(focal * resolution / 32.0)


def _distance_from_fov(camera_angle_x, grid_point, target_point, mesh_scale, image_resolution):
    rot = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    gp = grid_point.to(torch.float32) @ rot.T
    gp = gp / mesh_scale / 2
    xw, yw, zw = gp[0].item(), gp[1].item(), gp[2].item()
    xt, yt = float(target_point[0].item()), float(target_point[1].item())
    f_pixels = _compute_f_pixels(camera_angle_x, image_resolution)
    x_ndc = xt - image_resolution / 2.0
    y_ndc = -(yt - image_resolution / 2.0)
    distance_x = f_pixels * xw / x_ndc - yw
    return float(distance_x)


def estimate_camera(
    image: torch.Tensor,
    mesh_scale: float = 1.0,
    extend_pixel: int = 0,
    image_resolution: int = 512,
) -> dict:
    moge = init_moge()

    pil = comfy_image_to_pil(image)
    width, height = pil.size
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).to(comfy.model_management.get_torch_device())

    with torch.no_grad():
        out = moge.infer(tensor)

    intrinsics = out["intrinsics"].squeeze().cpu().numpy()
    fx_normalized = float(intrinsics[0, 0])
    fx = fx_normalized * width
    camera_angle_x = 2.0 * math.atan(width / (2.0 * fx))

    grid_point = torch.tensor([-1.0, 0.0, 0.0])
    distance = _distance_from_fov(
        camera_angle_x,
        grid_point,
        torch.tensor([0 - extend_pixel, image_resolution - 1 + extend_pixel]),
        mesh_scale,
        image_resolution,
    )
    return {
        "camera_angle_x": float(camera_angle_x),
        "distance": float(distance),
        "mesh_scale": float(mesh_scale),
    }


# ============================================================================
# Cascade + GLB export, split into composable helpers.
#
# The cascade returns an internal-coords (Z-up, [-0.5, 0.5]^3) DC mesh + a
# sparse PBR voxel grid. From there two output paths:
#
#   1) Monolithic vertex-color path (generate_glb): light cleanup, query
#      per-vertex PBR from the voxel grid, write a vertex-colored GLB. Fast,
#      no UV unwrap, lower texture fidelity.
#   2) Split UV-bake path (generate_mesh_and_voxelgrid -> process_mesh ->
#      rasterize_pbr -> export_glb): heavy cleanup + UV unwrap + drtk UV-space
#      rasterize + BVH-snap + grid_sample_3d + cv2.inpaint + bake baseColor +
#      metallic/roughness textures. Matches upstream o_voxel.postprocess.to_glb
#      (which our o_voxel_vb_ap fork strips, since it depended on nvdiffrast).
#      Ports of TRELLIS2's Trellis2ProcessMesh / Trellis2RasterizePBR.
#
# IPC contract: TRIMESH = trimesh.Trimesh (CPU numpy). PIXAL3D_VOXELGRID = dict
# of numpy arrays / floats / dict-of-slices. Both pickle cleanly across the
# comfy-env boundary.
# ============================================================================


def _run_cascade(
    image: torch.Tensor,
    camera_params: dict,
    seed: int,
    pipeline_type: str,
    attn_backend: str,
    max_num_tokens: int,
    ss_steps: int, ss_guidance: float, ss_rescale: float, ss_rescale_t: float,
    shape_steps: int, shape_guidance: float, shape_rescale: float, shape_rescale_t: float,
    tex_steps: int, tex_guidance: float, tex_rescale: float, tex_rescale_t: float,
):
    """Run the 4-stage cascade. Returns (pipeline, MeshWithVoxel, resolution)."""
    pipeline = init_pipeline(attn_backend=attn_backend)
    pil = comfy_image_to_pil(image)
    torch.manual_seed(seed)
    log.info(f"[pixal3d] Running cascade ({pipeline_type}, seed={seed})")
    mesh_list, (shape_slat, tex_slat, res) = pipeline.run(
        pil,
        camera_params=camera_params,
        seed=seed,
        sparse_structure_sampler_params={
            "steps": ss_steps, "guidance_strength": ss_guidance,
            "guidance_rescale": ss_rescale, "rescale_t": ss_rescale_t,
        },
        shape_slat_sampler_params={
            "steps": shape_steps, "guidance_strength": shape_guidance,
            "guidance_rescale": shape_rescale, "rescale_t": shape_rescale_t,
        },
        tex_slat_sampler_params={
            "steps": tex_steps, "guidance_strength": tex_guidance,
            "guidance_rescale": tex_rescale, "rescale_t": tex_rescale_t,
        },
        preprocess_image=False,
        return_latent=True,
        pipeline_type=pipeline_type,
        max_num_tokens=max_num_tokens,
    )
    mw = mesh_list[0]
    log.info(f"[pixal3d] Mesh extracted at resolution {res}")
    del shape_slat, tex_slat, mesh_list
    return pipeline, mw, int(res)


def _meshwithvoxel_to_dict(mw, pipeline) -> dict:
    """Serialize a MeshWithVoxel's voxel side into an IPC-safe dict."""
    origin = mw.origin
    if hasattr(origin, "detach"):
        origin = origin.detach().cpu().tolist()
    else:
        origin = list(origin)
    # MeshWithVoxel.voxel_shape is torch.Size([1, C, X, Y, Z]) -- the full 5D shape
    # that grid_sample_3d expects directly. We store ONLY the 3D spatial extent here;
    # downstream consumers (_query_vertex_pbr, rasterize_pbr) rebuild the 5D shape as
    # torch.Size([1, attrs.shape[1], *voxel_shape]) at call time. Matches TRELLIS2's
    # 'grid_size' convention.
    full_shape = tuple(int(x) for x in mw.voxel_shape)
    spatial_shape = full_shape[-3:]
    return {
        "attrs": mw.attrs.detach().cpu().numpy(),                  # [N, C] float
        "coords": mw.coords.detach().cpu().numpy().astype(np.float32),  # [N, 3] voxel idx
        "voxel_size": float(mw.voxel_size),
        "voxel_shape": spatial_shape,                              # (X, Y, Z) only
        "origin": origin,
        "pbr_attr_layout": dict(pipeline.pbr_attr_layout),
    }


def _trimesh_from_meshwithvoxel(mw):
    """Wrap MeshWithVoxel's geometry in a plain CPU trimesh.Trimesh (no material)."""
    import trimesh as Trimesh
    return Trimesh.Trimesh(
        vertices=mw.vertices.detach().cpu().numpy().astype(np.float32),
        faces=mw.faces.detach().cpu().numpy().astype(np.int32),
        process=False,
    )


def _light_clean(tri, remove_inner_faces: bool = False):
    """In-place cumesh cleanup: dedup + repair_non_manifold + unify_face_orientations,
    plus optional BVH-raystab inner-face removal. Updates tri.vertices / tri.faces
    and returns the same trimesh."""
    import cumesh
    import trimesh as Trimesh
    device = comfy.model_management.get_torch_device()
    verts = torch.tensor(tri.vertices, dtype=torch.float32, device=device).contiguous()
    faces = torch.tensor(tri.faces, dtype=torch.int32, device=device).contiguous()

    with _phase("light_clean: dedup + repair + unify"):
        cm = cumesh.CuMesh()
        cm.init(verts, faces)
        cm.remove_duplicate_faces()
        cm.repair_non_manifold_edges()
        cm.unify_face_orientations()
        verts, faces = cm.read()
        del cm

    if remove_inner_faces:
        with _phase("light_clean: remove_inner_faces (BVH raystab)"):
            v = verts.float().contiguous()
            f = faces.int().contiguous()
            face_v = v[f.long()]
            face_centers = face_v.mean(dim=1)
            e1 = face_v[:, 1] - face_v[:, 0]
            e2 = face_v[:, 2] - face_v[:, 0]
            face_normals = torch.nn.functional.normalize(torch.cross(e1, e2, dim=-1), dim=-1)
            bbox_diag = float((v.amax(0) - v.amin(0)).norm())
            eps = bbox_diag * 1e-3
            test_pts = face_centers + face_normals * eps
            bvh = cumesh.cuBVH(v, f)
            sdf_chunk = 524_288
            sdf = torch.empty(test_pts.shape[0], dtype=torch.float32, device=device)
            for i in range(0, test_pts.shape[0], sdf_chunk):
                end = min(i + sdf_chunk, test_pts.shape[0])
                sdf[i:end] = bvh.signed_distance(test_pts[i:end], mode="raystab")[0]
            keep = sdf >= -eps * 0.1
            n_total = int(f.shape[0])
            n_removed = int((~keep).sum().item())
            log.info(f"[pixal3d] remove_inner_faces: dropped {n_removed}/{n_total} faces")
            kept_faces = f[keep]
            used = torch.zeros(v.shape[0], dtype=torch.bool, device=device)
            used[kept_faces.flatten().long()] = True
            old_to_new = torch.full((v.shape[0],), -1, dtype=torch.long, device=device)
            old_to_new[used] = torch.arange(int(used.sum()), device=device)
            verts = v[used]
            faces = old_to_new[kept_faces.long()].int()
            del bvh, face_v, face_centers, face_normals, e1, e2, test_pts, sdf, keep, used, old_to_new
            torch.cuda.empty_cache()

    cleaned = Trimesh.Trimesh(
        vertices=verts.detach().cpu().numpy().astype(np.float32),
        faces=faces.detach().cpu().numpy().astype(np.int32),
        process=False,
    )
    return cleaned


def _query_vertex_pbr(tri, voxelgrid: dict) -> np.ndarray:
    """Trilinearly sample the sparse PBR voxel grid at each mesh vertex.
    Returns [N, C] numpy float in [0, 1]."""
    from flex_gemm_ap.ops.grid_sample import grid_sample_3d
    device = comfy.model_management.get_torch_device()
    attrs = torch.from_numpy(voxelgrid["attrs"]).to(device)
    coords = torch.from_numpy(voxelgrid["coords"]).to(device)
    voxel_shape = voxelgrid["voxel_shape"]
    origin = torch.tensor(voxelgrid["origin"], dtype=torch.float32, device=device)
    voxel_size = float(voxelgrid["voxel_size"])
    verts = torch.tensor(tri.vertices, dtype=torch.float32, device=device)
    grid = ((verts - origin) / voxel_size).reshape(1, -1, 3)
    coords_padded = torch.cat([torch.zeros_like(coords[..., :1]), coords], dim=-1)
    vattrs = grid_sample_3d(
        attrs,
        coords_padded,
        torch.Size([1, attrs.shape[1], *voxel_shape]),
        grid,
        mode="trilinear",
    )[0]
    return vattrs.clamp(0.0, 1.0).detach().cpu().numpy()


def _bake_vertex_colors(tri, voxelgrid: dict, force_opaque: bool, double_sided: bool):
    """Attach vertex colors (+ optional PBRMaterial) to tri. Returns the updated mesh."""
    import trimesh as Trimesh
    layout = voxelgrid["pbr_attr_layout"]
    vattrs = _query_vertex_pbr(tri, voxelgrid)
    rgb = (vattrs[:, layout.get("base_color", slice(0, 3))] * 255).astype(np.uint8)
    alpha_slice = layout.get("alpha", None)

    material = None
    if force_opaque:
        vertex_colors = rgb
        if double_sided:
            material = Trimesh.visual.material.PBRMaterial(
                name="pixal3d_opaque_double_sided",
                alphaMode="OPAQUE", doubleSided=True,
                metallicFactor=0.0, roughnessFactor=1.0,
            )
    elif alpha_slice is not None:
        a = (vattrs[:, alpha_slice] * 255).astype(np.uint8)
        if a.ndim == 2 and a.shape[1] == 1:
            a = a[:, 0]
        vertex_colors = np.concatenate([rgb, a[:, None]], axis=1)
        material = Trimesh.visual.material.PBRMaterial(
            name="pixal3d_translucent",
            alphaMode="BLEND", doubleSided=double_sided,
            metallicFactor=0.0, roughnessFactor=1.0,
        )
    else:
        vertex_colors = rgb
        if double_sided:
            material = Trimesh.visual.material.PBRMaterial(
                name="pixal3d_opaque_double_sided",
                alphaMode="OPAQUE", doubleSided=True,
                metallicFactor=0.0, roughnessFactor=1.0,
            )

    out = Trimesh.Trimesh(
        vertices=tri.vertices, faces=tri.faces,
        vertex_colors=vertex_colors,
        process=False,
    )
    if material is not None:
        out.visual.material = material
    return out


# ----------------------------------------------------------------------------
# UV-bake path: ports of TRELLIS2's Trellis2ProcessMesh / Trellis2RasterizePBR.
# ----------------------------------------------------------------------------


def _dbg(msg: str) -> None:
    """Single-line stderr-flushed debug print, prefixed for grep-ability."""
    print(f"[pixal3d:dbg] {msg}", file=sys.stderr, flush=True)


def _log_mesh_stats(label: str, cm) -> None:
    """Print cumesh state after a step. Cheap (just reads counters)."""
    try:
        _dbg(f"  {label}: {cm.num_vertices} verts / {cm.num_faces} faces")
    except Exception as e:
        _dbg(f"  {label}: could not read cumesh stats ({e})")


def _rasterize_uv(vertices, faces, uvs, texture_size, device, debug=False):
    """drtk port of nvdiffrast's UV-space rasterize+interpolate.
    Returns (mask: [S,S] bool, valid_pos: [N, 3] float, face_ids: [N] long,
             bary_masked: [N, 3] float, rast_face_ids: [S, S] int32) -- the
    last entry is the per-texel face id image, preserved for debug dumps.
    Mirrors trellis2/nodes_unwrap.py:1026."""
    import drtk
    chunk_size = 100_000
    S = int(texture_size)
    if debug:
        _dbg(f"_rasterize_uv: V={uvs.shape[0]}, F={faces.shape[0]}, S={S}")
        _dbg(f"  UV bbox: min={uvs.amin(dim=0).tolist()}, max={uvs.amax(dim=0).tolist()}")
        _dbg(f"  vert bbox: min={vertices.amin(dim=0).tolist()}, max={vertices.amax(dim=0).tolist()}")
    verts_uv = torch.stack([
        uvs[:, 0] * S - 0.5,
        uvs[:, 1] * S - 0.5,
        torch.ones(uvs.shape[0], device=device),
    ], dim=-1).float().unsqueeze(0)  # [1, V, 3]

    rast_face_ids = torch.full((S, S), -1, dtype=torch.int32, device=device)
    for i in range(0, faces.shape[0], chunk_size):
        comfy.model_management.throw_exception_if_processing_interrupted()
        chunk_vi = faces[i:i + chunk_size].int()
        index_img = drtk.rasterize(verts_uv, chunk_vi, height=S, width=S)
        chunk_hit = index_img[0] >= 0
        rast_face_ids[chunk_hit] = (index_img[0][chunk_hit] + i).int()
        del index_img, chunk_hit

    mask = rast_face_ids >= 0
    _, bary_img = drtk.render(verts_uv, faces.int(), rast_face_ids.unsqueeze(0))
    bary = bary_img[0].permute(1, 2, 0)
    bary_masked = bary[mask]
    face_ids = rast_face_ids[mask].long()
    face_verts = vertices[faces[face_ids].long()]
    valid_pos = (face_verts * bary_masked.unsqueeze(-1)).sum(dim=1)

    if debug:
        cov = mask.sum().item() / mask.numel()
        n_distinct = int(torch.unique(rast_face_ids[mask]).numel()) if mask.any() else 0
        _dbg(f"  rasterized: coverage={cov:.1%}, distinct face_ids={n_distinct}/{faces.shape[0]}")
        _dbg(f"  bary range: [{bary_masked.min().item():.4f}, {bary_masked.max().item():.4f}]")
        if valid_pos.numel():
            _dbg(f"  valid_pos bbox: min={valid_pos.amin(dim=0).tolist()}, max={valid_pos.amax(dim=0).tolist()}")

    # Keep rast_face_ids alive for caller if debug dump is on.
    rast_face_ids_out = rast_face_ids.clone() if debug else None
    del verts_uv, rast_face_ids, bary_img, bary, face_verts
    comfy.model_management.soft_empty_cache()
    return mask, valid_pos, face_ids, bary_masked, rast_face_ids_out


def process_mesh(
    tri,
    remesh: bool = False,
    remesh_band: float = 1.0,
    remesh_resolution: int = 512,
    fill_holes: bool = True,
    fill_holes_perimeter: float = 0.03,
    floater_threshold: float = 1e-3,
    target_face_count: int = 200_000,
    remove_inner_faces: bool = False,
    weld_vertices: bool = True,
    weld_digits: int = 4,
    chart_cone_angle: float = 90.0,
    chart_refine_iterations: int = 0,
    chart_global_iterations: int = 1,
    chart_smooth_strength: int = 1,
):
    """Heavy mesh cleanup + UV unwrap. Port of TRELLIS2 Trellis2ProcessMesh.execute().
    Returns a trimesh.Trimesh with .visual.uv set and vertex_normals populated."""
    import cumesh as CuMesh
    import trimesh as Trimesh

    device = comfy.model_management.get_torch_device()
    _dbg(f"process_mesh: in {len(tri.vertices)} verts / {len(tri.faces)} faces, target {target_face_count}")
    in_v = np.asarray(tri.vertices)
    _dbg(f"  vert bbox: min={in_v.min(axis=0).tolist()}, max={in_v.max(axis=0).tolist()}")

    verts = torch.tensor(tri.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(tri.faces, dtype=torch.int32, device=device)

    with _phase("process_mesh: cumesh.init"):
        cm = CuMesh.CuMesh()
        cm.init(verts, faces)
        del verts, faces
    _log_mesh_stats("after init", cm)

    if remesh:
        with _phase("process_mesh: DC remesh"):
            curr_v, curr_f = cm.read()
            aabb = torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], device=device)
            center = aabb.mean(dim=0)
            scale = (aabb[1] - aabb[0]).max().item()
            cm.init(*CuMesh.remeshing.remesh_narrow_band_dc_quad(
                curr_v, curr_f,
                center=center,
                scale=scale * 1.1,
                resolution=remesh_resolution,
                band=remesh_band,
                project_back=0.0,
                verbose=True,
                remove_inner_faces=remove_inner_faces,
            ))
            del curr_v, curr_f
        _log_mesh_stats("after DC remesh", cm)

    if floater_threshold > 0:
        with _phase(f"process_mesh: remove_small_connected_components({floater_threshold})"):
            cm.remove_small_connected_components(floater_threshold)
        _log_mesh_stats(f"after remove_small_cc({floater_threshold})", cm)

    comfy.model_management.throw_exception_if_processing_interrupted()

    if not remesh:
        # 2-pass simplify pattern from upstream to_glb.
        with _phase("process_mesh: 2-pass simplify + cleanup"):
            cm.remove_duplicate_faces()
            cm.repair_non_manifold_edges()
            if floater_threshold > 0:
                cm.remove_small_connected_components(floater_threshold)
            if fill_holes:
                cm.fill_holes(max_hole_perimeter=fill_holes_perimeter)
            _log_mesh_stats("cleanup pass 1", cm)
            cm.simplify(target_face_count * 3, verbose=True)
            _log_mesh_stats(f"after simplify({target_face_count * 3})", cm)
            cm.remove_duplicate_faces()
            cm.repair_non_manifold_edges()
            if floater_threshold > 0:
                cm.remove_small_connected_components(floater_threshold)
            if fill_holes:
                cm.fill_holes(max_hole_perimeter=fill_holes_perimeter)
            cm.simplify(target_face_count, verbose=True)
            _log_mesh_stats(f"after simplify({target_face_count})", cm)
            cm.remove_duplicate_faces()
            cm.repair_non_manifold_edges()
            if floater_threshold > 0:
                cm.remove_small_connected_components(floater_threshold)
            if fill_holes:
                cm.fill_holes(max_hole_perimeter=fill_holes_perimeter)
            cm.unify_face_orientations()
            _log_mesh_stats("after unify_face_orientations", cm)
    else:
        with _phase("process_mesh: simplify (post-remesh)"):
            cm.simplify(target_face_count, verbose=True)
        _log_mesh_stats(f"after simplify({target_face_count})", cm)

    comfy.model_management.throw_exception_if_processing_interrupted()

    if weld_vertices:
        with _phase(f"process_mesh: weld_vertices(digits={weld_digits})"):
            wv, wf = cm.read()
            wm = Trimesh.Trimesh(
                vertices=wv.cpu().numpy(), faces=wf.cpu().numpy(), process=False,
            )
            pre = len(wm.vertices)
            wm.merge_vertices(digits_vertex=weld_digits)
            wm.remove_unreferenced_vertices()
            wm.update_faces(wm.nondegenerate_faces())
            _dbg(f"  weld: {pre} -> {len(wm.vertices)} verts (digits={weld_digits})")
            cm.init(
                torch.tensor(wm.vertices, dtype=torch.float32, device=device),
                torch.tensor(wm.faces, dtype=torch.int32, device=device),
            )
            del wv, wf, wm
        _log_mesh_stats("after weld", cm)

    with _phase("process_mesh: uv_unwrap (xatlas)"):
        _log_mesh_stats("pre-uv_unwrap", cm)
        out_v, out_f, out_uvs, out_vmaps = cm.uv_unwrap(
            compute_charts_kwargs={
                "threshold_cone_half_angle_rad": float(np.radians(chart_cone_angle)),
                "refine_iterations": int(chart_refine_iterations),
                "global_iterations": int(chart_global_iterations),
                "smooth_strength": int(chart_smooth_strength),
            },
            return_vmaps=True,
            verbose=True,
        )
        cm.compute_vertex_normals()
        out_normals = cm.read_vertex_normals()[out_vmaps.to(device)].cpu().numpy()
        # Per-vertex-split count (out_v >= pre-unwrap verts, due to seams).
        uvs_np = out_uvs.cpu().numpy() if hasattr(out_uvs, "cpu") else np.asarray(out_uvs)
        _dbg(
            f"  uv_unwrap out: {out_v.shape[0]} verts (split-at-seams) / "
            f"{out_f.shape[0]} faces; UV bbox=[{uvs_np.min(axis=0).tolist()}, "
            f"{uvs_np.max(axis=0).tolist()}]; pre-unwrap verts={int(out_vmaps.max().item()) + 1}"
        )

    result = Trimesh.Trimesh(
        vertices=out_v.cpu().numpy(),
        faces=out_f.cpu().numpy(),
        vertex_normals=out_normals,
        process=False,
    )
    result.visual = Trimesh.visual.TextureVisuals(uv=out_uvs.cpu().numpy())
    _dbg(f"process_mesh: OUT {len(result.vertices)} verts / {len(result.faces)} faces (UVs ready)")

    del cm, out_v, out_f, out_uvs, out_vmaps
    gc.collect()
    comfy.model_management.soft_empty_cache()
    return result


def rasterize_pbr(
    tri,
    voxelgrid: dict,
    texture_size: int = 2048,
    original_mesh=None,
    double_sided: bool = False,
    bake_mode: str = "pbr",
    debug_dump: bool = False,
):
    """drtk UV-space PBR bake. Port of TRELLIS2 Trellis2RasterizePBR.execute().
    Returns a trimesh.Trimesh with a PBRMaterial(baseColorTexture, metallicRoughnessTexture).

    bake_mode:
      'pbr'           -- bake baseColor + metallic/roughness/alpha from the voxel grid (production).
      'xyz_position'  -- diagnostic: paint texels with (x, y, z) position as RGB.
                         Maps the mesh's AABB into [0, 1]^3 so red=+X, green=+Y, blue=+Z.
                         If the mesh is in Z-up working frame (post-GenerateMesh rotation),
                         blue=Z will be the "up" axis on the rendered model.
      'xyz_normal'    -- diagnostic: paint texels with the interpolated surface normal
                         as RGB. Normals in [-1, 1] are mapped to [0, 1]. Lets you
                         visually verify face winding / unify_face_orientations.

    debug_dump: when True, prints extra stats to stderr AND saves diagnostic PNG/OBJ
                files to ComfyUI/output/ (prefixed pixal3d_debug_<ts>_). Use to
                inspect chart layout / mesh state when textures look wrong."""
    import cv2
    import cumesh as CuMesh
    from flex_gemm_ap.ops.grid_sample import grid_sample_3d
    import trimesh as Trimesh

    if not hasattr(tri.visual, "uv") or tri.visual.uv is None:
        raise ValueError("rasterize_pbr: input trimesh has no UVs. Wire ProcessMesh first.")
    if "attrs" not in voxelgrid:
        raise ValueError("rasterize_pbr: voxelgrid dict is missing 'attrs'.")

    device = comfy.model_management.get_torch_device()
    _dbg(
        f"rasterize_pbr: {len(tri.vertices)} verts / {len(tri.faces)} faces, "
        f"texture {texture_size}px, bake_mode={bake_mode}, "
        f"original_mesh={'yes' if original_mesh is not None else 'no'}, "
        f"debug_dump={debug_dump}"
    )

    vertices = torch.tensor(tri.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(tri.faces, dtype=torch.int32, device=device)
    uvs = torch.tensor(tri.visual.uv, dtype=torch.float32, device=device)

    attr_volume = torch.from_numpy(voxelgrid["attrs"]).to(device)
    coords = torch.from_numpy(voxelgrid["coords"]).to(device)
    voxel_size_v = float(voxelgrid["voxel_size"])
    origin = torch.tensor(voxelgrid.get("origin", [-0.5, -0.5, -0.5]), dtype=torch.float32, device=device)
    voxel_shape = voxelgrid.get("voxel_shape")
    layout = voxelgrid.get("pbr_attr_layout", {
        "base_color": slice(0, 3), "metallic": slice(3, 4),
        "roughness": slice(4, 5), "alpha": slice(5, 6),
    })
    aabb_min = origin
    aabb_max = origin + torch.tensor([1.0, 1.0, 1.0], device=device)

    if voxel_shape is None:
        grid_size = ((aabb_max - aabb_min) / voxel_size_v).round().int()
        voxel_shape = tuple(int(x) for x in grid_size.tolist())
    voxel_size_t = torch.tensor([voxel_size_v] * 3, dtype=torch.float32, device=device)

    with _phase("rasterize_pbr: drtk UV rasterize"):
        mask, valid_pos, face_ids, bary_masked, rast_face_ids_dbg = _rasterize_uv(
            vertices, faces, uvs, texture_size, device, debug=debug_dump,
        )

    # Disk dumps (one-shot). Save mask/face_ids PNGs + OBJ for offline inspection.
    if debug_dump:
        with _phase("rasterize_pbr: debug dump (mask/face_ids/obj)"):
            ts_dbg = int(time.time() * 1000)
            out_dir_dbg = folder_paths.get_output_directory()
            prefix = os.path.join(out_dir_dbg, f"pixal3d_debug_{ts_dbg}")
            try:
                # Mask PNG: 255 where rasterized, 0 elsewhere.
                mask_img = (mask.to(torch.uint8) * 255).cpu().numpy()
                Image.fromarray(mask_img, mode="L").save(f"{prefix}_mask.png")
                # Face-id PNG: mod 256 to keep it in uint8 range. Tiny isolated
                # patches in this image indicate tiny UV charts.
                if rast_face_ids_dbg is not None:
                    fid = rast_face_ids_dbg.clone()
                    fid_vis = torch.where(fid >= 0, fid % 256, torch.zeros_like(fid)).to(torch.uint8).cpu().numpy()
                    Image.fromarray(fid_vis, mode="L").save(f"{prefix}_face_ids.png")
                    del fid, fid_vis
                # Mesh OBJ (vertices + faces + UVs). Lets you inspect mesh in Blender.
                import trimesh as _Trimesh
                obj_mesh = _Trimesh.Trimesh(
                    vertices=tri.vertices, faces=tri.faces, process=False,
                    visual=_Trimesh.visual.TextureVisuals(uv=tri.visual.uv),
                )
                obj_mesh.export(f"{prefix}_mesh.obj")
                del obj_mesh
                _dbg(f"  wrote {prefix}_{{mask.png, face_ids.png, mesh.obj}}")
            except Exception as e:
                _dbg(f"  debug dump failed: {e}")
            del rast_face_ids_dbg

    if original_mesh is not None:
        with _phase("rasterize_pbr: BVH-snap texels to original mesh"):
            orig_v = torch.tensor(original_mesh.vertices, dtype=torch.float32, device=device)
            orig_f = torch.tensor(original_mesh.faces, dtype=torch.int32, device=device)
            bvh = CuMesh.cuBVH(orig_v, orig_f)
            _, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)
            orig_tri_v = orig_v[orig_f[face_id.long()]]
            valid_pos = (orig_tri_v * uvw.unsqueeze(-1)).sum(dim=1)
            del bvh, orig_v, orig_f, face_id, uvw, orig_tri_v

    comfy.model_management.soft_empty_cache()
    mask_np = mask.cpu().numpy()

    # --------------------------------------------------------------------
    # Diagnostic bake modes: paint texels with mesh-frame XYZ position or
    # interpolated surface normal as RGB. Skip the voxel sample entirely.
    # --------------------------------------------------------------------
    if bake_mode in ("xyz_position", "xyz_normal"):
        with _phase(f"rasterize_pbr: {bake_mode} bake"):
            if bake_mode == "xyz_position":
                # Map mesh-frame valid_pos AABB into [0, 1]^3 so red=+X axis,
                # green=+Y axis, blue=+Z axis on the textured model.
                v_min = valid_pos.amin(dim=0)
                v_max = valid_pos.amax(dim=0)
                span = (v_max - v_min).clamp(min=1e-6)
                rgb_vals = ((valid_pos - v_min) / span).clamp(0.0, 1.0)
                log.info(
                    f"[pixal3d] xyz_position: AABB min={v_min.tolist()}, "
                    f"max={v_max.tolist()} (mapped to [0,1] RGB)"
                )
            else:  # xyz_normal
                if not hasattr(tri, "vertex_normals") or tri.vertex_normals is None:
                    raise ValueError(
                        "rasterize_pbr xyz_normal: input mesh has no vertex_normals. "
                        "Pixal3DProcessMesh should populate them."
                    )
                vnorm = torch.tensor(np.asarray(tri.vertex_normals), dtype=torch.float32, device=device)
                per_vert_normals = vnorm[faces[face_ids].long()]  # [N_texel, 3, 3]
                texel_normals = (per_vert_normals * bary_masked.unsqueeze(-1)).sum(dim=1)
                texel_normals = torch.nn.functional.normalize(texel_normals, dim=-1)
                rgb_vals = (texel_normals * 0.5 + 0.5).clamp(0.0, 1.0)
                log.info("[pixal3d] xyz_normal: per-texel normal mapped from [-1,1] to [0,1] RGB")

            rgb_img = torch.zeros(texture_size, texture_size, 3, device=device)
            rgb_img[mask] = rgb_vals
            base_color = np.clip(rgb_img.cpu().numpy() * 255, 0, 255).astype(np.uint8)
            del rgb_img, rgb_vals

        with _phase("rasterize_pbr: cv2.inpaint (UV seam pad, diagnostic)"):
            mask_inv = (~mask_np).astype(np.uint8)
            base_color = cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA)

        # Build a fully opaque, unlit-ish material so the colors read clean.
        alpha = np.full((texture_size, texture_size, 1), 255, dtype=np.uint8)
        material = Trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(np.concatenate([base_color, alpha], axis=-1)),
            baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
            metallicFactor=0.0,
            roughnessFactor=1.0,
            alphaMode="OPAQUE",
            doubleSided=double_sided,
        )
        result = Trimesh.Trimesh(
            vertices=tri.vertices,
            faces=tri.faces,
            vertex_normals=tri.vertex_normals if hasattr(tri, "vertex_normals") else None,
            process=False,
            visual=Trimesh.visual.TextureVisuals(uv=tri.visual.uv, material=material),
        )
        log.info(f"[pixal3d] rasterize_pbr: {bake_mode} diagnostic texture baked ({texture_size}x{texture_size})")
        del attr_volume, coords, mask, valid_pos, face_ids, bary_masked
        gc.collect()
        comfy.model_management.soft_empty_cache()
        return result

    # --------------------------------------------------------------------
    # Production PBR bake.
    # --------------------------------------------------------------------
    # Mesh + valid_pos are in Z-up (rotated by Pixal3DGenerateMesh to match
    # TRELLIS2's verified-working cumesh/drtk regime). The voxelgrid is left
    # in cascade-native Y-up. Rotate valid_pos Z-up -> Y-up via column swap
    # so it lines up with the voxelgrid's frame.
    valid_pos_yup = torch.stack(
        [valid_pos[:, 0], valid_pos[:, 2], -valid_pos[:, 1]],
        dim=-1,
    )

    with _phase("rasterize_pbr: grid_sample_3d (texels)"):
        attrs = torch.zeros(texture_size, texture_size, attr_volume.shape[1], device=device)
        coords_padded = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1)
        attrs[mask] = grid_sample_3d(
            attr_volume,
            coords_padded,
            shape=torch.Size([1, attr_volume.shape[1], *voxel_shape]),
            grid=((valid_pos_yup - aabb_min) / voxel_size_t).reshape(1, -1, 3),
            mode="trilinear",
        )

    del valid_pos, valid_pos_yup, face_ids, bary_masked
    comfy.model_management.soft_empty_cache()

    bc_slice = layout.get("base_color", slice(0, 3))
    me_slice = layout.get("metallic", slice(3, 4))
    ro_slice = layout.get("roughness", slice(4, 5))
    al_slice = layout.get("alpha", slice(5, 6))

    def _channel(np_slice, channels):
        out = np.clip(attrs[..., np_slice].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        # cv2.inpaint wants HxW or HxWxC contiguous uint8; for 1-channel keep [..., None].
        return out if out.ndim == 3 and out.shape[-1] == channels else out[..., None]

    base_color = _channel(bc_slice, 3)
    metallic = _channel(me_slice, 1)
    roughness = _channel(ro_slice, 1)
    alpha = _channel(al_slice, 1)

    del attrs, mask, attr_volume, coords
    gc.collect()
    comfy.model_management.soft_empty_cache()

    with _phase("rasterize_pbr: cv2.inpaint (UV seam pad)"):
        mask_inv = (~mask_np).astype(np.uint8)
        base_color = cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA)
        metallic = cv2.inpaint(metallic, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
        roughness = cv2.inpaint(roughness, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
        alpha = cv2.inpaint(alpha, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]

    material = Trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(np.concatenate([base_color, alpha], axis=-1)),
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=Image.fromarray(np.concatenate([
            np.zeros_like(metallic), roughness, metallic,
        ], axis=-1)),
        metallicFactor=1.0,
        roughnessFactor=1.0,
        alphaMode="OPAQUE",
        doubleSided=double_sided,
    )

    result = Trimesh.Trimesh(
        vertices=tri.vertices,
        faces=tri.faces,
        vertex_normals=tri.vertex_normals if hasattr(tri, "vertex_normals") else None,
        process=False,
        visual=Trimesh.visual.TextureVisuals(uv=tri.visual.uv, material=material),
    )
    log.info(f"[pixal3d] rasterize_pbr: {texture_size}x{texture_size} PBR textures baked")
    return result


# ----------------------------------------------------------------------------
# Coordinate frame rotations.
#
# Pixal3D's cascade emits MeshWithVoxel.vertices in Y-up natively (the model
# is camera-conditioned via MoGe-2 / DinoV3, training data is Y-up). TRELLIS2's
# cascade is Z-up. The cumesh/drtk UV-bake regime in TRELLIS2 is visually
# verified working on Z-up data; on Pixal3D's Y-up mesh the same code produces
# garbled, seam-bleed textures. Fix: route the split-node UV-bake path through
# Z-up internally (rotate at GenerateMesh, rotate `valid_pos` back for the
# voxel sample, rotate back to Y-up at ExportGLB). Monolithic vertex-color
# path stays Y-up throughout.
# ----------------------------------------------------------------------------

_YUP_TO_ZUP_ROT = np.array(
    # (x, y, z) -> (x, -z, y). Inverse of _ZUP_TO_YUP_ROT.
    [[1, 0,  0, 0],
     [0, 0, -1, 0],
     [0, 1,  0, 0],
     [0, 0,  0, 1]],
    dtype=np.float64,
)

_ZUP_TO_YUP_ROT = np.array(
    # (x, y, z) -> (x, z, -y). Matches TRELLIS2 nodes_unwrap.py:1325 verbatim
    # assignment: vertices[:, 1], vertices[:, 2] = vertices[:, 2], -vertices[:, 1].
    [[1,  0, 0, 0],
     [0,  0, 1, 0],
     [0, -1, 0, 0],
     [0,  0, 0, 1]],
    dtype=np.float64,
)


def export_glb(tri, filename_prefix: str = "pixal3d") -> str:
    """Write the trimesh to ComfyUI's output dir as a GLB and return the absolute path.

    No coordinate rotation is applied. Used by the monolithic vertex-color path
    where the mesh stays Y-up throughout (Pixal3D's cascade-native frame).
    For the split-node UV-bake path use export_glb_yup, which un-rotates the
    Z-up working frame back to Y-up for the final GLB."""
    out_dir = folder_paths.get_output_directory()
    ts = int(time.time() * 1000)
    out_path = os.path.join(out_dir, f"{filename_prefix}_{ts}.glb")
    tri.export(out_path)
    log.info(f"[pixal3d] Saved GLB to {out_path}")
    return out_path


def export_glb_yup(tri, filename_prefix: str = "pixal3d") -> str:
    """Apply Z-up -> Y-up rotation + UV V flip, then write to ComfyUI's output dir.

    Mirrors TRELLIS2 Trellis2ExportTrimesh.execute() lines 1397-1411 EXACTLY:
      1. Rotate vertices  (x, y, z) -> (x, z, -y)   (Z-up to Y-up).
      2. Rotate normals   (x, y, z) -> (x, z, -y)   (same swap; we apply it
         manually instead of via trimesh.apply_transform because the latter
         invalidates user-supplied vertex_normals and forces a recompute).
      3. Flip UV V        v -> 1 - v                (CRITICAL: glTF's V
         convention is inverted relative to cumesh/drtk's UV output.
         Without this flip the rasterized texture appears V-mirrored per
         chart on the rendered model, which combined with chart packing
         looks garbled / discontinuous).

    Deepcopy first so we don't mutate the caller's trimesh (matches TRELLIS2)."""
    import copy as _copy
    export_mesh = _copy.deepcopy(tri)

    verts = export_mesh.vertices.copy()
    verts[:, 1], verts[:, 2] = verts[:, 2].copy(), -verts[:, 1].copy()
    export_mesh.vertices = verts

    if (hasattr(export_mesh, "vertex_normals")
            and export_mesh.vertex_normals is not None
            and len(export_mesh.vertex_normals) > 0):
        normals = export_mesh.vertex_normals.copy()
        normals[:, 1], normals[:, 2] = normals[:, 2].copy(), -normals[:, 1].copy()
        export_mesh.vertex_normals = normals

    if hasattr(export_mesh.visual, "uv") and export_mesh.visual.uv is not None:
        uvs = export_mesh.visual.uv.copy()
        uvs[:, 1] = 1 - uvs[:, 1]
        export_mesh.visual.uv = uvs

    return export_glb(export_mesh, filename_prefix=filename_prefix)


# ----------------------------------------------------------------------------
# Convenience: cascade + mesh + voxelgrid, as a single helper exposed to nodes.
# ----------------------------------------------------------------------------


def generate_mesh_and_voxelgrid(
    image: torch.Tensor,
    camera_params: dict,
    seed: int = 42,
    pipeline_type: str = "1024_cascade",
    attn_backend: str = "auto",
    max_num_tokens: int = 49152,
    ss_steps: int = 12, ss_guidance: float = 7.5, ss_rescale: float = 0.7, ss_rescale_t: float = 5.0,
    shape_steps: int = 12, shape_guidance: float = 7.5, shape_rescale: float = 0.5, shape_rescale_t: float = 3.0,
    tex_steps: int = 12, tex_guidance: float = 1.0, tex_rescale: float = 0.0, tex_rescale_t: float = 3.0,
):
    """Run the cascade and split the result into IPC-safe (TRIMESH, PIXAL3D_VOXELGRID).
    The trimesh is the raw DC mesh in pixal3d internal coords ([-0.5, 0.5]^3, Z-up)."""
    pipeline, mw, _res = _run_cascade(
        image, camera_params, seed, pipeline_type, attn_backend, max_num_tokens,
        ss_steps, ss_guidance, ss_rescale, ss_rescale_t,
        shape_steps, shape_guidance, shape_rescale, shape_rescale_t,
        tex_steps, tex_guidance, tex_rescale, tex_rescale_t,
    )
    tri = _trimesh_from_meshwithvoxel(mw)
    voxelgrid = _meshwithvoxel_to_dict(mw, pipeline)
    del mw
    gc.collect()
    torch.cuda.empty_cache()
    return tri, voxelgrid


# ----------------------------------------------------------------------------
# Monolithic vertex-color path. Backwards-compatible Pixal3DGenerateGLB body.
# ----------------------------------------------------------------------------


def generate_glb(
    image: torch.Tensor,
    camera_params: dict,
    seed: int = 42,
    pipeline_type: str = "1024_cascade",
    attn_backend: str = "auto",
    max_num_tokens: int = 49152,
    ss_steps: int = 12, ss_guidance: float = 7.5, ss_rescale: float = 0.7, ss_rescale_t: float = 5.0,
    shape_steps: int = 12, shape_guidance: float = 7.5, shape_rescale: float = 0.5, shape_rescale_t: float = 3.0,
    tex_steps: int = 12, tex_guidance: float = 1.0, tex_rescale: float = 0.0, tex_rescale_t: float = 3.0,
    decimation_target: int = 200000,
    texture_size: int = 2048,
    pre_simplify: bool = True,
    pre_simplify_target_faces: int = 2_000_000,
    force_opaque: bool = True,
    double_sided: bool = False,
    remove_inner_faces: bool = False,
    filename_prefix: str = "pixal3d",
) -> str:
    """Cascade -> light cleanup -> vertex-color bake -> GLB. The monolithic convenience
    path: no UV unwrap, no texture map, fast. For UV-baked output use the split node
    chain (GenerateMesh -> ProcessMesh -> RasterizePBR -> ExportGLB)."""
    tri, voxelgrid = generate_mesh_and_voxelgrid(
        image, camera_params, seed, pipeline_type, attn_backend, max_num_tokens,
        ss_steps, ss_guidance, ss_rescale, ss_rescale_t,
        shape_steps, shape_guidance, shape_rescale, shape_rescale_t,
        tex_steps, tex_guidance, tex_rescale, tex_rescale_t,
    )
    cleaned = _light_clean(tri, remove_inner_faces=remove_inner_faces)
    colored = _bake_vertex_colors(cleaned, voxelgrid, force_opaque=force_opaque, double_sided=double_sided)
    out_path = export_glb(colored, filename_prefix=filename_prefix)
    del tri, cleaned, colored, voxelgrid
    gc.collect()
    torch.cuda.empty_cache()

    return out_path
