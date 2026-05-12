"""Pixal3D pipeline stages.

Module-level caches: first call loads from disk, subsequent calls reuse from RAM.
All heavy work happens inside the isolated comfy-env subprocess.
"""

import gc
import logging
import math
import os
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
    # (typically CPU after from_pretrained).
    patcher = comfy.model_patcher.ModelPatcher(
        model, load_device=load_device, offload_device=offload_device,
    )
    _model_patchers[id(model)] = patcher

    _orig_to = model.to
    _orig_cpu = model.cpu

    def _patched_to(*args, **kwargs):
        # Detect a request to move to the active CUDA device (pixal3d's swap-in).
        tgt = args[0] if args else kwargs.get("device")
        is_cuda_target = False
        if isinstance(tgt, torch.device):
            is_cuda_target = tgt.type == "cuda"
        elif isinstance(tgt, str):
            is_cuda_target = tgt.startswith("cuda")
        if is_cuda_target:
            comfy.model_management.load_models_gpu([patcher])
            return model
        # Anything else (dtype-only, explicit CPU, etc.): pass through.
        return _orig_to(*args, **kwargs)

    def _patched_cpu():
        # pixal3d's swap-out: unpatch + tell ComfyUI we're done with this slot.
        patcher.unpatch_model(device_to=offload_device)
        comfy.model_management.soft_empty_cache()
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
# Generate + Extract GLB (fused for MVP — keeps SparseTensors out of IPC)
# ============================================================================

def generate_glb(
    image: torch.Tensor,
    camera_params: dict,
    seed: int = 42,
    pipeline_type: str = "1024_cascade",
    attn_backend: str = "auto",
    max_num_tokens: int = 49152,
    # Sampler knobs (defaults match inference.py)
    ss_steps: int = 12,
    ss_guidance: float = 7.5,
    ss_rescale: float = 0.7,
    ss_rescale_t: float = 5.0,
    shape_steps: int = 12,
    shape_guidance: float = 7.5,
    shape_rescale: float = 0.5,
    shape_rescale_t: float = 3.0,
    tex_steps: int = 12,
    tex_guidance: float = 1.0,
    tex_rescale: float = 0.0,
    tex_rescale_t: float = 3.0,
    # GLB extraction
    decimation_target: int = 200000,
    texture_size: int = 2048,
    pre_simplify: bool = True,
    pre_simplify_target_faces: int = 2_000_000,
    force_opaque: bool = True,
    filename_prefix: str = "pixal3d",
) -> str:
    """Run cascade + extract GLB. Returns absolute path to the saved GLB."""
    pipeline = init_pipeline(attn_backend=attn_backend)

    pil = comfy_image_to_pil(image)

    torch.manual_seed(seed)
    log.info(f"[pixal3d] Running cascade ({pipeline_type}, seed={seed})")
    mesh_list, (shape_slat, tex_slat, res) = pipeline.run(
        pil,
        camera_params=camera_params,
        seed=seed,
        sparse_structure_sampler_params={
            "steps": ss_steps,
            "guidance_strength": ss_guidance,
            "guidance_rescale": ss_rescale,
            "rescale_t": ss_rescale_t,
        },
        shape_slat_sampler_params={
            "steps": shape_steps,
            "guidance_strength": shape_guidance,
            "guidance_rescale": shape_rescale,
            "rescale_t": shape_rescale_t,
        },
        tex_slat_sampler_params={
            "steps": tex_steps,
            "guidance_strength": tex_guidance,
            "guidance_rescale": tex_rescale,
            "rescale_t": tex_rescale_t,
        },
        preprocess_image=False,
        return_latent=True,
        pipeline_type=pipeline_type,
        max_num_tokens=max_num_tokens,
    )

    mesh = mesh_list[0]
    log.info(f"[pixal3d] Mesh extracted at resolution {res}; baking vertex colors + GLB")

    # MVP export path: query per-vertex PBR from the voxel grid, write a vertex-colored
    # GLB. Upstream o_voxel.postprocess.to_glb (which does UV unwrap + texture baking) is
    # not available in the o_voxel_vb_ap fork (its nvdiffrast-dependent paths were
    # stripped). We trade texture-map fidelity for an MVP that uses only the fork's API.
    # A future iteration can do the drtk-based UV bake — see plan file Phase 2.
    import trimesh

    verts = mesh.vertices.detach()
    faces = mesh.faces.detach()
    with torch.no_grad():
        vattrs = mesh.query_vertex_attrs()  # [N, C], C covers base_color/metallic/roughness/alpha
    base_color_slice = pipeline.pbr_attr_layout.get("base_color", slice(0, 3))
    rgb = vattrs[:, base_color_slice].clamp(0.0, 1.0).cpu().numpy()
    rgb = (rgb * 255).astype(np.uint8)
    if force_opaque:
        a = np.full((rgb.shape[0], 1), 255, dtype=np.uint8)
        vertex_colors = np.concatenate([rgb, a], axis=1)
    else:
        alpha_slice = pipeline.pbr_attr_layout.get("alpha", None)
        if alpha_slice is not None:
            a = vattrs[:, alpha_slice].clamp(0.0, 1.0).cpu().numpy()
            a = (a * 255).astype(np.uint8)
            if a.ndim == 2 and a.shape[1] == 1:
                a = a[:, 0]
            vertex_colors = np.concatenate([rgb, a[:, None]], axis=1)
        else:
            vertex_colors = rgb

    tri = trimesh.Trimesh(
        vertices=verts.cpu().numpy(),
        faces=faces.cpu().numpy(),
        vertex_colors=vertex_colors,
        process=False,
    )

    # GLTF rotation (Y-up) — verbatim from inference.py:181
    rot = np.array(
        [
            [-1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    tri.apply_transform(rot)

    out_dir = folder_paths.get_output_directory()
    ts = int(time.time() * 1000)
    out_path = os.path.join(out_dir, f"{filename_prefix}_{ts}.glb")
    tri.export(out_path)
    log.info(f"[pixal3d] Saved GLB to {out_path}")

    # Drop large intermediates before returning across IPC.
    del mesh, mesh_list, shape_slat, tex_slat, tri, vattrs
    gc.collect()
    torch.cuda.empty_cache()

    return out_path
