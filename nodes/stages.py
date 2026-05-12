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
from typing import Optional

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


# ============================================================================
# Pipeline init
# ============================================================================

def _patch_rembg_to_public_model():
    """Pixal3D's pipeline.json pins briaai/RMBG-2.0 which is a gated HF repo.
    Substitute the public ZhengPeng7/BiRefNet (same architecture, MIT-licensed).
    Idempotent.
    """
    from .pixal3d.pipelines import rembg as _rembg
    if getattr(_rembg.BiRefNet, "_pixal3d_patched", False):
        return
    _orig = _rembg.BiRefNet.__init__

    def _patched(self, model_name="ZhengPeng7/BiRefNet", **kwargs):
        if model_name == "briaai/RMBG-2.0":
            log.info(
                "[rembg] Substituting gated 'briaai/RMBG-2.0' -> public "
                "'ZhengPeng7/BiRefNet' (same arch; accept the license at "
                "huggingface.co/briaai/RMBG-2.0 and set HF_TOKEN for the original)."
            )
            model_name = "ZhengPeng7/BiRefNet"
        _orig(self, model_name=model_name, **kwargs)
        # ZhengPeng7/BiRefNet ships as fp16 but the upstream caller doesn't cast inputs.
        # Force float32 so transforms (which produce float32 by default) work.
        # Also pin to CUDA — the pipeline's low_vram swap (`rembg_model.to(self.device)`)
        # doesn't reliably take effect through the wrapper's `to()` signature; ~1 GB on
        # GPU is a cheap price for correctness.
        self.model.float().cuda()

    _rembg.BiRefNet.__init__ = _patched
    _rembg.BiRefNet._pixal3d_patched = True


def _set_attention_backends(backend: str):
    """Wire pixal3d's native dense + sparse attention dispatch.

    Options: 'auto' (skip — let pixal3d auto-detect), 'flash_attn', 'flash_attn_3',
    'sdpa', 'xformers', 'naive'. flash_attn_3 needs the separate
    flash_attn_interface package; we ship flash-attn 2.x.
    """
    if backend == "auto":
        return
    from .pixal3d.modules.attention.config import set_backend as set_dense
    from .pixal3d.modules.sparse.config import set_attn_backend as set_sparse
    set_dense(backend)
    set_sparse(backend)
    log.info(f"[attn] dense + sparse backend = {backend}")


def init_pipeline(low_vram: bool = False, attn_backend: str = "auto") -> "object":
    """Load + cache Pixal3D pipeline + 4 DinoV3 cond models. Idempotent."""
    global _pipeline
    if _pipeline is not None:
        _pipeline.low_vram = low_vram
        _set_attention_backends(attn_backend)
        return _pipeline

    _check_gpu_or_raise()
    local_dir = _download_pixal3d_weights()

    _patch_rembg_to_public_model()
    _set_attention_backends(attn_backend)

    from .pixal3d.pipelines import Pixal3DImageTo3DPipeline
    from .pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
        DinoV3ProjFeatureExtractor,
    )

    log.info(f"[pixal3d] Loading pipeline from {local_dir}")
    pipeline = Pixal3DImageTo3DPipeline.from_pretrained(str(local_dir))

    log.info("[pixal3d] Building DinoV3 cond models")
    pipeline.image_cond_model_ss = _build_cond("ss")
    pipeline.image_cond_model_shape_512 = _build_cond("shape_512")
    pipeline.image_cond_model_shape_1024 = _build_cond("shape_1024")
    pipeline.image_cond_model_tex_1024 = _build_cond("tex_1024")

    pipeline.low_vram = low_vram
    pipeline.cuda()

    pipeline.image_cond_model_ss.cuda()
    pipeline.image_cond_model_shape_512.cuda()
    pipeline.image_cond_model_shape_1024.cuda()
    pipeline.image_cond_model_tex_1024.cuda()

    log.info("[pixal3d] Pre-loading NAF upsamplers")
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
    from moge.model.v2 import MoGeModel

    log.info(f"[moge] Loading {MOGE_REPO}")
    moge = MoGeModel.from_pretrained(MOGE_REPO).to("cuda")
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

def preprocess_image(image: torch.Tensor, bg_color=(0, 0, 0)) -> torch.Tensor:
    """Background removal + alpha bbox crop + 1024-max resize via pipeline.preprocess_image."""
    pipeline = init_pipeline()
    pil = comfy_image_to_pil(image)
    out = pipeline.preprocess_image(pil, bg_color=tuple(bg_color))
    return pil_to_comfy_image(out)


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
    tensor = torch.from_numpy(arr).permute(2, 0, 1).to("cuda")

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
    max_num_tokens: int = 49152,
    low_vram: bool = False,
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
    filename_prefix: str = "pixal3d",
) -> str:
    """Run cascade + extract GLB. Returns absolute path to the saved GLB."""
    pipeline = init_pipeline(low_vram=low_vram)

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
