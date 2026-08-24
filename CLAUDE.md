# ComfyUI-Pixal3D — Wrapping Plan

## Context

**Why**: Pixal3D (SIGGRAPH 2026, TencentARC) is a state-of-the-art single-image → textured-GLB pipeline built on the TRELLIS.2 backbone. It's already wrappable: a clean Python API (`Pixal3DImageTo3DPipeline.from_pretrained()` + `.run(image, camera_params=...)`) and a reference `inference.py`. Our `ComfyUI-TRELLIS2` pack is the closest precedent — same dependency stack (torch 2.6 / cu124, flash-attn-3, cumesh, flex_gemm, o_voxel, DinoV3) and same cascade architecture. We will mirror its layout, swap in the Pixal3D-specific pipeline, and ship as an MVP that produces a GLB end-to-end.

**Licensing posture (locked)**:
- **Wrapper code**: MIT, with a prominent README disclaimer.
- **Pixal3D model weights**: academic-only per Tencent's license (`repo/LICENSE:15`); README must state non-commercial and EU-excluded clauses verbatim.
- **No nvdiffrast / nvdiffrec_render** anywhere — they are non-commercial-only NVIDIA licenses and the core inference path doesn't need them. We will use the same `_vb_ap` / `_ap` cuda-wheels variants TRELLIS2 uses, which are sourced from `PozzettiAndrea/Trellis.2.drtk` (already nvdiffrast-free). Preview = GLB viewer only. A later phase will port `pixal3d/renderers/{pbr_mesh_renderer,mesh_renderer}.py` from `nvdiffrast` → `drtk` (mirroring what we did for TRELLIS2) to unlock HDRI/clay/basecolor preview modes; **out of scope for this PR**.

**Distribution posture (locked)**:
- **First publish: `PozzettiAndrea/ComfyUI-Pixal3D` as a PRIVATE GitHub repo.** Do not flip to public until we have done a license-review pass and confirmed the academic-only / EU-exclusion disclaimers are correctly surfaced in the README.
- **Install entry point**: `cds get pixal3d`. The config lives at `/home/work/coding-scripts/comfy-dev-cli/config/setup/pixal3d.yml` and follows the same shape as `trellis2.yml` / `sam3dobjects.yml`.
- Since the repo is private, `cds get pixal3d` will require `gh auth` on the dev machine. Once we go public this is transparent.

**Intended outcome**: a `ComfyUI-Pixal3D` pack (private GitHub repo) installable via `cds get pixal3d`, exposing 5 MVP nodes that reproduce `repo/inference.py` end-to-end inside a ComfyUI workflow, isolated in its own pixi env via comfy-env, with no nvdiffrast dependency.

---

## Approach

Process-isolated pack, mirroring `ComfyUI-TRELLIS2`. All Pixal3D Python code (torch 2.6 + cu124) runs in a comfy-env subprocess; ComfyUI's host env is untouched. The `pixal3d/` package is **vendored** (not pip-installed from the repo root) so we control the import surface.

---

## Repo Layout

```
ComfyUI-Pixal3D/
├── comfy-env-root.toml          # [cuda] = ["flash-attn","sageattention"]; [node_packs] for GeometryPack
├── install.py                   # from comfy_env import install; install()
├── prestartup_script.py         # setup_env(); copy_files(assets); copy_viewer("glb_three", web/)
├── __init__.py                  # register_nodes() + register TRELLIS2-compatible model configs if helpful
├── requirements.txt             # comfy-env, comfy-3d-viewers, comfy-sparse-attn, trimesh[easy]
├── pyproject.toml               # version, MIT, publisher
├── LICENSE                      # MIT (wrapper code only)
├── README.md                    # Must include Pixal3D academic-only + EU-exclusion disclaimers verbatim
├── nodes/
│   ├── comfy-env.toml           # python 3.10; [cuda]=["flex_gemm_ap","cumesh_vb","o_voxel_vb_ap","flash-attn","sageattention","drtk"]
│   ├── __init__.py              # node class mappings
│   ├── pixal3d/                 # vendored copy of repo/pixal3d/ (no compiled extensions)
│   ├── stages.py                # heavy lifting: load, preprocess, camera, generate, extract
│   ├── nodes_loader.py          # LoadPixal3DPipeline
│   ├── nodes_inference.py       # Pixal3DPreprocessImage, Pixal3DEstimateCamera, Pixal3DGenerateGLB
│   └── utils/
│       └── hf_download.py       # _comfy_tqdm() shim
├── web/                         # populated at prestartup by copy_viewer("glb_three")
└── workflows/
    └── pixal3d_basic.json
```

---

## MVP Node Decomposition

Four nodes; reproduces `inference.py` end-to-end. Mirrors SAM3DObjects' Load → Estimate → Generate → Export pattern.

| Node | File | Wraps |
|------|------|-------|
| `Pixal3DLoadPipeline` | `nodes_loader.py` | (was) `init_pipeline()`. (Now planned) returns thin config dict only. |
| `Pixal3DPreprocessImage` | `nodes_inference.py` | `pipeline.preprocess_image()` (rembg + alpha bbox crop + 1024 resize). |
| `Pixal3DEstimateCamera` | `nodes_inference.py` | `get_camera_params_wild_moge()` — refactored to take a tensor (not a temp file). |
| `Pixal3DGenerateGLB` | `nodes_inference.py` | Fused cascade `pipeline.run()` + vertex-color GLB export via trimesh. `is_output_node=True`. |

---

## Verification status

1. **Install via `cds dev get pixal3d`** ✅ done — pixi env at `~/.ce/.pixi/envs/pixal3d-nodes/` (Python 3.10.20, torch 2.8+cu128) with all 6 cuda-wheels (`flex_gemm_ap`, `cumesh_vb`, `o_voxel_vb_ap`, `flash-attn 2.8.3`, `sageattention 2.2.0`, `drtk 0.1.0`).
2. **Sanity import** ✅ done — pixal3d.pipelines.Pixal3DImageTo3DPipeline + DinoV3 + MoGe + sparse backends import cleanly; `nvdiffrast` raises `ModuleNotFoundError` as expected.
3. **Weights download** ✅ done — `ComfyUI/models/pixal3d/ckpts/*.safetensors` (8 files, 23 GB).
4. **Stages smoke test (Python script, off the node graph)** ✅ done — produced a 162.8 MB GLB in 207 s.
5. **ComfyUI-graph end-to-end via /prompt API** ✅ done — 220.8 s on RTX 3090, valid GLB.
6. **VRAM** ✅ confirmed `low_vram=True` runs `1024_cascade` to completion on a 24 GB 3090.
7. **No nvdiffrast leak** ✅ confirmed.

---

## Patches accumulated during the smoke-test debug (already pushed)

These are the runtime fixes from the stages-level E2E run that the upstream `inference.py` didn't anticipate:

- `stages.py` — strip `GITHUB_TOKEN` before any `torch.hub` call (`_validate_not_a_forked_repo` 401s on bad/over-restrictive tokens; NAF model lives on `valeoai/NAF`).
- `stages.py` — patch `pixal3d.pipelines.rembg.BiRefNet.__init__` to swap gated `briaai/RMBG-2.0` → public `ZhengPeng7/BiRefNet`; cast inputs to model dtype at `__call__` time.
- `stages.py` — alias `DINOv3ViTModel.layer → .model.layer` in `_build_cond` (transformers ≥5 moved the layer ModuleList one level deeper).
- `stages.py` — final-stage GLB export rewrite: `o_voxel_vb_ap` has no `postprocess.to_glb` (the drtk-fork stripped nvdiffrast paths). Replaced with `MeshWithVoxel.query_vertex_attrs()` + `trimesh.Trimesh(..., vertex_colors=...)` MVP. `force_opaque` toggle on the node forces alpha=1.0.
- `comfy-env.toml` — `natten` upgraded to the SHI-Labs torch2.8+cu128 wheel.
- `nodes/__init__.py` + `nodes/pixal3d/__init__.py` — `sys.modules` aliases `cumesh_vb→cumesh`, `o_voxel_vb_ap→o_voxel`, `flex_gemm_ap→flex_gemm`.
- `.gitignore` — removed the unanchored `models/` pattern that was hiding `nodes/pixal3d/models/`.
- `nodes/comfy-env.toml` — dropped `utils3d` URL pin (conflicted with MoGe's transitive dep; not used on inference path).
- 21 vendored files — `.cuda()` / `.to("cuda")` → `comfy.model_management.get_torch_device()`.
- 12 vendored model files — `nn.Linear` / `nn.Conv*d` → `comfy.ops.disable_weight_init.Linear` / `Conv*d`.

---

## ComfyUI-native memory management (done across commits)

1. **Vendored code refactor (`7d095a6` + `b5f5734`)**: every `.cuda()` / `.to("cuda")` in vendored pixal3d swapped for `comfy.model_management.get_torch_device()`; every `nn.Linear` / `nn.Conv*d` swapped for `comfy.ops.disable_weight_init.Linear` / `Conv*d`.

2. **ModelPatcher integration (`bf4e139`)**: each cascade model (8) + DinoV3 cond models (4) + BiRefNet rembg wrapped in `comfy.model_patcher.ModelPatcher`. Their `.to(device)` / `.cpu()` monkey-patched to route through `comfy.model_management.load_models_gpu([patcher])` / `patcher.unpatch_model(device_to=offload)`. Pixal3D's `pipeline.run()` is unchanged.

3. **NAF: local + shared (`502e8c2`)**: NAF (`valeoai/NAF`) is now hosted in `ComfyUI/models/naf/` (clone of the source + `naf_release.pth`). Constructed once into module-level `_naf`, attached via `self.__dict__["naf_model"] = _naf` on each cond extractor (bypassing nn.Module child registration). 1 NAF in memory instead of 3.

4. **Per-phase timing prints (`502e8c2`)**: `_phase(label)` context manager prints `[pixal3d] >>> label ...` / `[pixal3d] <<< label  (Xs)` to stderr (flushed). Wraps every major phase of `init_pipeline` and `init_moge`. Cold boot now reports:
   - download weights: ~0.1 s (cached)
   - from_pretrained: 8 cascade safetensors → CPU: ~72 s
   - build DinoV3 cond × 4: ~0.7 s each
   - ModelPatcher wrap: ~0.4 s
   - NAF: ~0.3 s
   - **TOTAL: ~78 s**

---

## Make Pixal3DLoadPipeline a thin config dict (TRELLIS2-style)

**Context**: The 78 s cold-boot currently fires inside `Pixal3DLoadPipeline.execute()`. TRELLIS2's equivalent returns just a config dict; the actual model load is lazy, fired from the inference nodes.

**User intent (verbatim)**: "the pipeline should be loaded within this node here Pixal3D Generate GLB if anything, the 'model' config should just be a thin fucking dict like it is in trellis2".

**Plan** (minimal):

1. `nodes/nodes_loader.py:Pixal3DLoadPipeline.execute()` — drop the `init_pipeline(attn_backend=...)` call. Return only `{"pipeline_type": pipeline_type, "attn_backend": attn_backend}`. Node executes in ~0 ms.
2. `nodes/nodes_inference.py:Pixal3DGenerateGLB.execute()` — read `attn_backend` from the dict and pass it through `generate_glb(..., attn_backend=...)`.
3. `nodes/stages.py:generate_glb()` — accept `attn_backend: str = "auto"` and forward to lazy `init_pipeline(attn_backend=attn_backend)`.

**Where the 78 s shows up after the change**: whichever node executes first in the graph that touches the cascade — that's `Pixal3DPreprocessImage` (calls `pipeline.preprocess_image` via stages, which lazy-inits).

---

## Out of Scope (future PRs)

- **Phase 2 — nvdiffrast → drtk preview port**.
- **Stage-split nodes**: separate `Pixal3DSampleSS`, `Pixal3DSampleShapeLR`, `Pixal3DSampleShapeHR`, `Pixal3DSampleTexture`, `Pixal3DDecodeLatent` (TRELLIS2 has these — useful for per-stage seed/guidance tuning and latent caching).
- **Multi-view input**: paper mentions but the released `Pixal3DImageTo3DPipeline.run()` is single-image.
- **`1536_cascade`**: expose with a clear VRAM warning.
- **Manual camera-param override**: bypass MoGe for synthetic / clean-camera inputs.
- **Add `natten` to cuda-wheels**: drop the URL pin once added.
- **Split BiRefNet rembg into its own `Pixal3DRemoveBackground` node** so `Pixal3DPreprocessImage` becomes pure-PIL (no model load).

---

## Open Risks

- **`utils3d` is a personal GitHub Release** — pinned wheel URL could 404.
- **`camenduru/dinov3-vitl16-pretrain-lvd1689m` is a community mirror** — fall back to `facebook/dinov3-vitl16-pretrain-lvd1689m` if it goes away.
- **First-run download is ~25 GB.**
- **SM <8.0 unsupported** — flash-attn-3 has no fallback. Loader probes and errors early.
- **Tencent EU clause** — surface in README; no enforcement at wrapper level.

---

# 🔴 Tiny plan: diagnose the post-`init_pipeline` stall

## Symptom

Latest worker log shows clean init:
```
[pixal3d] <<< build DinoV3 cond 'tex_1024'  (0.6s)
[pixal3d] <<< ModelPatcher wrap: 13 models  (0.4s)
[pixal3d] <<< NAF: build singleton + attach to 3 cond models  (0.3s)
[pixal3d] <<< init_pipeline TOTAL  (78.0s)
[ComfyUI-Pixal3D] Health check: ping (timeout=5.0s)...
[ComfyUI-Pixal3D] Health check: ok
... ComfyUI-Manager startup fetches finish ...
[ComfyUI-Manager] All startup tasks have been completed.
```

…and then **nothing**. The next node (`Pixal3DPreprocessImage`) never logs activity. No traceback, no progress, no error — just silence. User Ctrl-C's.

## Cause

Unknown. Three plausible buckets:
- **(a)** `Pixal3DLoadPipeline.execute()` actually still hung somewhere after the `TOTAL` print (e.g. inside `io.NodeOutput(...)` serialization or the comfy-env IPC reply).
- **(b)** LoadPipeline returned cleanly to the worker, but the IPC reply to ComfyUI's main process is wedged → main process doesn't know LoadPipeline finished → won't dispatch the next node.
- **(c)** ComfyUI did dispatch Preprocess to the worker, but Preprocess is hanging silently inside `pipeline.preprocess_image(...)` (e.g. lazy rembg fetch, BiRefNet patcher load_models_gpu deadlock).

We have **zero log signal** to distinguish these three right now, because no node executor has phase prints around its `execute()` body.

## Diagnostic plan (one commit)

1. **Add `_phase("<NodeName>.execute")`** around each node's `execute()` body in `nodes_inference.py` (Pixal3DPreprocessImage, Pixal3DEstimateCamera, Pixal3DGenerateGLB) and `nodes_loader.py` (Pixal3DLoadPipeline).

2. **Make `Pixal3DLoadPipeline` thin (the user-requested refactor)**: drop the `init_pipeline(...)` call from its `execute()`. Pass `attn_backend` through the dict to `generate_glb()` which lazy-inits.

3. **Restart ComfyUI, queue the workflow, capture the log.** What we'll see:

| Symptom | Diagnosis |
|---|---|
| `>>> LoadPipeline.execute` but no `<<< LoadPipeline.execute` | Hang inside LoadPipeline. With the refactor in (2), LoadPipeline should be <1ms — if it hangs, the bug is in `io.NodeOutput` / IPC serialization. |
| `<<< LoadPipeline.execute (0.0s)` but no `>>> PreprocessImage.execute` | Gap is in ComfyUI's queue or comfy-env's IPC handoff (outside our wrapper). Report the time gap; next debug target is comfy-env's `metadata.py:proxy()` and `subprocess.py:call_method`. |
| `>>> PreprocessImage.execute` but no `<<< PreprocessImage.execute` | Hang inside our stages code (likely rembg / BiRefNet load via ModelPatcher). Add finer `_phase` inside `preprocess_image()`. |

## Why this is the right next move

- **One small commit, two changes** — same file scope as the user's already-approved "thin dict" refactor.
- **No speculation** — we add visibility before changing anything else. Once we see which bucket the stall falls into, the fix is targeted.
- **Side benefit**: even if the stall is in comfy-env / ComfyUI itself (bucket b), having `_phase` on every node executor is a permanent observability win for future bugs.
