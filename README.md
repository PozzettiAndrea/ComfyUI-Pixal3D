WORK IN PROGRESS

# ComfyUI-Pixal3D

ComfyUI nodes for **Pixal3D** (SIGGRAPH 2026, TencentARC) — pixel-aligned image-to-3D generation. Single image in, textured GLB out.

- Project page: <https://ldyang694.github.io/projects/pixal3d/>
- Paper: <https://arxiv.org/abs/2605.10922>
- Upstream code: <https://github.com/TencentARC/Pixal3D>
- Model weights: <https://huggingface.co/TencentARC/Pixal3D>

## License & Usage Restrictions — read first

This wrapper code is MIT. The **Pixal3D model weights** downloaded at runtime are subject to Tencent's separate Pixal3D license, which is **not** open-source-compatible. From `LICENSE` in the upstream Pixal3D repo:

> You agree to use the Pixal3D only for academic purposes, and refrain from using it for any commercial or production purposes under any circumstances.

> Pixal3D IS NOT INTENDED FOR USE WITHIN THE EUROPEAN UNION. IN THE EVENT OF ANY CONFLICT, THIS CLAUSE SHALL PREVAIL.

By installing this node pack you are responsible for complying with that license. The MIT wrapper does not relax those restrictions.

The image encoder is **DinoV3** (Meta). Per Meta's DinoV3 license, downstream uses must attribute: "Built with DINOv3". Military, weapons, and surveillance uses are prohibited.

This pack **does not use `nvdiffrast` or `nvdiffrec_render`** (both are NVIDIA non-commercial-only) — the core image→GLB path does not need them. A future preview node will use `drtk` (Meta, MIT-style) instead, matching ComfyUI-TRELLIS2.

## Install

```bash
cds get pixal3d
```

This sets up an isolated pixi env (Python 3.10, torch 2.6.0+cu124) and installs the pack into `ComfyUI/custom_nodes/ComfyUI-Pixal3D/`. The first run downloads ~25 GB of model weights from HuggingFace (Pixal3D, DinoV3, MoGe-2).

## Nodes (MVP)

| Node | Purpose |
|------|---------|
| `Pixal3DLoadPipeline` | Loads the cascade pipeline + four DinoV3 cond models. Auto-downloads Pixal3D weights. |
| `Pixal3DLoadMoGe` | Loads MoGe-2 for camera-intrinsic estimation. |
| `Pixal3DPreprocessImage` | Background removal + alpha bbox crop + 1024-max resize. |
| `Pixal3DEstimateCamera` | Runs MoGe-2 to estimate camera_angle_x and distance from the input image. |
| `Pixal3DGenerate` | Runs the four-stage cascade (SS → shape LR 512 → shape HR 1024 → texture 1024). |
| `Pixal3DExtractGLB` | Extracts a textured GLB via `o_voxel.postprocess.to_glb`. Saves to `output/`. |

## Hardware

- NVIDIA GPU with **SM ≥ 8.0** (Ampere/Ada/Hopper/Blackwell). flash-attn-3 has no fallback for older GPUs.
- ≥24 GB VRAM recommended for `1024_cascade` with `low_vram=True`. More for `1536_cascade`.
- ~30 GB free disk for model weights.

## Acknowledgements

Built on the work of the Pixal3D authors (Li, Zhao, Chen, Hu, Guo, Zhang, Shan, Hu — Tsinghua / Tencent ARC / Victoria University of Wellington), Microsoft TRELLIS.2, Direct3D-S2, MoGe (Microsoft), and DINOv3 (Meta).

Wrapper authored by Andrea Pozzetti.
