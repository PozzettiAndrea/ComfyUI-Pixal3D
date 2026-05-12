"""Pixal3D loader nodes.

Pixal3DLoadPipeline triggers weight download + pipeline construction. It
returns a sentinel so it can act as an upstream dependency in the graph;
the actual pipeline lives in the module-level cache inside the isolation env.
"""

import logging

import torch
from comfy_api.latest import io

log = logging.getLogger("pixal3d")


class Pixal3DLoadPipeline(io.ComfyNode):
    """Load (and download on first run) the Pixal3D cascade pipeline."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Pixal3DLoadPipeline",
            display_name="Load Pixal3D Pipeline",
            category="Pixal3D",
            description=(
                "Downloads ~22 GB of Pixal3D weights from HuggingFace on first run, "
                "then builds the cascade pipeline + four DinoV3 image-cond models. "
                "Returns a sentinel that downstream nodes depend on."
            ),
            inputs=[
                io.Combo.Input(
                    "pipeline_type",
                    options=["1024_cascade", "1536_cascade"],
                    default="1024_cascade",
                    tooltip="Cascade target resolution. 1536_cascade needs significantly more VRAM.",
                ),
                io.Boolean.Input(
                    "low_vram",
                    default=False,
                    tooltip="Swap each stage to/from CPU. Slower but fits on 24 GB cards.",
                    optional=True,
                ),
                io.Combo.Input(
                    "attn_backend",
                    options=["auto", "flash_attn", "flash_attn_3", "sdpa", "xformers", "naive"],
                    default="auto",
                    tooltip=(
                        "Dense + sparse attention backend (pixal3d native dispatch). "
                        "'auto' lets pixal3d pick the fastest available (flash_attn 2 by default). "
                        "'flash_attn_3' needs the separate flash_attn_interface package. "
                        "Note: sageattention is NOT in pixal3d's native dispatch — would require an upstream patch."
                    ),
                    optional=True,
                ),
            ],
            outputs=[
                io.Custom("PIXAL3D_PIPELINE").Output(display_name="pipeline"),
            ],
        )

    @classmethod
    def execute(
        cls,
        pipeline_type: str = "1024_cascade",
        low_vram: bool = False,
        attn_backend: str = "auto",
    ):
        from .stages import init_pipeline

        init_pipeline(low_vram=low_vram, attn_backend=attn_backend)
        # Sentinel — pipeline lives in module-level cache.
        return io.NodeOutput({
            "pipeline_type": pipeline_type,
            "low_vram": low_vram,
            "attn_backend": attn_backend,
        })


NODE_CLASS_MAPPINGS = {
    "Pixal3DLoadPipeline": Pixal3DLoadPipeline,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Pixal3DLoadPipeline": "Load Pixal3D Pipeline",
}
