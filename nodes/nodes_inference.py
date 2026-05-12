"""Pixal3D inference nodes — preprocess, camera, and the fused generate-to-GLB node."""

import logging

import torch
from comfy_api.latest import io

log = logging.getLogger("pixal3d")


class Pixal3DPreprocessImage(io.ComfyNode):
    """Background removal + alpha bbox crop + 1024-max resize."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Pixal3DPreprocessImage",
            display_name="Pixal3D Preprocess Image",
            category="Pixal3D",
            description=(
                "Runs the pipeline's rembg + alpha bbox crop + 1024-max resize. "
                "Skip this only if your image is already a clean RGBA with the "
                "subject centered."
            ),
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("bg_r", default=0, min=0, max=255, optional=True),
                io.Int.Input("bg_g", default=0, min=0, max=255, optional=True),
                io.Int.Input("bg_b", default=0, min=0, max=255, optional=True),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    def execute(cls, image, bg_r: int = 0, bg_g: int = 0, bg_b: int = 0):
        from .stages import preprocess_image

        out = preprocess_image(image, bg_color=(bg_r, bg_g, bg_b))
        return io.NodeOutput(out)


class Pixal3DEstimateCamera(io.ComfyNode):
    """Run MoGe-2 to infer camera_angle_x + distance from the preprocessed image."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Pixal3DEstimateCamera",
            display_name="Pixal3D Estimate Camera",
            category="Pixal3D",
            description=(
                "Uses MoGe-2 (Ruicheng/moge-2-vitl, ~0.9 GB, downloaded on first run) "
                "to estimate camera intrinsics and a default distance for back-projection."
            ),
            inputs=[
                io.Image.Input("image", tooltip="Preprocessed (square, 1024-max) image."),
                io.Float.Input("mesh_scale", default=1.0, min=0.1, max=10.0, step=0.05, optional=True),
                io.Int.Input("extend_pixel", default=0, min=0, max=128, optional=True),
                io.Int.Input("image_resolution", default=512, min=256, max=2048, step=64, optional=True),
            ],
            outputs=[
                io.Custom("PIXAL3D_CAMERA").Output(display_name="camera"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image,
        mesh_scale: float = 1.0,
        extend_pixel: int = 0,
        image_resolution: int = 512,
    ):
        from .stages import estimate_camera

        cam = estimate_camera(
            image,
            mesh_scale=mesh_scale,
            extend_pixel=extend_pixel,
            image_resolution=image_resolution,
        )
        log.info(
            f"[Pixal3DEstimateCamera] camera_angle_x={cam['camera_angle_x']:.4f}, "
            f"distance={cam['distance']:.4f}, mesh_scale={cam['mesh_scale']:.4f}"
        )
        return io.NodeOutput(cam)


class Pixal3DGenerateGLB(io.ComfyNode):
    """Fused cascade run + GLB extraction. Returns the saved GLB filepath."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Pixal3DGenerateGLB",
            display_name="Pixal3D Generate GLB",
            category="Pixal3D",
            is_output_node=True,
            description=(
                "Runs the four-stage cascade (sparse structure → shape LR 512 → "
                "shape HR 1024 → texture 1024) and extracts a textured GLB. "
                "Cascade resolution can auto-shrink below the requested HR if the "
                "token budget is exceeded; check logs."
            ),
            inputs=[
                io.Custom("PIXAL3D_PIPELINE").Input("pipeline", tooltip="From Pixal3DLoadPipeline."),
                io.Image.Input("image", tooltip="Preprocessed image."),
                io.Custom("PIXAL3D_CAMERA").Input("camera", tooltip="From Pixal3DEstimateCamera."),
                io.Int.Input("seed", default=42, min=0, max=2**31 - 1),
                io.Int.Input("max_num_tokens", default=49152, min=1024, max=131072, step=1024, optional=True),
                # SS knobs
                io.Int.Input("ss_steps", default=12, min=1, max=64, optional=True),
                io.Float.Input("ss_guidance", default=7.5, min=0.0, max=15.0, step=0.1, optional=True),
                io.Float.Input("ss_rescale", default=0.7, min=0.0, max=1.0, step=0.05, optional=True),
                io.Float.Input("ss_rescale_t", default=5.0, min=0.0, max=10.0, step=0.1, optional=True),
                # Shape knobs
                io.Int.Input("shape_steps", default=12, min=1, max=64, optional=True),
                io.Float.Input("shape_guidance", default=7.5, min=0.0, max=15.0, step=0.1, optional=True),
                io.Float.Input("shape_rescale", default=0.5, min=0.0, max=1.0, step=0.05, optional=True),
                io.Float.Input("shape_rescale_t", default=3.0, min=0.0, max=10.0, step=0.1, optional=True),
                # Tex knobs
                io.Int.Input("tex_steps", default=12, min=1, max=64, optional=True),
                io.Float.Input("tex_guidance", default=1.0, min=0.0, max=15.0, step=0.1, optional=True),
                io.Float.Input("tex_rescale", default=0.0, min=0.0, max=1.0, step=0.05, optional=True),
                io.Float.Input("tex_rescale_t", default=3.0, min=0.0, max=10.0, step=0.1, optional=True),
                # GLB knobs
                io.Int.Input("decimation_target", default=200000, min=10000, max=1000000, step=10000, optional=True),
                io.Int.Input("texture_size", default=2048, min=512, max=4096, step=256, optional=True),
                io.String.Input("filename_prefix", default="pixal3d", optional=True),
            ],
            outputs=[
                io.String.Output(display_name="glb_filepath"),
            ],
        )

    @classmethod
    def execute(
        cls,
        pipeline,
        image,
        camera,
        seed: int = 42,
        max_num_tokens: int = 49152,
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
        decimation_target: int = 200000,
        texture_size: int = 2048,
        filename_prefix: str = "pixal3d",
    ):
        from .stages import generate_glb

        pipeline_type = pipeline.get("pipeline_type", "1024_cascade")
        low_vram = pipeline.get("low_vram", False)

        out = generate_glb(
            image=image,
            camera_params=camera,
            seed=seed,
            pipeline_type=pipeline_type,
            max_num_tokens=max_num_tokens,
            low_vram=low_vram,
            ss_steps=ss_steps,
            ss_guidance=ss_guidance,
            ss_rescale=ss_rescale,
            ss_rescale_t=ss_rescale_t,
            shape_steps=shape_steps,
            shape_guidance=shape_guidance,
            shape_rescale=shape_rescale,
            shape_rescale_t=shape_rescale_t,
            tex_steps=tex_steps,
            tex_guidance=tex_guidance,
            tex_rescale=tex_rescale,
            tex_rescale_t=tex_rescale_t,
            decimation_target=decimation_target,
            texture_size=texture_size,
            filename_prefix=filename_prefix,
        )
        return io.NodeOutput(out)


NODE_CLASS_MAPPINGS = {
    "Pixal3DPreprocessImage": Pixal3DPreprocessImage,
    "Pixal3DEstimateCamera": Pixal3DEstimateCamera,
    "Pixal3DGenerateGLB": Pixal3DGenerateGLB,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Pixal3DPreprocessImage": "Pixal3D Preprocess Image",
    "Pixal3DEstimateCamera": "Pixal3D Estimate Camera",
    "Pixal3DGenerateGLB": "Pixal3D Generate GLB",
}
