"""Pixal3D mesh-pipeline nodes -- split the monolithic GenerateGLB into 4 steps.

Workflow:
    Pixal3DGenerateMesh -> (TRIMESH, PIXAL3D_VOXELGRID)
    Pixal3DProcessMesh  -> TRIMESH (with UVs + normals)
    Pixal3DRasterizePBR -> TRIMESH (with PBRMaterial + baseColorTexture + mR texture)
    Pixal3DExportGLB    -> STRING glb_filepath

Mirrors TRELLIS2's Trellis2{Process,RasterizePBR,Export}* node decomposition.
The TRIMESH socket is a CPU-numpy trimesh.Trimesh; PIXAL3D_VOXELGRID is a dict
of numpy arrays. Both cross IPC by pickling.
"""

import logging

from comfy_api.latest import io

log = logging.getLogger("pixal3d")


class Pixal3DGenerateMesh(io.ComfyNode):
    """Run the 4-stage cascade. Emits the raw DC mesh + the sparse PBR voxel grid."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Pixal3DGenerateMesh",
            display_name="Pixal3D Generate Mesh",
            category="Pixal3D",
            description=(
                "Runs the four-stage cascade (sparse structure -> shape LR 512 -> "
                "shape HR 1024 -> texture 1024). Outputs the raw DC mesh (no cleanup, "
                "no UVs) and the sparse PBR voxel grid for downstream baking. Pipe "
                "into Pixal3DProcessMesh + Pixal3DRasterizePBR for upstream-parity "
                "UV-baked output, or use Pixal3DGenerateGLB for the vertex-color "
                "convenience path."
            ),
            inputs=[
                io.Custom("PIXAL3D_PIPELINE").Input("pipeline", tooltip="From Pixal3DLoadPipeline."),
                io.Image.Input("image", tooltip="Preprocessed image."),
                io.Custom("PIXAL3D_CAMERA").Input("camera", tooltip="From Pixal3DEstimateCamera."),
                io.Int.Input("seed", default=42, min=0, max=2**31 - 1),
                io.Int.Input("max_num_tokens", default=49152, min=1024, max=131072, step=1024, optional=True),
                io.Int.Input("ss_steps", default=12, min=1, max=64, optional=True),
                io.Float.Input("ss_guidance", default=7.5, min=0.0, max=15.0, step=0.1, optional=True),
                io.Float.Input("ss_rescale", default=0.7, min=0.0, max=1.0, step=0.05, optional=True),
                io.Float.Input("ss_rescale_t", default=5.0, min=0.0, max=10.0, step=0.1, optional=True),
                io.Int.Input("shape_steps", default=12, min=1, max=64, optional=True),
                io.Float.Input("shape_guidance", default=7.5, min=0.0, max=15.0, step=0.1, optional=True),
                io.Float.Input("shape_rescale", default=0.5, min=0.0, max=1.0, step=0.05, optional=True),
                io.Float.Input("shape_rescale_t", default=3.0, min=0.0, max=10.0, step=0.1, optional=True),
                io.Int.Input("tex_steps", default=12, min=1, max=64, optional=True),
                io.Float.Input("tex_guidance", default=1.0, min=0.0, max=15.0, step=0.1, optional=True),
                io.Float.Input("tex_rescale", default=0.0, min=0.0, max=1.0, step=0.05, optional=True),
                io.Float.Input("tex_rescale_t", default=3.0, min=0.0, max=10.0, step=0.1, optional=True),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="mesh"),
                io.Custom("PIXAL3D_VOXELGRID").Output(display_name="voxelgrid"),
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
        ss_steps: int = 12, ss_guidance: float = 7.5, ss_rescale: float = 0.7, ss_rescale_t: float = 5.0,
        shape_steps: int = 12, shape_guidance: float = 7.5, shape_rescale: float = 0.5, shape_rescale_t: float = 3.0,
        tex_steps: int = 12, tex_guidance: float = 1.0, tex_rescale: float = 0.0, tex_rescale_t: float = 3.0,
    ):
        from .stages import generate_mesh_and_voxelgrid, _YUP_TO_ZUP_ROT, _phase
        with _phase("Pixal3DGenerateMesh.execute"):
            tri, voxelgrid = generate_mesh_and_voxelgrid(
                image=image,
                camera_params=camera,
                seed=seed,
                pipeline_type=pipeline.get("pipeline_type", "1024_cascade"),
                attn_backend=pipeline.get("attn_backend", "auto"),
                max_num_tokens=max_num_tokens,
                ss_steps=ss_steps, ss_guidance=ss_guidance, ss_rescale=ss_rescale, ss_rescale_t=ss_rescale_t,
                shape_steps=shape_steps, shape_guidance=shape_guidance, shape_rescale=shape_rescale, shape_rescale_t=shape_rescale_t,
                tex_steps=tex_steps, tex_guidance=tex_guidance, tex_rescale=tex_rescale, tex_rescale_t=tex_rescale_t,
            )
            # Pixal3D's cascade outputs Y-up natively; TRELLIS2's verified-working
            # cumesh+drtk UV-bake regime expects Z-up. Rotate here so ProcessMesh /
            # RasterizePBR see the same frame TRELLIS2 was tested on; ExportGLB
            # rotates back to Y-up for the final GLB. Voxelgrid stays Y-up;
            # rasterize_pbr rotates valid_pos back to Y-up for the voxel sample.
            tri.apply_transform(_YUP_TO_ZUP_ROT)
            log.info(
                f"[Pixal3DGenerateMesh] mesh={len(tri.vertices)} verts / {len(tri.faces)} faces, "
                f"voxelgrid={voxelgrid['attrs'].shape[0]} voxels x{voxelgrid['attrs'].shape[1]} attrs "
                f"(mesh rotated Y-up -> Z-up for downstream bake)"
            )
            return io.NodeOutput(tri, voxelgrid)


class Pixal3DProcessMesh(io.ComfyNode):
    """Heavy cumesh cleanup + UV unwrap. Output mesh is ready for Pixal3DRasterizePBR."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Pixal3DProcessMesh",
            display_name="Pixal3D Process Mesh",
            category="Pixal3D",
            description=(
                "fill_holes -> (optional) DC remesh -> floater removal -> simplify -> "
                "weld vertices -> UV unwrap. Mirrors TRELLIS2 Trellis2ProcessMesh and "
                "upstream o_voxel.postprocess.to_glb's geometry stage."
            ),
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Boolean.Input("remesh", default=False, optional=True,
                    tooltip="Run dual-contouring remesh for cleaner topology. Slower; usually unneeded since the cascade already produces uniform DC output."),
                io.Int.Input("remesh_resolution", default=512, min=64, max=2048, step=64, optional=True),
                io.Float.Input("remesh_band", default=1.0, min=0.1, max=5.0, step=0.1, optional=True),
                io.Boolean.Input("remove_inner_faces", default=False, optional=True,
                    tooltip="Only effective when remesh=on. Drops quads whose centers fall inside the original mesh's bulk."),
                io.Boolean.Input("fill_holes", default=True, optional=True),
                io.Float.Input("fill_holes_perimeter", default=0.03, min=0.001, max=0.5, step=0.001, optional=True),
                io.Float.Input("floater_threshold", default=1e-3, min=0.0, max=0.1, step=0.001, optional=True,
                    tooltip="Min area for connected components. 0 disables."),
                io.Int.Input("target_face_count", default=200000, min=1000, max=5000000, step=1000),
                io.Boolean.Input("weld_vertices", default=True, optional=True),
                io.Int.Input("weld_digits", default=4, min=1, max=8, optional=True),
                io.Float.Input("chart_cone_angle", default=90.0, min=0.0, max=359.9, step=1.0, optional=True),
                io.Int.Input("chart_refine_iterations", default=0, min=0, max=10, optional=True),
                io.Int.Input("chart_global_iterations", default=1, min=0, max=10, optional=True),
                io.Int.Input("chart_smooth_strength", default=1, min=0, max=10, optional=True),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="mesh"),
            ],
        )

    @classmethod
    def execute(
        cls,
        trimesh,
        remesh: bool = False,
        remesh_resolution: int = 512,
        remesh_band: float = 1.0,
        remove_inner_faces: bool = False,
        fill_holes: bool = True,
        fill_holes_perimeter: float = 0.03,
        floater_threshold: float = 1e-3,
        target_face_count: int = 200000,
        weld_vertices: bool = True,
        weld_digits: int = 4,
        chart_cone_angle: float = 90.0,
        chart_refine_iterations: int = 0,
        chart_global_iterations: int = 1,
        chart_smooth_strength: int = 1,
    ):
        from .stages import process_mesh, _phase
        with _phase("Pixal3DProcessMesh.execute"):
            out = process_mesh(
                trimesh,
                remesh=remesh,
                remesh_resolution=remesh_resolution,
                remesh_band=remesh_band,
                remove_inner_faces=remove_inner_faces,
                fill_holes=fill_holes,
                fill_holes_perimeter=fill_holes_perimeter,
                floater_threshold=floater_threshold,
                target_face_count=target_face_count,
                weld_vertices=weld_vertices,
                weld_digits=weld_digits,
                chart_cone_angle=chart_cone_angle,
                chart_refine_iterations=chart_refine_iterations,
                chart_global_iterations=chart_global_iterations,
                chart_smooth_strength=chart_smooth_strength,
            )
            return io.NodeOutput(out)


class Pixal3DRasterizePBR(io.ComfyNode):
    """drtk UV-space PBR bake: trimesh+UVs+voxelgrid -> trimesh with baked PBR textures."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Pixal3DRasterizePBR",
            display_name="Pixal3D Rasterize PBR",
            category="Pixal3D",
            description=(
                "Bake baseColorTexture + metallicRoughnessTexture from the cascade's "
                "PBR voxel grid onto a UV-mapped mesh. Uses drtk for UV rasterization "
                "and flex_gemm_ap.grid_sample_3d for sparse voxel sampling. Optionally "
                "snap texel positions back to the pre-simplification mesh via cuBVH "
                "for higher texture accuracy."
            ),
            inputs=[
                io.Custom("TRIMESH").Input("trimesh", tooltip="Mesh WITH UVs (from Pixal3DProcessMesh)."),
                io.Custom("PIXAL3D_VOXELGRID").Input("voxelgrid", tooltip="From Pixal3DGenerateMesh."),
                io.Int.Input("texture_size", default=2048, min=512, max=8192, step=512),
                io.Custom("TRIMESH").Input("original_mesh", optional=True,
                    tooltip="Raw pre-simplification mesh (from Pixal3DGenerateMesh) for BVH-snap of texel positions. Improves sharpness."),
                io.Boolean.Input("double_sided", default=False, optional=True,
                    tooltip="Mark the baked material as double-sided."),
                io.Combo.Input(
                    "bake_mode",
                    options=["pbr", "xyz_position", "xyz_normal"],
                    default="pbr",
                    tooltip=(
                        "What to bake into baseColorTexture.\n"
                        "  pbr           - production: sample voxelgrid for base color + "
                        "metallic/roughness/alpha (default).\n"
                        "  xyz_position  - diagnostic: paint each texel with its mesh-frame "
                        "(x, y, z) position as RGB. Red=+X, Green=+Y, Blue=+Z. Lets you SEE "
                        "the mesh's axes on the model surface -- a flipped axis means the "
                        "wrong channel gradients across the model.\n"
                        "  xyz_normal    - diagnostic: paint each texel with its interpolated "
                        "surface normal as RGB (normals in [-1,1] mapped to [0,1]). Lets you "
                        "see face winding / normal direction issues."
                    ),
                    optional=True,
                ),
                io.Boolean.Input(
                    "debug_dump",
                    default=False,
                    tooltip=(
                        "When ON, prints per-stage mesh stats + UV/vertex bboxes to stderr, "
                        "and saves three diagnostic files to ComfyUI/output/ alongside the GLB:\n"
                        "  pixal3d_debug_<ts>_mask.png      - UV-space coverage\n"
                        "  pixal3d_debug_<ts>_face_ids.png  - per-texel face id (mod 256)\n"
                        "  pixal3d_debug_<ts>_mesh.obj      - post-ProcessMesh mesh + UVs (open in Blender)\n"
                        "Use this if textures look wrong; tiny isolated regions in face_ids.png "
                        "indicate xatlas produced too many small charts."
                    ),
                    optional=True,
                ),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="mesh"),
            ],
        )

    @classmethod
    def execute(
        cls,
        trimesh,
        voxelgrid,
        texture_size: int = 2048,
        original_mesh=None,
        double_sided: bool = False,
        bake_mode: str = "pbr",
        debug_dump: bool = False,
    ):
        from .stages import rasterize_pbr, _phase
        with _phase(f"Pixal3DRasterizePBR.execute ({bake_mode}{' +debug' if debug_dump else ''})"):
            out = rasterize_pbr(
                trimesh,
                voxelgrid,
                texture_size=texture_size,
                original_mesh=original_mesh,
                double_sided=double_sided,
                bake_mode=bake_mode,
                debug_dump=debug_dump,
            )
            return io.NodeOutput(out)


class Pixal3DExportGLB(io.ComfyNode):
    """Apply the Z-up -> Y-up rotation and write the trimesh to ComfyUI/output as GLB."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Pixal3DExportGLB",
            display_name="Pixal3D Export GLB",
            category="Pixal3D",
            is_output_node=True,
            description=(
                "Rotates the mesh from pixal3d internal Z-up to glTF Y-up and "
                "writes a GLB to ComfyUI's output directory. Returns the absolute "
                "filepath as a STRING (wire to Preview3D's model_file input)."
            ),
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.String.Input("filename_prefix", default="pixal3d", optional=True),
            ],
            outputs=[
                io.String.Output(display_name="glb_filepath"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, filename_prefix: str = "pixal3d"):
        from .stages import export_glb_yup, _phase
        with _phase("Pixal3DExportGLB.execute"):
            # ProcessMesh + RasterizePBR run in a Z-up working frame; rotate
            # back to glTF Y-up here for the final file.
            path = export_glb_yup(trimesh, filename_prefix=filename_prefix)
            return io.NodeOutput(path)


NODE_CLASS_MAPPINGS = {
    "Pixal3DGenerateMesh": Pixal3DGenerateMesh,
    "Pixal3DProcessMesh": Pixal3DProcessMesh,
    "Pixal3DRasterizePBR": Pixal3DRasterizePBR,
    "Pixal3DExportGLB": Pixal3DExportGLB,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Pixal3DGenerateMesh": "Pixal3D Generate Mesh",
    "Pixal3DProcessMesh": "Pixal3D Process Mesh",
    "Pixal3DRasterizePBR": "Pixal3D Rasterize PBR",
    "Pixal3DExportGLB": "Pixal3D Export GLB",
}
