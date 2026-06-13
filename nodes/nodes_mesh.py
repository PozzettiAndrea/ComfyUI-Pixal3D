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


def _long_lens_camera(camera, multiplier):
    """Isometric long-lens transform: push the camera back x m and shrink the FOV
    by the same factor, so framing is preserved while perspective foreshortening
    flattens toward orthographic (m=1 = identity). Ratios cancel in x_ndc=f/(-z)*x,
    so it's stable at large m. Returns (modified_camera_dict, m)."""
    import math
    m = max(1.0, float(multiplier))
    cam = dict(camera) if isinstance(camera, dict) else camera
    if m == 1.0:
        return cam, 1.0
    a0 = float(cam.get("camera_angle_x", 0.8575560450553894))
    d0 = float(cam.get("distance", 2.0))
    cam = dict(cam)
    cam["distance"] = d0 * m
    cam["camera_angle_x"] = 2.0 * math.atan(math.tan(a0 / 2.0) / m)
    log.info(f"[iso] x{m:.1f}: camera_angle_x {a0:.4f}->{cam['camera_angle_x']:.4f} rad, "
             f"distance {d0:.3f}->{cam['distance']:.3f}")
    return cam, m


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
                io.Custom("PIXAL3D_CAMERA").Input("camera", tooltip="From Pixal3DCameraFromFOV."),
                io.Int.Input("seed", default=42, min=0, max=2**31 - 1,
                    tooltip="Random seed for the whole cascade. Same seed + same inputs = "
                            "reproducible mesh. Change it to get a different generation."),
                io.Boolean.Input("generate_texture", default=True, optional=True,
                    tooltip="Run the texture stage. Turn OFF to skip texture generation "
                            "entirely (shape only) -- much faster; the voxelgrid output is "
                            "then empty (no PBR bake possible). Use for retopology / "
                            "geometry-only workflows."),
                io.Int.Input("max_num_tokens", default=49152, min=1024, max=131072, step=1024, optional=True,
                    tooltip="Upper bound on sparse tokens at the HR shape stage. The pipeline "
                            "lowers the HR resolution (1024 -> ... in 128 steps) until the token "
                            "count fits under this. Higher = more detail but more VRAM/time; "
                            "lower it if you hit out-of-memory."),

                # --- Stage 1: sparse structure (coarse 32^3 occupancy) ---
                io.Int.Input("ss_steps", default=12, min=1, max=64, optional=True,
                    tooltip="Sparse-structure sampler steps -- diffusion iterations for the coarse "
                            "voxel occupancy (overall silhouette/blockout). More = cleaner structure, "
                            "slower; 12 is a good default."),
                io.Float.Input("ss_guidance", default=7.5, min=0.0, max=15.0, step=0.1, optional=True,
                    tooltip="Sparse-structure classifier-free guidance strength: how hard the coarse "
                            "shape is pushed to match the image. Higher = closer to the image but can "
                            "over-sharpen/distort; lower = looser."),
                io.Float.Input("ss_rescale", default=0.7, min=0.0, max=1.0, step=0.05, optional=True,
                    tooltip="Sparse-structure guidance rescale (0-1). Counteracts over-saturation from "
                            "high guidance by renormalizing predictions. 0 = off, ~0.7 = strong."),
                io.Float.Input("ss_rescale_t", default=5.0, min=0.0, max=10.0, step=0.1, optional=True,
                    tooltip="Timestep threshold above which ss_rescale is applied (early/noisy steps). "
                            "Higher = rescale active over more of the schedule."),

                # --- Stages 2-3: shape (LR 512 -> HR 1024 geometry) ---
                io.Int.Input("shape_steps", default=12, min=1, max=64, optional=True,
                    tooltip="Shape SLat sampler steps -- diffusion iterations for the geometry latent "
                            "(LR 512 and HR 1024 passes). More = finer surface detail, slower."),
                io.Float.Input("shape_guidance", default=7.5, min=0.0, max=15.0, step=0.1, optional=True,
                    tooltip="Shape classifier-free guidance strength: how strongly the geometry follows "
                            "the image. Higher = more faithful detail, risk of artifacts; lower = smoother."),
                io.Float.Input("shape_rescale", default=0.5, min=0.0, max=1.0, step=0.05, optional=True,
                    tooltip="Shape guidance rescale (0-1) -- tames over-strong shape guidance. "
                            "0 = off, ~0.5 = moderate."),
                io.Float.Input("shape_rescale_t", default=3.0, min=0.0, max=10.0, step=0.1, optional=True,
                    tooltip="Timestep threshold above which shape_rescale is applied. "
                            "Higher = rescale active over more of the schedule."),

                # --- Stage 4: texture (PBR voxel attrs); ignored if generate_texture is off ---
                io.Int.Input("tex_steps", default=12, min=1, max=64, optional=True,
                    tooltip="Texture SLat sampler steps -- diffusion iterations for the PBR texture "
                            "voxels. More = sharper texture, slower. Ignored when generate_texture is OFF."),
                io.Float.Input("tex_guidance", default=1.0, min=0.0, max=15.0, step=0.1, optional=True,
                    tooltip="Texture classifier-free guidance strength. Texture needs far less guidance "
                            "than geometry (default 1.0); raising it tends to oversaturate colors. "
                            "Ignored when generate_texture is OFF."),
                io.Float.Input("tex_rescale", default=0.0, min=0.0, max=1.0, step=0.05, optional=True,
                    tooltip="Texture guidance rescale (0-1). Default 0 (off) since tex_guidance is already "
                            "low. Ignored when generate_texture is OFF."),
                io.Float.Input("tex_rescale_t", default=3.0, min=0.0, max=10.0, step=0.1, optional=True,
                    tooltip="Timestep threshold above which tex_rescale is applied. "
                            "Ignored when generate_texture is OFF."),
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
        generate_texture: bool = True,
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
                generate_texture=generate_texture,
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


class Pixal3DGenerateMeshIsometric(io.ComfyNode):
    """Pixal3D Generate Mesh tuned for isometric / axonometric (parallel-projection) inputs.

    Pixal3D's view-aligned projection conditioning (ProjGrid) assumes a *perspective*
    camera (object in [-1,1] at distance 2 -> ~3x front/back foreshortening). An
    isometric drawing has ~1x (parallel projection, camera at infinity), so under the
    default camera the projected features land at the wrong pixels.

    This node applies a 'long-lens' transform to the camera: push the camera back by
    `iso_distance_multiplier` and zoom in (lower the FOV) by the same factor so the
    object keeps the same on-screen size while the foreshortening flattens toward
    orthographic. multiplier=1 is the original perspective camera; large values
    approach a true orthographic / isometric projection. Everything else is identical
    to Pixal3D Generate Mesh.
    """

    @classmethod
    def define_schema(cls):
        base = Pixal3DGenerateMesh.define_schema()
        inputs = list(base.inputs)
        iso_knob = io.Float.Input(
            "iso_distance_multiplier", default=8.0, min=1.0, max=1000.0, step=0.5, optional=True,
            tooltip="Flattens the projective conditioning toward orthographic for isometric/"
                    "axonometric inputs. 1.0 = original perspective camera; higher pushes the "
                    "camera back + zooms in (same framing, less foreshortening); very high "
                    "(~100+) approaches true orthographic. The object's on-screen size is "
                    "preserved automatically. Sweep this if alignment looks off on flat CAD views.")
        # place the knob right after the 'camera' input (index 2)
        cam_idx = next((i for i, inp in enumerate(inputs) if getattr(inp, "id", None) == "camera"), 2)
        inputs.insert(cam_idx + 1, iso_knob)
        return io.Schema(
            node_id="Pixal3DGenerateMeshIsometric",
            display_name="Pixal3D Generate Mesh isometric",
            category="Pixal3D",
            description=(
                "Same four-stage cascade as Pixal3D Generate Mesh, but flattens the camera "
                "toward orthographic so the projective conditioning matches isometric / "
                "axonometric (parallel-projection) inputs. Tune iso_distance_multiplier."
            ),
            inputs=inputs,
            outputs=list(base.outputs) + [
                io.Custom("PIXAL3D_CAMERA").Output(display_name="camera"),
            ],
        )

    @classmethod
    def execute(
        cls,
        pipeline,
        image,
        camera,
        iso_distance_multiplier: float = 8.0,
        seed: int = 42,
        generate_texture: bool = True,
        max_num_tokens: int = 49152,
        ss_steps: int = 12, ss_guidance: float = 7.5, ss_rescale: float = 0.7, ss_rescale_t: float = 5.0,
        shape_steps: int = 12, shape_guidance: float = 7.5, shape_rescale: float = 0.5, shape_rescale_t: float = 3.0,
        tex_steps: int = 12, tex_guidance: float = 1.0, tex_rescale: float = 0.0, tex_rescale_t: float = 3.0,
    ):
        from .stages import generate_mesh_and_voxelgrid, _YUP_TO_ZUP_ROT, _phase

        cam, m = _long_lens_camera(camera, iso_distance_multiplier)

        with _phase("Pixal3DGenerateMeshIsometric.execute"):
            tri, voxelgrid = generate_mesh_and_voxelgrid(
                image=image,
                camera_params=cam,
                seed=seed,
                generate_texture=generate_texture,
                pipeline_type=pipeline.get("pipeline_type", "1024_cascade"),
                attn_backend=pipeline.get("attn_backend", "auto"),
                max_num_tokens=max_num_tokens,
                ss_steps=ss_steps, ss_guidance=ss_guidance, ss_rescale=ss_rescale, ss_rescale_t=ss_rescale_t,
                shape_steps=shape_steps, shape_guidance=shape_guidance, shape_rescale=shape_rescale, shape_rescale_t=shape_rescale_t,
                tex_steps=tex_steps, tex_guidance=tex_guidance, tex_rescale=tex_rescale, tex_rescale_t=tex_rescale_t,
            )
            tri.apply_transform(_YUP_TO_ZUP_ROT)
            log.info(
                f"[Pixal3DGenerateMeshIsometric] mesh={len(tri.vertices)} verts / {len(tri.faces)} faces, "
                f"voxelgrid={voxelgrid['attrs'].shape[0]} voxels x{voxelgrid['attrs'].shape[1]} attrs"
            )
            # Emit the flattened camera so downstream nodes (e.g. Process Mesh Visibility
            # in perspective mode) can use the SAME view the mesh was generated under.
            return io.NodeOutput(tri, voxelgrid, cam)


def _voxels_to_surface(coords, res, mode="voxel_cubes"):
    """Boundary surface of an occupied voxel set in the [-0.5, 0.5] cube.

    voxel_cubes: blocky shell -- only faces between an occupied voxel and empty
    space (internal shared faces culled). marching_cubes: smoother watertight hull.
    """
    import numpy as np
    import trimesh

    occ = np.zeros((res, res, res), dtype=bool)
    occ[coords[:, 0], coords[:, 1], coords[:, 2]] = True
    vs = 1.0 / res
    origin = -0.5

    if mode == "marching_cubes":
        from trimesh.voxel import ops as vox_ops
        m = vox_ops.matrix_to_marching_cubes(occ, pitch=vs)
        m.apply_translation([origin, origin, origin])
        return m

    # voxel_cubes: emit exposed faces only, per the 6 axis directions
    DIRS = [
        # (neighbour-occupancy builder, 4 CCW-outward corner offsets in voxel units)
        (lambda o: np.pad(o[1:, :, :], ((0, 1), (0, 0), (0, 0))), [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]),   # +X
        (lambda o: np.pad(o[:-1, :, :], ((1, 0), (0, 0), (0, 0))), [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)]),  # -X
        (lambda o: np.pad(o[:, 1:, :], ((0, 0), (0, 1), (0, 0))), [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)]),   # +Y
        (lambda o: np.pad(o[:, :-1, :], ((0, 0), (1, 0), (0, 0))), [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]),  # -Y
        (lambda o: np.pad(o[:, :, 1:], ((0, 0), (0, 0), (0, 1))), [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]),   # +Z
        (lambda o: np.pad(o[:, :, :-1], ((0, 0), (0, 0), (1, 0))), [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)]),  # -Z
    ]
    vparts, fparts, voff = [], [], 0
    for nb_fn, offs in DIRS:
        exposed = occ & ~nb_fn(occ)
        idx = np.argwhere(exposed)
        if not len(idx):
            continue
        base = idx.astype(np.float64) * vs + origin            # (M,3) min corner
        quad = base[:, None, :] + np.asarray(offs, np.float64) * vs  # (M,4,3)
        vparts.append(quad.reshape(-1, 3))
        b = np.arange(len(idx)) * 4 + voff
        fparts.append(np.stack([b, b + 1, b + 2], 1))
        fparts.append(np.stack([b, b + 2, b + 3], 1))
        voff += len(idx) * 4
    if not vparts:
        return trimesh.Trimesh()
    return trimesh.Trimesh(vertices=np.concatenate(vparts, 0),
                           faces=np.concatenate(fparts, 0), process=True)


class Pixal3DGenerateSparse(io.ComfyNode):
    """Stage 1 only: the sparse-structure 'blockout' as a surface mesh.

    Runs just the sparse-structure stage of the cascade (projective + global
    conditioning -> a 32^3 occupancy) and returns the boundary surface of the
    occupied voxels -- the coarse volume the object lives in. Note the occupancy
    is a SURFACE shell (voxels the surface passes through), hollow inside, not a
    solid fill. Fast (one short diffusion stage); good for previewing the camera/
    conditioning before the full generate.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Pixal3DGenerateSparse",
            display_name="Pixal3D Generate Sparse",
            category="Pixal3D",
            description=(
                "Run only the sparse-structure stage and return the occupied voxels as a "
                "surface mesh (the coarse blockout volume). Hollow surface shell, not solid."
            ),
            inputs=[
                io.Custom("PIXAL3D_PIPELINE").Input("pipeline", tooltip="From Pixal3DLoadPipeline."),
                io.Image.Input("image", tooltip="Preprocessed image (from Pixal3D Preprocess Image)."),
                io.Custom("PIXAL3D_CAMERA").Input("camera", tooltip="From Pixal3DCameraFromFOV."),
                io.Int.Input("seed", default=42, min=0, max=2**31 - 1),
                io.Float.Input("iso_distance_multiplier", default=1.0, min=1.0, max=1000.0, step=0.5, optional=True,
                    tooltip="Isometric long-lens flattening (same as Generate Mesh isometric). 1.0 = the "
                            "original perspective camera; higher pushes the camera back + shrinks FOV so "
                            "the projective conditioning matches isometric/parallel inputs (~8 strong, "
                            "100+ ~orthographic)."),
                io.Combo.Input("mode", options=["voxel_cubes", "marching_cubes"], default="voxel_cubes",
                    tooltip="voxel_cubes = blocky shell (exact voxels, internal faces culled). "
                            "marching_cubes = smoother watertight hull of the same occupancy."),
                io.Int.Input("ss_steps", default=12, min=1, max=64, optional=True,
                    tooltip="Sparse-structure diffusion steps."),
                io.Float.Input("ss_guidance", default=7.5, min=0.0, max=15.0, step=0.1, optional=True),
                io.Float.Input("ss_rescale", default=0.7, min=0.0, max=1.0, step=0.05, optional=True),
                io.Float.Input("ss_rescale_t", default=5.0, min=0.0, max=10.0, step=0.1, optional=True),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="sparse_mesh"),
                io.Int.Output(display_name="voxel_count"),
                io.String.Output(display_name="summary"),
                io.Custom("PIXAL3D_CAMERA").Output(display_name="camera"),
            ],
        )

    @classmethod
    def execute(cls, pipeline, image, camera, seed=42, iso_distance_multiplier=1.0, mode="voxel_cubes",
                ss_steps=12, ss_guidance=7.5, ss_rescale=0.7, ss_rescale_t=5.0):
        from .stages import generate_sparse_structure, _YUP_TO_ZUP_ROT, _phase
        cam, m = _long_lens_camera(camera, iso_distance_multiplier)
        with _phase("Pixal3DGenerateSparse.execute"):
            coords, res = generate_sparse_structure(
                image=image, camera_params=cam, seed=seed,
                attn_backend=pipeline.get("attn_backend", "auto"),
                ss_res=32, ss_steps=ss_steps, ss_guidance=ss_guidance,
                ss_rescale=ss_rescale, ss_rescale_t=ss_rescale_t,
            )
            mesh = _voxels_to_surface(coords, res, mode=mode)
            # match Generate Mesh's frame (Y-up native -> Z-up for downstream)
            if len(mesh.vertices):
                mesh.apply_transform(_YUP_TO_ZUP_ROT)
            n = int(len(coords))
            summary = (f"Pixal3D Generate Sparse: {n} occupied voxels @ {res}^3 ({mode}, iso x{m:.1f}), "
                       f"{len(mesh.vertices)} verts / {len(mesh.faces)} faces (hollow surface shell).")
            log.info(f"[Pixal3DGenerateSparse] {summary}")
            return io.NodeOutput(mesh, n, summary, cam)


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
                io.Custom("TRIMESH").Input("trimesh",
                    tooltip="Input mesh to clean up (typically the raw DC mesh from "
                            "Pixal3D Generate Mesh)."),
                io.Boolean.Input("remesh", default=False, optional=True,
                    tooltip="Run dual-contouring remesh for cleaner, uniform topology. Slower; "
                            "usually unneeded since the cascade already produces uniform DC output."),
                io.Int.Input("remesh_resolution", default=512, min=64, max=2048, step=64, optional=True,
                    tooltip="Voxel grid resolution for the DC remesh (only used when remesh is ON). "
                            "Higher = finer topology, slower, more faces."),
                io.Float.Input("remesh_band", default=1.0, min=0.1, max=5.0, step=0.1, optional=True,
                    tooltip="Narrow-band width (in voxels) around the surface for the DC remesh "
                            "(only used when remesh is ON). Wider captures more but costs memory."),
                io.Boolean.Input("remove_inner_faces", default=False, optional=True,
                    tooltip="Only effective when remesh is ON. Drops quads whose centers fall "
                            "inside the original mesh's bulk (removes internal/hidden geometry)."),
                io.Boolean.Input("fill_holes", default=True, optional=True,
                    tooltip="Close small holes in the surface during cleanup so the mesh is "
                            "watertight before simplification."),
                io.Float.Input("fill_holes_perimeter", default=0.03, min=0.001, max=0.5, step=0.001, optional=True,
                    tooltip="Max hole perimeter to fill (fraction of mesh scale). Larger value "
                            "fills bigger holes; too large may bridge gaps you wanted to keep."),
                io.Float.Input("floater_threshold", default=1e-3, min=0.0, max=0.1, step=0.001, optional=True,
                    tooltip="Minimum area for a connected component to survive -- removes small "
                            "disconnected 'floater' islands. 0 disables floater removal."),
                io.Int.Input("target_face_count", default=200000, min=1000, max=5000000, step=1000,
                    tooltip="Target triangle count after simplification. Lower = lighter mesh, "
                            "less detail. The cleanup does a 2-pass simplify down to this count."),
                io.Boolean.Input("weld_vertices", default=True, optional=True,
                    tooltip="Merge coincident vertices after cleanup so the mesh is properly "
                            "connected (no split seams from numerical duplicates)."),
                io.Int.Input("weld_digits", default=4, min=1, max=8, optional=True,
                    tooltip="Decimal places of vertex-position rounding used when welding. "
                            "Higher = stricter (welds only very-close verts); lower = more aggressive."),
                # UV atlas mode: choosing 'skip' hides the chart_* parameters entirely.
                io.DynamicCombo.Input("uv_mode",
                    options=[
                        io.DynamicCombo.Option("unwrap", [
                            io.Float.Input("chart_cone_angle", default=90.0, min=0.0, max=359.9, step=1.0, optional=True,
                                tooltip="Max cone half-angle (deg) for grouping faces into a UV chart. "
                                        "Larger = fewer, bigger charts (more stretch); smaller = more "
                                        "charts/seams (less stretch)."),
                            io.Int.Input("chart_refine_iterations", default=0, min=0, max=10, optional=True,
                                tooltip="Local chart-boundary refinement passes. More = cleaner chart "
                                        "edges at some cost; 0 is usually fine."),
                            io.Int.Input("chart_global_iterations", default=1, min=0, max=10, optional=True,
                                tooltip="Global chart re-segmentation passes. More can improve overall "
                                        "atlas layout/packing at higher cost."),
                            io.Int.Input("chart_smooth_strength", default=1, min=0, max=10, optional=True,
                                tooltip="Smoothing strength applied to chart boundaries. Higher = "
                                        "smoother seams, can merge small charts."),
                        ]),
                        io.DynamicCombo.Option("skip", []),
                    ],
                    tooltip="UV handling. 'unwrap' = build a UV atlas (xatlas) so the mesh is ready "
                            "for Pixal3D Rasterize PBR (reveals chart settings). 'skip' = no atlas, "
                            "output geometry only (no UVs) -- faster, for retopology / non-textured flows."),
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
        uv_mode: dict = None,
    ):
        from .stages import process_mesh, _phase
        # uv_mode is a DynamicCombo dict: {"uv_mode": "unwrap"|"skip", + chart_* when "unwrap"}.
        uv_mode = uv_mode or {}
        unwrap_uv = uv_mode.get("uv_mode", "unwrap") != "skip"
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
                unwrap_uv=unwrap_uv,
                chart_cone_angle=uv_mode.get("chart_cone_angle", 90.0),
                chart_refine_iterations=uv_mode.get("chart_refine_iterations", 0),
                chart_global_iterations=uv_mode.get("chart_global_iterations", 1),
                chart_smooth_strength=uv_mode.get("chart_smooth_strength", 1),
            )
            return io.NodeOutput(out)


class Pixal3DProcessMeshVisibility(io.ComfyNode):
    """Pixal3D Process Mesh + per-face visibility tagging.

    Runs the same cleanup/UV as Pixal3D Process Mesh, then determines, for the
    processed mesh, which triangles are actually seen from the generation view --
    i.e. which faces a camera ray (pixel) reaches first, discounting transparency.
    The Z-up mesh the cascade produces is already in the conditioning camera's
    frame (camera on -Y looking +Y), so 'viewed' = "associatable to an input
    pixel"; everything else is occluded/back-facing (hallucinated back side).

    Writes a 0/1 scalar field as a trimesh attribute -- face_attributes['viewed']
    and vertex_attributes['viewed'] -- so it shows up as 'face.viewed' / 'viewed'
    in GeometryPack's Preview Mesh (Dual/VTK) field viewers. Optionally tints faces
    (viewed=green / unviewed=red) for plain previews, or keeps only viewed/unviewed.

    Default visibility uses Pixal3D's actual PERSPECTIVE camera (wire the Generate
    Mesh 'camera' output in) so 'seen' is faithful to how the mesh was generated.
    An orthographic option exists as a camera-free approximation for pure isometric.
    """

    @classmethod
    def define_schema(cls):
        base = Pixal3DProcessMesh.define_schema()
        inputs = list(base.inputs)
        vis_inputs = [
            io.Boolean.Input("tag_visibility", default=True, optional=True,
                tooltip="Compute the viewed/not-viewed field. OFF = behaves exactly like "
                        "Pixal3D Process Mesh."),
            io.Combo.Input("projection", options=["perspective", "orthographic"], default="perspective",
                tooltip="perspective = pinhole rays from the camera's ACTUAL FOV/distance -- matches the "
                        "camera Pixal3D used to generate the mesh (the faithful choice; wire the Generate "
                        "Mesh 'camera' output here). orthographic = parallel rays along the view axis "
                        "(a camera-free approximation, only for pure isometric)."),
            io.Combo.Input("view_from", options=["-Y", "+Y", "-X", "+X", "-Z", "+Z"], default="-Y",
                tooltip="Which side the camera sits on, in the mesh frame. Default -Y matches Pixal3D's "
                        "conditioning camera (on -Y, looking +Y). Flip if the tinted preview looks "
                        "inside-out."),
            io.Int.Input("view_resolution", default=1024, min=64, max=4096, step=64, optional=True,
                tooltip="Ray-grid density used to probe visibility. Higher catches thin slivers but "
                        "costs more rays; 1024 is plenty for most meshes."),
            io.Boolean.Input("per_face_probe", default=True, optional=True,
                tooltip="Also cast one ray per face through its centroid, so every face is directly "
                        "tested. Eliminates the speckle holes that come from the image-grid rays "
                        "undersampling small faces / missing exact edge hits."),
            io.Boolean.Input("fill_pinholes", default=True, optional=True,
                tooltip="Mop up residual specks: flip small unviewed islands that are fully enclosed "
                        "by viewed faces (interior holes). The large unviewed back region is kept."),
            io.Int.Input("max_hole_faces", default=64, min=1, max=100000, step=1, optional=True,
                tooltip="Largest enclosed unviewed island (in faces) that fill_pinholes will close. "
                        "Bigger leaves genuine occluded pockets alone."),
            io.Boolean.Input("colorize", default=True, optional=True,
                tooltip="Tint faces for preview: viewed = green, not-viewed = red. (Sets face colors; "
                        "turn off if you'll bake PBR downstream.)"),
            io.Combo.Input("keep", options=["all", "viewed_only", "unviewed_only"], default="all",
                tooltip="all = full mesh with the field. viewed_only / unviewed_only = return just "
                        "those faces as the output mesh."),
            io.Custom("PIXAL3D_CAMERA").Input("camera", optional=True,
                tooltip="From Pixal3DCameraFromFOV. Only used by 'perspective' projection (FOV + "
                        "distance). Ignored for orthographic."),
        ]
        return io.Schema(
            node_id="Pixal3DProcessMeshVisibility",
            display_name="Pixal3D Process Mesh Visibility",
            category="Pixal3D",
            description=(
                "Pixal3D Process Mesh, plus a per-face viewed/not-viewed field marking which "
                "triangles are reachable by a camera pixel from the generation view (the rest is "
                "the occluded/back-facing hallucinated side)."
            ),
            inputs=inputs + vis_inputs,
            outputs=[
                io.Custom("TRIMESH").Output(display_name="mesh"),
                io.String.Output(display_name="summary"),
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
        uv_mode: dict = None,
        tag_visibility: bool = True,
        projection: str = "perspective",
        view_from: str = "-Y",
        view_resolution: int = 1024,
        per_face_probe: bool = True,
        fill_pinholes: bool = True,
        max_hole_faces: int = 64,
        colorize: bool = True,
        keep: str = "all",
        camera: dict = None,
    ):
        import math
        import numpy as np
        import trimesh as tm
        from .stages import process_mesh, _phase

        uv_mode = uv_mode or {}
        unwrap_uv = uv_mode.get("uv_mode", "unwrap") != "skip"
        with _phase("Pixal3DProcessMeshVisibility.execute"):
            out = process_mesh(
                trimesh,
                remesh=remesh, remesh_resolution=remesh_resolution, remesh_band=remesh_band,
                remove_inner_faces=remove_inner_faces, fill_holes=fill_holes,
                fill_holes_perimeter=fill_holes_perimeter, floater_threshold=floater_threshold,
                target_face_count=target_face_count, weld_vertices=weld_vertices, weld_digits=weld_digits,
                unwrap_uv=unwrap_uv,
                chart_cone_angle=uv_mode.get("chart_cone_angle", 90.0),
                chart_refine_iterations=uv_mode.get("chart_refine_iterations", 0),
                chart_global_iterations=uv_mode.get("chart_global_iterations", 1),
                chart_smooth_strength=uv_mode.get("chart_smooth_strength", 1),
            )

            if not tag_visibility:
                return io.NodeOutput(out, "visibility tagging off")

            nF = len(out.faces)
            b0, b1 = out.bounds
            center = (b0 + b1) * 0.5
            ext = np.maximum(b1 - b0, 1e-9)
            axis = {"X": 0, "Y": 1, "Z": 2}[view_from[1]]
            sgn = -1.0 if view_from[0] == "-" else 1.0  # camera sits on this side of `axis`
            perp = [i for i in range(3) if i != axis]
            res = int(view_resolution)

            us = np.linspace(b0[perp[0]], b1[perp[0]], res)
            vs = np.linspace(b0[perp[1]], b1[perp[1]], res)
            gu, gv = np.meshgrid(us, vs)
            n_rays = gu.size

            # --- camera geometry, shared by the image grid and the per-face probe ---
            pad = 0.05 * ext[axis] + 1e-6
            campos = None
            if projection == "perspective":
                cam_angle = float((camera or {}).get("camera_angle_x", 0.8575560450553894))
                radius = float(np.linalg.norm(ext) * 0.5)
                # Pixal3D's actual camera distance (mesh is canonical ~[-0.5,0.5] scale, so
                # distance is faithful as-is); floor only so the camera isn't inside the mesh.
                cam_dist = max(float((camera or {}).get("distance", 2.0)), radius * 1.05)
                fwd = np.zeros(3); fwd[axis] = -sgn          # looking toward the object
                up_guess = np.array([0.0, 0.0, 1.0]) if axis != 2 else np.array([0.0, 1.0, 0.0])
                right = np.cross(fwd, up_guess); right /= (np.linalg.norm(right) + 1e-9)
                up = np.cross(right, fwd)
                campos = center - fwd * cam_dist
                half = math.tan(cam_angle * 0.5)
                su, sv = np.meshgrid(np.linspace(-half, half, res), np.linspace(-half, half, res))
                dirs = (fwd[None, :] + su.ravel()[:, None] * right[None, :] + sv.ravel()[:, None] * up[None, :])
                dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9)
                origins = np.tile(campos, (n_rays, 1))
            else:
                # orthographic: parallel rays along the view axis from just outside the bbox
                origins = np.zeros((n_rays, 3))
                origins[:, perp[0]] = gu.ravel()
                origins[:, perp[1]] = gv.ravel()
                origins[:, axis] = (b1[axis] + pad) if sgn > 0 else (b0[axis] - pad)
                dirs = np.zeros((n_rays, 3))
                dirs[:, axis] = -sgn  # shoot toward the object

            hit_faces = out.ray.intersects_first(ray_origins=origins, ray_directions=dirs)
            viewed = np.zeros(nF, dtype=bool)
            seen = hit_faces[hit_faces >= 0]
            if seen.size:
                viewed[np.unique(seen)] = True
            n_grid = int(viewed.sum())

            # --- per-face centroid probe: one dedicated ray per face, so undersampling /
            #     missed exact-edge hits can't leave speckle holes inside the viewed region ---
            if per_face_probe and nF:
                cen = out.triangles_center
                if projection == "perspective":
                    pdirs = cen - campos[None, :]
                    pdirs /= (np.linalg.norm(pdirs, axis=1, keepdims=True) + 1e-9)
                    porig = np.tile(campos, (nF, 1))
                else:
                    porig = cen.copy()
                    porig[:, axis] = (b1[axis] + pad) if sgn > 0 else (b0[axis] - pad)
                    pdirs = np.zeros((nF, 3)); pdirs[:, axis] = -sgn
                pf = out.ray.intersects_first(ray_origins=porig, ray_directions=pdirs)
                viewed |= (pf == np.arange(nF))
            n_probe = int(viewed.sum())

            # --- fill small enclosed unviewed islands (residual embree pinholes) ---
            n_fill = 0
            if fill_pinholes and nF and not viewed.all():
                adj = out.face_adjacency
                if len(adj):
                    from collections import defaultdict as _dd
                    deg = np.bincount(adj.ravel(), minlength=nF)
                    boundary = deg < 3  # triangle touching an open mesh edge
                    parent = np.arange(nF)

                    def _find(x):
                        while parent[x] != x:
                            parent[x] = parent[parent[x]]
                            x = parent[x]
                        return x

                    for a, b in adj:
                        a, b = int(a), int(b)
                        if not viewed[a] and not viewed[b]:
                            ra, rb = _find(a), _find(b)
                            if ra != rb:
                                parent[ra] = rb
                    comp = _dd(list)
                    for f in np.nonzero(~viewed)[0]:
                        comp[_find(int(f))].append(int(f))
                    maxh = int(max_hole_faces)
                    for faces in comp.values():
                        if len(faces) <= maxh and not boundary[faces].any():
                            viewed[faces] = True
                            n_fill += len(faces)

            ratio = float(viewed.mean()) if nF else 0.0

            # Store as trimesh face/vertex attributes (a 0/1 scalar field) so it shows up
            # as 'face.viewed' / 'viewed' in GeometryPack's Preview Mesh (Dual/VTK) field
            # viewers; also keep the ratio in metadata.
            out.face_attributes["viewed"] = viewed.astype(np.float32)
            vviewed = np.zeros(len(out.vertices), np.float32)
            if viewed.any():
                vviewed[np.unique(out.faces[viewed])] = 1.0  # vertex viewed if any incident face is
            out.vertex_attributes["viewed"] = vviewed
            out.metadata = dict(out.metadata or {})
            out.metadata["viewed_ratio"] = ratio

            if colorize:
                fc = np.empty((nF, 4), np.uint8)
                fc[viewed] = (40, 190, 60, 255)
                fc[~viewed] = (210, 50, 40, 255)
                out.visual = tm.visual.ColorVisuals(mesh=out, face_colors=fc)

            mesh_out = out
            if keep != "all":
                want = viewed if keep == "viewed_only" else ~viewed
                idx = np.nonzero(want)[0]
                if idx.size:
                    mesh_out = out.submesh([idx], append=True)

            summary = (f"Visibility ({projection}, view {view_from}, {res}x{res} rays): "
                       f"{int(viewed.sum())}/{nF} faces viewed ({100.0 * ratio:.1f}%) "
                       f"[grid {n_grid} -> +probe {n_probe - n_grid} -> +fill {n_fill}]; keep={keep}.")
            log.info(f"[Pixal3DProcessMeshVisibility] {summary}")
            return io.NodeOutput(mesh_out, summary)


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
    "Pixal3DGenerateMeshIsometric": Pixal3DGenerateMeshIsometric,
    "Pixal3DGenerateSparse": Pixal3DGenerateSparse,
    "Pixal3DProcessMesh": Pixal3DProcessMesh,
    "Pixal3DProcessMeshVisibility": Pixal3DProcessMeshVisibility,
    "Pixal3DRasterizePBR": Pixal3DRasterizePBR,
    "Pixal3DExportGLB": Pixal3DExportGLB,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Pixal3DGenerateMesh": "Pixal3D Generate Mesh",
    "Pixal3DGenerateMeshIsometric": "Pixal3D Generate Mesh isometric",
    "Pixal3DGenerateSparse": "Pixal3D Generate Sparse",
    "Pixal3DProcessMesh": "Pixal3D Process Mesh",
    "Pixal3DProcessMeshVisibility": "Pixal3D Process Mesh Visibility",
    "Pixal3DRasterizePBR": "Pixal3D Rasterize PBR",
    "Pixal3DExportGLB": "Pixal3D Export GLB",
}
