"""Blender backend for camera placement, Cycles beauty / AOV rendering, BSDF probing.

Public functions mirror ``src/render.py`` (``render_beauty``, ``render_aov``,
``render_depth_for_filter``, ``derive_material_lut``) so the orchestrator can
swap backends with a one-line import change. ``AOVImages`` is re-exported from
``src/render.py`` so downstream code is untouched.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import imageio.v2 as imageio
import numpy as np
from numpy.typing import NDArray

from src.camera_sampling import CameraPose
from src.io_utils import Intrinsics
from src.pose_utils import pose_to_c2w
from src.render import AOVImages  # re-used: identical shape contract
from src.scene_blender import _import_bpy

if TYPE_CHECKING:  # pragma: no cover
    import bpy  # noqa: F401

log = logging.getLogger(__name__)


# OpenCV (+Z forward, +Y up in pose_to_c2w's convention) -> Blender camera
# convention (-Z forward, +Y up). Right-multiplying by this flips the camera's
# Z basis without touching its X/Y, so the same world-space look direction is
# preserved.
_OPENCV_TO_BLENDER_CAM = np.diag([1.0, 1.0, -1.0, 1.0])


# ---- Camera placement -------------------------------------------------------


def _ensure_camera(bpy: Any) -> Any:
    cam_obj = bpy.data.objects.get("PBRCaptureCamera")
    if cam_obj is None or cam_obj.type != "CAMERA":
        cam_data = bpy.data.cameras.new("PBRCaptureCamera")
        cam_obj = bpy.data.objects.new("PBRCaptureCamera", cam_data)
        bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    return cam_obj


def set_camera(pose: CameraPose, intrinsics: Intrinsics) -> Any:
    """Position + orient the active Blender camera to match ``pose``.

    Also sets the focal length and sensor fit from the pinhole intrinsics so
    the rendered image matches what the Mitsuba sensor would produce.
    """
    bpy = _import_bpy()
    cam_obj = _ensure_camera(bpy)

    c2w_cv = pose_to_c2w(pose)
    c2w_bl = c2w_cv @ _OPENCV_TO_BLENDER_CAM

    import mathutils  # type: ignore[import-not-found]
    cam_obj.matrix_world = mathutils.Matrix(c2w_bl.tolist())

    cam_data = cam_obj.data
    cam_data.type = "PERSP"
    cam_data.lens_unit = "FOV"
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.angle = float(intrinsics.fov_x_rad)
    cam_data.shift_x = 0.0
    cam_data.shift_y = 0.0
    cam_data.clip_start = 0.001
    cam_data.clip_end = 1000.0
    return cam_obj


# ---- Render configuration ---------------------------------------------------


def _set_resolution(bpy: Any, width: int, height: int) -> None:
    r = bpy.context.scene.render
    r.resolution_x = int(width)
    r.resolution_y = int(height)
    r.resolution_percentage = 100


def _set_samples(bpy: Any, spp: int, max_depth: int, *, denoise: bool) -> None:
    cy = bpy.context.scene.cycles
    cy.samples = int(spp)
    cy.use_adaptive_sampling = False
    cy.max_bounces = int(max_depth)
    cy.diffuse_bounces = int(max_depth)
    cy.glossy_bounces = int(max_depth)
    cy.transmission_bounces = int(max_depth)
    cy.transparent_max_bounces = int(max_depth)
    vl = bpy.context.scene.view_layers[0]
    vl.cycles.use_denoising = bool(denoise)


def _set_passes(
    bpy: Any,
    *,
    beauty: bool,
    depth: bool,
    normal: bool,
    albedo: bool,
    object_index: bool,
    glossy_color: bool = False,
    emission: bool = False,
    material_index: bool = False,
) -> None:
    vl = bpy.context.scene.view_layers[0]
    # Always keep the Combined pass on: Cycles 4.2 won't actually trace any
    # rays when use_pass_combined is False, which causes all derived passes
    # (Z / Normal / Diffuse Color / ObjectIndex) to come back as zeros.
    # ``beauty`` no longer controls this; it's a no-op flag kept for the
    # legacy signature. The combined output simply isn't wired to a file
    # output in the compositor when ``beauty`` is False.
    _ = beauty
    vl.use_pass_combined = True
    vl.use_pass_z = bool(depth)
    vl.use_pass_normal = bool(normal)
    vl.use_pass_diffuse_color = bool(albedo)
    vl.use_pass_object_index = bool(object_index)
    vl.use_pass_glossy_color = bool(glossy_color)
    vl.use_pass_emit = bool(emission)
    vl.use_pass_material_index = bool(material_index)


# ---- Shader-level AOVs (per-pixel material parameters) ----------------------
#
# Cycles' built-in passes give us color-domain quantities (Diffuse Color,
# Glossy Color, Emit) but not arbitrary BSDF inputs like roughness or
# metallic. To capture those per-pixel, we wire a ShaderNodeOutputAOV into
# every Principled-BSDF material, sourcing the value from the BSDF's
# Roughness / Metallic socket. The view-layer must also declare an AOV slot
# matching the output node's name. This mirrors FIPT's approach in
# class_renderer_blender_mitsubaScene_3D.py.

_SHADER_AOV_NAMES: tuple[str, ...] = ("PixelRoughness", "PixelMetallic")
# Map the AOV name to the Principled-BSDF input it samples.
_SHADER_AOV_INPUT: dict[str, str] = {
    "PixelRoughness": "Roughness",
    "PixelMetallic": "Metallic",
}


def _register_view_layer_aovs(bpy: Any) -> None:
    """Ensure the view layer declares each shader AOV exactly once."""
    vl = bpy.context.scene.view_layers[0]
    existing = {a.name for a in vl.aovs}
    for name in _SHADER_AOV_NAMES:
        if name in existing:
            continue
        bpy.ops.scene.view_layer_add_aov()
        vl.aovs[-1].name = name
        vl.aovs[-1].type = "VALUE"


def _wire_shader_aov_for_material(mat: Any, aov_name: str, input_name: str) -> None:
    """Add a ShaderNodeOutputAOV to ``mat`` sourcing from a Principled input.

    No-op if the material has no Principled BSDF, or if a same-named AOV
    output already exists (so this function is idempotent across re-renders).
    """
    if mat is None or not getattr(mat, "use_nodes", False):
        return
    tree = mat.node_tree
    if tree is None:
        return

    for n in tree.nodes:
        if n.bl_idname == "ShaderNodeOutputAOV" and getattr(n, "name", "") == aov_name:
            return  # already wired

    principled = None
    for n in tree.nodes:
        if n.bl_idname == "ShaderNodeBsdfPrincipled":
            principled = n
            break
    if principled is None:
        return

    socket = principled.inputs.get(input_name)
    if socket is None:
        return

    aov_node = tree.nodes.new("ShaderNodeOutputAOV")
    aov_node.name = aov_name

    if socket.is_linked:
        src_socket = socket.links[0].from_socket
    else:
        # Materialise the default scalar value into a Value node so it can
        # drive the AOV output (sockets without links can't be wired directly).
        val = tree.nodes.new("ShaderNodeValue")
        val.outputs[0].default_value = float(socket.default_value)
        src_socket = val.outputs[0]
    tree.links.new(src_socket, aov_node.inputs["Value"])


def setup_shader_aovs(bpy: Any) -> None:
    """Register shader AOVs at the view-layer level and wire them per material.

    Idempotent: safe to call before every render even though the per-frame
    cost is tiny. Materials added later (e.g. by re-importing the scene) are
    picked up automatically on the next call.
    """
    _register_view_layer_aovs(bpy)
    for mat in bpy.data.materials:
        for aov_name in _SHADER_AOV_NAMES:
            _wire_shader_aov_for_material(mat, aov_name, _SHADER_AOV_INPUT[aov_name])


# ---- Compositor setup -------------------------------------------------------


def _clear_compositor(bpy: Any) -> Any:
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree
    for n in list(tree.nodes):
        tree.nodes.remove(n)
    return tree


def _make_file_output(tree: Any, label: str, out_dir: Path,
                      data_type: str = "OPEN_EXR") -> Any:
    """Create a File Output node that writes a single .exr.

    Caller wires inputs and sets ``file_slots[i].path`` for the leaf name.
    Output filename will be ``<path><frame:04d>.exr`` — we fix
    ``scene.frame_current = 1`` so it always lands at ``<path>0001.exr``.

    Notes on color_mode='RGB': Cycles 4.2 emits single-channel (value)
    passes (Depth, IndexOB, etc.) with channel name 'V' in the EXR. The
    FreeImage reader we use elsewhere ignores those V channels and
    returns zeros. Forcing 'RGB' makes Blender broadcast the single value
    into all three channels, so FreeImage can read them. The caller
    treats the first channel as the value.
    """
    n = tree.nodes.new("CompositorNodeOutputFile")
    n.label = label
    n.base_path = str(out_dir)
    n.format.file_format = data_type
    if data_type == "OPEN_EXR":
        n.format.color_depth = "32"
        n.format.exr_codec = "ZIP"
        n.format.color_mode = "RGB"
    return n


def _setup_compositor_for_aov(
    bpy: Any, out_dir: Path, *,
    write_beauty: bool, write_depth: bool, write_normal: bool,
    write_albedo: bool, write_object_index: bool,
    write_glossy_color: bool = False, write_emission: bool = False,
    write_material_index: bool = False,
    write_shader_aov_roughness: bool = False,
    write_shader_aov_metallic: bool = False,
) -> dict[str, Path]:
    """Build a compositor pipeline writing each enabled pass to its own EXR.

    Returns a dict mapping pass name -> output file path (filled in by Blender).
    """
    tree = _clear_compositor(bpy)
    rl = tree.nodes.new("CompositorNodeRLayers")
    rl.location = (-400, 0)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    def _broadcast_to_rgb(value_socket: Any) -> Any:
        """Route a single-value socket through a CombineColor->RGB node.

        Cycles 4.2 emits single-channel passes (Depth, IndexOB) with channel
        name 'V' in the EXR, which the FreeImage reader returns as zeros.
        Broadcasting the value into R/G/B before the file output makes the
        EXR a conventional 3-channel RGB float file the reader handles.
        """
        cc = tree.nodes.new("CompositorNodeCombineColor")
        cc.mode = "RGB"
        for inp in ("Red", "Green", "Blue"):
            tree.links.new(value_socket, cc.inputs[inp])
        return cc.outputs[0]

    def _vector_to_rgb(vector_socket: Any) -> Any:
        """Relabel a Vector socket (X/Y/Z) as RGB.

        Cycles 4.2 writes the Normal pass with channel names X/Y/Z which
        FreeImage refuses to load ("Unsupported color model: X/Y/Z").
        Splitting into scalars and rebuilding as a Color socket forces the
        file output to use R/G/B channel names instead.
        """
        sep = tree.nodes.new("CompositorNodeSeparateXYZ")
        tree.links.new(vector_socket, sep.inputs[0])
        cc = tree.nodes.new("CompositorNodeCombineColor")
        cc.mode = "RGB"
        tree.links.new(sep.outputs["X"], cc.inputs["Red"])
        tree.links.new(sep.outputs["Y"], cc.inputs["Green"])
        tree.links.new(sep.outputs["Z"], cc.inputs["Blue"])
        return cc.outputs[0]

    def add(name: str, slot_in_rl: str, prefix: str, *,
            broadcast: bool = False, vector: bool = False) -> None:
        fo = _make_file_output(tree, name, out_dir)
        fo.file_slots[0].path = prefix
        src_socket = rl.outputs[slot_in_rl]
        if broadcast:
            src_socket = _broadcast_to_rgb(src_socket)
        elif vector:
            src_socket = _vector_to_rgb(src_socket)
        tree.links.new(src_socket, fo.inputs[0])
        paths[name] = out_dir / f"{prefix}0001.exr"

    if write_beauty:
        add("beauty", "Image", "beauty_")
    if write_depth:
        add("depth", "Depth", "depth_", broadcast=True)
    if write_normal:
        add("normal", "Normal", "normal_", vector=True)
    if write_albedo:
        # Cycles "Diffuse Color" pass is named "DiffCol" on the Render Layers node.
        add("albedo", "DiffCol", "albedo_")
    if write_object_index:
        add("object_index", "IndexOB", "objidx_", broadcast=True)
    if write_glossy_color:
        add("glossy_color", "GlossCol", "gloss_")
    if write_emission:
        add("emission", "Emit", "emit_")
    if write_material_index:
        add("material_index", "IndexMA", "matidx_", broadcast=True)
    if write_shader_aov_roughness:
        # Shader AOVs appear on the RLayers node under their declared name.
        add("shader_roughness", "PixelRoughness", "rough_", broadcast=True)
    if write_shader_aov_metallic:
        add("shader_metallic", "PixelMetallic", "metal_", broadcast=True)

    bpy.context.scene.frame_current = 1
    return paths


def _read_exr(path: Path) -> NDArray[np.float32]:
    if not path.exists():
        raise FileNotFoundError(f"Expected render output not written: {path}")
    # Force the FreeImage EXR plugin. Without this, imageio sometimes picks
    # the wrong format (e.g. its 'SPE' Princeton Instruments plugin) and
    # raises spurious "Index 0 exceeds number of frames" errors.
    img = np.asarray(
        imageio.imread(str(path), format="EXR-FI"),  # type: ignore[arg-type]
        dtype=np.float32,
    )
    return img


# ---- Z (camera-space) -> ray.t conversion -----------------------------------


def _ray_t_from_z(z: NDArray[np.float32], intrinsics: Intrinsics) -> NDArray[np.float32]:
    """Convert a Cycles Z pass (camera-space perpendicular depth) to ray distance."""
    h, w = z.shape[:2]
    px = np.arange(w, dtype=np.float32)
    py = np.arange(h, dtype=np.float32)
    u = (px[None, :] - float(intrinsics.cx)) / float(intrinsics.fl_x)
    v = (py[:, None] - float(intrinsics.cy)) / float(intrinsics.fl_y)
    scale = np.sqrt(1.0 + u * u + v * v).astype(np.float32)
    out = z * scale
    # Cycles writes 1e10 (or similar large value) for "no hit". Treat them as +inf
    # for consistency with the Mitsuba pipeline.
    out = np.where(np.isfinite(z) & (z < 1e9), out, np.float32(np.inf))
    return out


# ---- Top-level render entry points ------------------------------------------


def _render_now(bpy: Any, *, seed: int) -> None:
    bpy.context.scene.cycles.seed = int(seed)
    # We rely on the compositor file-output nodes to write the data we need,
    # not on scene.render.filepath. Render without writing the combined image
    # to scene.render.filepath.
    bpy.ops.render.render(write_still=False)


def render_beauty_blender(
    pose: CameraPose,
    intrinsics: Intrinsics,
    spp: int,
    max_depth: int,
    seed: int = 0,
) -> NDArray[np.float32]:
    """Cycles HDR linear-radiance RGB. Returns ``(H, W, 3)`` float32."""
    bpy = _import_bpy()
    set_camera(pose, intrinsics)
    _set_resolution(bpy, intrinsics.width, intrinsics.height)
    _set_samples(bpy, spp=spp, max_depth=max_depth, denoise=True)
    _set_passes(bpy, beauty=True, depth=False, normal=False,
                albedo=False, object_index=False)
    with tempfile.TemporaryDirectory(prefix="pbr_blender_beauty_") as td:
        paths = _setup_compositor_for_aov(
            bpy, Path(td),
            write_beauty=True, write_depth=False, write_normal=False,
            write_albedo=False, write_object_index=False,
        )
        _render_now(bpy, seed=seed)
        img = _read_exr(paths["beauty"])
    if img.ndim == 3 and img.shape[-1] >= 3:
        return np.ascontiguousarray(img[..., :3], dtype=np.float32)
    raise RuntimeError(f"Unexpected beauty image shape {img.shape}")


def _to_value_channel(arr: NDArray[np.float32]) -> NDArray[np.float32]:
    """Collapse the broadcast-RGB EXR back to a single channel."""
    if arr.ndim == 3:
        return np.ascontiguousarray(arr[..., 0], dtype=np.float32)
    return np.ascontiguousarray(arr, dtype=np.float32)


def render_aov_blender(
    pose: CameraPose,
    intrinsics: Intrinsics,
    spp: int,
    max_depth: int,
    seed: int = 0,
) -> AOVImages:
    """Render the full AOV stack for one camera.

    Returns an :class:`AOVImages` populated with depth, world-space normal,
    diffuse color (k_d), object-index segmentation, and (Blender-only)
    glossy color, emission, material-index segmentation, and per-pixel
    roughness + metallic via shader AOVs.
    """
    bpy = _import_bpy()
    set_camera(pose, intrinsics)
    _set_resolution(bpy, intrinsics.width, intrinsics.height)
    # AOVs don't need denoising — they're per-pixel deterministic geometry.
    _set_samples(bpy, spp=spp, max_depth=max_depth, denoise=False)
    _set_passes(
        bpy,
        beauty=False,
        depth=True, normal=True, albedo=True, object_index=True,
        glossy_color=True, emission=True, material_index=True,
    )
    setup_shader_aovs(bpy)

    with tempfile.TemporaryDirectory(prefix="pbr_blender_aov_") as td:
        paths = _setup_compositor_for_aov(
            bpy, Path(td),
            write_beauty=False, write_depth=True, write_normal=True,
            write_albedo=True, write_object_index=True,
            write_glossy_color=True, write_emission=True,
            write_material_index=True,
            write_shader_aov_roughness=True, write_shader_aov_metallic=True,
        )
        _render_now(bpy, seed=seed + 1)
        z = _to_value_channel(_read_exr(paths["depth"]))
        normal = _read_exr(paths["normal"])
        albedo = _read_exr(paths["albedo"])
        obj_idx = _to_value_channel(_read_exr(paths["object_index"]))
        gloss = _read_exr(paths["glossy_color"])
        emit = _read_exr(paths["emission"])
        mat_idx = _to_value_channel(_read_exr(paths["material_index"]))
        rough_px = _to_value_channel(_read_exr(paths["shader_roughness"]))
        metal_px = _to_value_channel(_read_exr(paths["shader_metallic"]))

    depth_t = _ray_t_from_z(z, intrinsics)
    if normal.ndim != 3 or normal.shape[-1] < 3:
        raise RuntimeError(f"Unexpected normal shape {normal.shape}")
    normal = np.ascontiguousarray(normal[..., :3], dtype=np.float32)
    if albedo.ndim != 3 or albedo.shape[-1] < 3:
        raise RuntimeError(f"Unexpected albedo shape {albedo.shape}")
    albedo = np.ascontiguousarray(albedo[..., :3], dtype=np.float32)
    if gloss.ndim != 3 or gloss.shape[-1] < 3:
        raise RuntimeError(f"Unexpected glossy-color shape {gloss.shape}")
    gloss = np.ascontiguousarray(gloss[..., :3], dtype=np.float32)
    if emit.ndim != 3 or emit.shape[-1] < 3:
        raise RuntimeError(f"Unexpected emission shape {emit.shape}")
    emit = np.ascontiguousarray(emit[..., :3], dtype=np.float32)

    shape_index = np.rint(obj_idx).astype(np.int32)
    material_index = np.rint(mat_idx).astype(np.int32)

    return AOVImages(
        depth=depth_t,
        normal=normal,
        albedo=albedo,
        shape_index=shape_index,
        glossy_color=gloss,
        emission=emit,
        material_index=material_index,
        roughness_per_pixel=rough_px,
        metallic_per_pixel=metal_px,
    )


def render_depth_for_filter_blender(
    pose: CameraPose,
    intrinsics: Intrinsics,
    spp: int,
    max_depth: int,
    seed: int = 0,
) -> NDArray[np.float32]:
    """Fast low-resolution depth render used by the camera-validity filter."""
    bpy = _import_bpy()
    set_camera(pose, intrinsics)
    _set_resolution(bpy, intrinsics.width, intrinsics.height)
    _set_samples(bpy, spp=spp, max_depth=max_depth, denoise=False)
    _set_passes(bpy, beauty=False, depth=True, normal=False,
                albedo=False, object_index=False)
    with tempfile.TemporaryDirectory(prefix="pbr_blender_filter_") as td:
        paths = _setup_compositor_for_aov(
            bpy, Path(td),
            write_beauty=False, write_depth=True, write_normal=False,
            write_albedo=False, write_object_index=False,
        )
        _render_now(bpy, seed=seed + 2)
        z = _read_exr(paths["depth"])
        if z.ndim == 3:
            z = z[..., 0]
    return _ray_t_from_z(z.astype(np.float32), intrinsics)


# ---- Material LUT -----------------------------------------------------------


def _principled_inputs(obj: Any) -> tuple[float, float]:
    """Return (roughness, metallic) from the first Principled BSDF in obj's material.

    Falls back to (1.0, 0.0) — same default as the Mitsuba ``diffuse`` fallback —
    when no Principled BSDF is found.
    """
    if not obj.material_slots:
        return 1.0, 0.0
    mat = obj.material_slots[0].material
    if mat is None or not mat.use_nodes or mat.node_tree is None:
        return 1.0, 0.0
    for node in mat.node_tree.nodes:
        if node.bl_idname == "ShaderNodeBsdfPrincipled":
            try:
                r = float(node.inputs["Roughness"].default_value)
            except (KeyError, TypeError):
                r = 0.5
            try:
                m = float(node.inputs["Metallic"].default_value)
            except (KeyError, TypeError):
                m = 0.0
            return r, m
    return 1.0, 0.0


def derive_material_lut_blender(
    pass_index_to_object: dict[int, Any],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Build per-pass-index (roughness, metallic) LUTs from Principled BSDFs.

    Returned arrays have length ``max_pass_index + 1`` so they index directly
    into the object-index AOV. Index 0 is the "no hit" sentinel (NaN).
    """
    if not pass_index_to_object:
        return (
            np.full(1, np.nan, dtype=np.float32),
            np.full(1, np.nan, dtype=np.float32),
        )
    n = max(pass_index_to_object.keys()) + 1
    rough = np.full(n, np.nan, dtype=np.float32)
    metal = np.full(n, np.nan, dtype=np.float32)
    for idx, obj in pass_index_to_object.items():
        r, m = _principled_inputs(obj)
        rough[idx] = float(r)
        metal[idx] = float(m)
    return rough, metal
