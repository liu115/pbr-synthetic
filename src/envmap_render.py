"""Per-pixel environment-map rendering via Blender Cycles.

Captures a spatially-varying lighting field for one frame by laying an
``env_row × env_col`` grid of "anchor" pixels over the image plane (with
``env_row = ceil(H / patch_size)`` and ``env_col = ceil(W / patch_size)``),
back-projecting each anchor to a world-space surface point using the
already-rendered depth + normal AOVs, and rendering a small equirectangular
panorama from each surface point.

All panoramas for one frame are tiled into a single EXR of shape
``(env_row * env_height, env_col * env_width, 3)``. Smaller patch sizes give
denser spatial coverage; smaller per-envmap resolution shrinks storage. Both
are flags on the unified CLI.

Output convention: each envmap's local frame has the surface normal as +Y, so
the top row of the equirectangular image maps to the direction straight away
from the surface (the "north pole" of the hemisphere). With ``latitude_min =
0`` set on the panoramic Cycles camera, only the upper hemisphere is sampled
— light coming from below the surface is unreachable.

Follows FIPT's ``render_lighting_envmap`` (class_renderer_blender_mitsuba
Scene_3D.py) but uses a per-cell render loop instead of Blender's multiview
machinery. Slower per frame but much easier to reason about; can be migrated
to multiview as a follow-up if rendering becomes the bottleneck.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from src.camera_sampling import CameraPose
from src.io_utils import Intrinsics
from src.pose_utils import pose_to_c2w
from src.render_blender import (
    _clear_compositor,
    _make_file_output,
    _read_exr,
    _render_now,
    _set_passes,
    _set_resolution,
    _set_samples,
)
from src.scene_blender import _import_bpy

log = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class EnvmapConfig:
    """Resolution + grid spacing for per-pixel envmap rendering.

    ``patch_size`` is the side length (in pixels of the source image) of one
    grid cell. ``env_row`` and ``env_col`` are derived per-frame from the
    image resolution. ``env_height`` × ``env_width`` is the panorama
    resolution at each grid cell.

    ``spp`` controls Cycles samples for the envmap renders specifically (the
    main beauty/AOV render uses its own SPP). Envmaps are small + denoised,
    so a low value is usually plenty.
    """

    patch_size: int = 40
    env_height: int = 64
    env_width: int = 128
    spp: int = 32
    max_depth: int = 4

    def grid_shape(self, image_height: int, image_width: int) -> tuple[int, int]:
        env_row = max(1, image_height // self.patch_size)
        env_col = max(1, image_width // self.patch_size)
        return env_row, env_col

    def mosaic_shape(self, image_height: int, image_width: int) -> tuple[int, int]:
        env_row, env_col = self.grid_shape(image_height, image_width)
        return env_row * self.env_height, env_col * self.env_width


# ---- Back-projection ---------------------------------------------------------


def _backproject_pixel(
    px: float, py: float,
    depth_t: float,
    intrinsics: Intrinsics,
    c2w: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Back-project image pixel ``(px, py)`` with ray-distance ``depth_t`` to world space.

    ``depth_t`` is the ray.t convention used everywhere else in this repo:
    distance from camera origin along the unit ray through the pixel. ``c2w``
    is the OpenCV-convention camera-to-world matrix from :func:`pose_to_c2w`.
    """
    u = (px - intrinsics.cx) / intrinsics.fl_x
    v = (py - intrinsics.cy) / intrinsics.fl_y
    # OpenCV camera convention: +Z forward, +Y down (we use it consistently
    # in transforms.json and pose_to_c2w).
    ray_cam = np.array([u, v, 1.0], dtype=np.float64)
    ray_cam /= np.linalg.norm(ray_cam)
    world_dir = c2w[:3, :3] @ ray_cam
    origin = c2w[:3, 3]
    return origin + depth_t * world_dir


def _local_frame_from_normal(
    normal: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Build a 3x3 rotation matrix whose +Y column is ``normal``.

    Columns are (right, up=normal, back). The right axis is cross(normal,
    world_up); when normal is parallel to world_up we fall back to world_x.
    """
    n = normal / max(float(np.linalg.norm(normal)), 1e-9)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(n, world_up))) > 0.99:
        world_up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    right = np.cross(n, world_up)
    right /= max(float(np.linalg.norm(right)), 1e-9)
    back = np.cross(right, n)
    back /= max(float(np.linalg.norm(back)), 1e-9)
    R = np.stack([right, n, back], axis=1)
    return cast(NDArray[np.float64], R.astype(np.float64, copy=False))


def _world_matrix_for_envmap(
    point: NDArray[np.float64], normal: NDArray[np.float64], offset: float = 1e-3
) -> NDArray[np.float64]:
    """4x4 world matrix for a panoramic camera at ``point`` looking along ``normal``."""
    R = _local_frame_from_normal(normal)
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = R
    # Push slightly above the surface so the panorama doesn't render the
    # underside of geometry due to floating-point self-intersection.
    m[:3, 3] = point + normal * offset
    return m


# ---- Blender configuration --------------------------------------------------


def _configure_panoramic_camera(
    bpy: Any, cam_obj: Any, world_matrix: NDArray[np.float64],
    env_height: int, env_width: int,
) -> None:
    cam_data = cam_obj.data
    cam_data.type = "PANO"
    # bpy 4.x moved the panoramic Cycles settings from cam_data.cycles to the
    # Camera data block directly. ``latitude_min = 0`` restricts to the upper
    # hemisphere (above the surface).
    cam_data.panorama_type = "EQUIRECTANGULAR"
    cam_data.latitude_min = 0.0
    cam_data.latitude_max = float(np.pi / 2.0)
    cam_data.longitude_min = float(-np.pi)
    cam_data.longitude_max = float(np.pi)
    cam_data.clip_start = 1e-4
    cam_data.clip_end = 1000.0

    import mathutils  # type: ignore[import-not-found]
    cam_obj.matrix_world = mathutils.Matrix(world_matrix.tolist())

    bpy.context.scene.camera = cam_obj
    bpy.context.scene.render.resolution_x = int(env_width)
    bpy.context.scene.render.resolution_y = int(env_height)
    bpy.context.scene.render.resolution_percentage = 100


def _ensure_envmap_camera(bpy: Any) -> Any:
    name = "PBRCaptureEnvmapCamera"
    cam_obj = bpy.data.objects.get(name)
    if cam_obj is None or cam_obj.type != "CAMERA":
        cam_data = bpy.data.cameras.new(name)
        cam_obj = bpy.data.objects.new(name, cam_data)
        bpy.context.collection.objects.link(cam_obj)
    return cam_obj


def _setup_compositor_for_envmap(bpy: Any, out_dir: Path) -> Path:
    """Wire RLayers.Image → File Output (RGB OpenEXR) into ``out_dir/env_0001.exr``."""
    tree = _clear_compositor(bpy)
    rl = tree.nodes.new("CompositorNodeRLayers")
    out_dir.mkdir(parents=True, exist_ok=True)
    fo = _make_file_output(tree, "envmap", out_dir)
    fo.file_slots[0].path = "env_"
    tree.links.new(rl.outputs["Image"], fo.inputs[0])
    bpy.context.scene.frame_current = 1
    return out_dir / "env_0001.exr"


# ---- Top-level entry point --------------------------------------------------


def render_envmaps_for_frame(
    pose: CameraPose,
    intrinsics: Intrinsics,
    depth: NDArray[np.float32],
    normal: NDArray[np.float32],
    cfg: EnvmapConfig,
    seed: int = 0,
) -> NDArray[np.float32]:
    """Render the SV envmap grid for one camera frame.

    ``depth`` is the ray-t map (meters from camera origin) and ``normal`` is
    the world-space surface normal — both straight from
    :func:`src.render_blender.render_aov_blender`. Pixels with non-finite
    depth (no surface hit) fall back to rendering the envmap at the camera
    origin so the mosaic stays a valid, finite tensor.

    Returns a tiled mosaic of shape
    ``(env_row * cfg.env_height, env_col * cfg.env_width, 3)`` in linear
    HDR radiance.
    """
    if depth.ndim != 2:
        raise ValueError(f"depth must be 2D; got {depth.shape}")
    if normal.ndim != 3 or normal.shape[-1] < 3:
        raise ValueError(f"normal must be (H, W, 3); got {normal.shape}")
    h, w = depth.shape
    env_row, env_col = cfg.grid_shape(h, w)
    mosaic = np.zeros(
        (env_row * cfg.env_height, env_col * cfg.env_width, 3),
        dtype=np.float32,
    )

    bpy = _import_bpy()
    cam_obj = _ensure_envmap_camera(bpy)
    # Disable every AOV pass — envmap renders only need the Combined image.
    _set_passes(
        bpy, beauty=True, depth=False, normal=False, albedo=False,
        object_index=False, glossy_color=False, emission=False,
        material_index=False,
    )
    _set_samples(bpy, spp=cfg.spp, max_depth=cfg.max_depth, denoise=True)

    c2w = pose_to_c2w(pose).astype(np.float64)
    cam_origin = np.asarray(pose.position, dtype=np.float64)

    with tempfile.TemporaryDirectory(prefix="pbr_envmap_") as td:
        out_path = _setup_compositor_for_envmap(bpy, Path(td))
        for i in range(env_row):
            for j in range(env_col):
                # Anchor pixel: centre of this grid cell.
                py = min(int((i + 0.5) * cfg.patch_size), h - 1)
                px = min(int((j + 0.5) * cfg.patch_size), w - 1)
                t = float(depth[py, px])
                n = np.asarray(normal[py, px], dtype=np.float64)
                if not np.isfinite(t) or t <= 0.0 or float(np.linalg.norm(n)) < 1e-6:
                    # No surface here — anchor the envmap at the camera origin
                    # with the camera's local up axis as the "normal" so the
                    # resulting envmap is at least valid (free-space lighting
                    # estimate). Caller can mask via depth.
                    surface_pt = cam_origin
                    surf_n = pose.world_up()
                else:
                    surface_pt = _backproject_pixel(
                        float(px) + 0.5, float(py) + 0.5, t, intrinsics, c2w
                    )
                    surf_n = n

                wm = _world_matrix_for_envmap(surface_pt, surf_n)
                _configure_panoramic_camera(
                    bpy, cam_obj, wm, cfg.env_height, cfg.env_width
                )
                _render_now(bpy, seed=seed + i * env_col + j)
                tile = _read_exr(out_path)
                if tile.ndim != 3 or tile.shape[-1] < 3:
                    raise RuntimeError(f"Unexpected envmap shape {tile.shape}")
                tile = np.ascontiguousarray(tile[..., :3], dtype=np.float32)
                row0, row1 = i * cfg.env_height, (i + 1) * cfg.env_height
                col0, col1 = j * cfg.env_width, (j + 1) * cfg.env_width
                mosaic[row0:row1, col0:col1] = tile

                # File Output appends frames at the same path each iteration;
                # delete the file so the next iteration's render isn't
                # confused if Blender starts version-suffixing.
                if out_path.exists():
                    out_path.unlink()

    return mosaic
