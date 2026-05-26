"""Mitsuba beauty rendering + shared AOV / RenderConfig dataclasses.

Since the FIPT-parity refactor, **all AOVs and the depth-validity filter
are rendered with Blender Cycles** (see ``src/render_blender.py``). The
Mitsuba side keeps only the beauty pass — ``render_beauty`` — plus the
shared dataclasses (``AOVImages``, ``RenderConfig``) and small utilities
(``make_contact_sheet``).

The old Mitsuba AOV path, per-shape material LUT, and depth-filter helpers
were removed because they produced per-shape constants for roughness and
metallic; the Blender shader-AOV path now provides true per-pixel maps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

import mitsuba as mi
import numpy as np
from numpy.typing import NDArray

from src.camera_sampling import CameraPose
from src.io_utils import Intrinsics
from src.pose_utils import pose_to_c2w  # re-exported for backward compatibility

__all__ = [
    "AOVImages",
    "RenderConfig",
    "make_contact_sheet",
    "pose_to_c2w",
    "render_beauty",
]

log = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class RenderConfig:
    intrinsics_full: Intrinsics
    intrinsics_debug: Intrinsics
    spp_beauty_full: int
    spp_beauty_debug: int
    spp_aov: int
    spp_filter: int
    filter_width: int
    filter_height: int
    max_depth: int
    debug: bool

    @property
    def beauty_intrinsics(self) -> Intrinsics:
        return self.intrinsics_debug if self.debug else self.intrinsics_full

    @property
    def spp_beauty(self) -> int:
        return self.spp_beauty_debug if self.debug else self.spp_beauty_full


@dataclass(slots=True, frozen=True)
class AOVImages:
    """Decoded AOV channels for one camera.

    All fields are now produced by the Blender backend
    (:func:`src.render_blender.render_aov_blender`). The four geometric
    fields are required; the rest are optional only because some unit
    tests construct fake ``AOVImages`` instances.
    """

    depth: NDArray[np.float32]
    normal: NDArray[np.float32]
    albedo: NDArray[np.float32]                          # k_d (Cycles DiffCol)
    shape_index: NDArray[np.int32]
    glossy_color: NDArray[np.float32] | None = None      # k_s (Cycles GlossCol)
    emission: NDArray[np.float32] | None = None          # Cycles Emit
    material_index: NDArray[np.int32] | None = None      # Cycles IndexMA
    roughness_per_pixel: NDArray[np.float32] | None = None  # shader AOV
    metallic_per_pixel: NDArray[np.float32] | None = None   # shader AOV


# Rotation that maps Mitsuba's Y-up world to Blender's Z-up world (this is what
# the mitsuba-blender addon applies on import). Multiplying on the LEFT
# transforms vectors / matrices from Mitsuba space into Blender space.
_M_Y_UP_TO_Z_UP: NDArray[np.float64] = np.array(
    [[1.0,  0.0,  0.0, 0.0],
     [0.0,  0.0, -1.0, 0.0],
     [0.0,  1.0,  0.0, 0.0],
     [0.0,  0.0,  0.0, 1.0]],
    dtype=np.float64,
)


def _build_sensor(
    pose: CameraPose,
    intrinsics: Intrinsics,
    spp: int,
    sample_seed_offset: int = 0,
    *,
    world_axis_transform: NDArray[np.float64] | None = None,
) -> mi.Sensor:
    """Build a Mitsuba sensor for the given pose.

    ``world_axis_transform`` is an optional 4x4 matrix applied to the
    look-at origin/target/up before passing them to Mitsuba's
    ``ScalarTransform4f.look_at``. Used to bridge the coordinate-system
    mismatch when poses are sampled in Blender's Z-up world (because the
    mitsuba-blender addon rotates the scene on import) but the Mitsuba
    scene was loaded directly from a Y-up XML.
    """
    origin = np.asarray(pose.position, dtype=np.float64)
    target = np.asarray(pose.target(distance=1.0), dtype=np.float64)
    up = pose.world_up()

    if world_axis_transform is not None:
        R = world_axis_transform[:3, :3]
        t = world_axis_transform[:3, 3]
        origin = R @ origin + t
        target = R @ target + t
        up = R @ up

    sensor_dict: dict[str, Any] = {
        "type": "perspective",
        "fov": float(np.rad2deg(intrinsics.fov_x_rad)),
        "fov_axis": "x",
        "to_world": mi.ScalarTransform4f.look_at(
            origin=list(origin.tolist()),
            target=list(target.tolist()),
            up=list(up.tolist()),
        ),
        "film": {
            "type": "hdrfilm",
            "width": intrinsics.width,
            "height": intrinsics.height,
            "pixel_format": "rgb",
            "file_format": "openexr",
            "rfilter": {"type": "tent"},
        },
        "sampler": {
            "type": "independent",
            "sample_count": spp,
            "seed": sample_seed_offset,
        },
    }
    return mi.load_dict(sensor_dict)


def render_beauty(
    scene: mi.Scene,
    pose: CameraPose,
    intrinsics: Intrinsics,
    spp: int,
    max_depth: int,
    seed: int = 0,
    *,
    pose_frame: Literal["mitsuba", "blender"] = "mitsuba",
) -> NDArray[np.float32]:
    """Path-traced HDR RGB in linear radiance. Returns ``(H, W, 3)``.

    ``pose_frame`` selects how ``pose`` should be interpreted:
    ``"mitsuba"`` (the default) treats it as already living in the
    Mitsuba scene's world frame; ``"blender"`` applies the inverse of the
    mitsuba-blender addon's Y-up → Z-up rotation so a Z-up Blender pose
    aligns with the loaded Y-up Mitsuba scene.
    """
    world_xf: NDArray[np.float64] | None = None
    if pose_frame == "blender":
        # Inverse of Y-up -> Z-up rotation.
        world_xf = np.linalg.inv(_M_Y_UP_TO_Z_UP)
    sensor = _build_sensor(
        pose, intrinsics, spp, sample_seed_offset=seed,
        world_axis_transform=world_xf,
    )
    integrator = mi.load_dict({"type": "path", "max_depth": max_depth})
    img = mi.render(scene, sensor=sensor, integrator=integrator, spp=spp, seed=seed)
    arr = np.asarray(img, dtype=np.float32)
    if arr.shape[-1] >= 3:
        return arr[..., :3].copy()
    raise RuntimeError(f"Unexpected beauty image shape {arr.shape}")


def make_contact_sheet(
    ldr_images: list[NDArray[np.uint8]], grid: tuple[int, int] = (4, 4)
) -> NDArray[np.uint8]:
    """Tile up to ``grid[0] * grid[1]`` LDR images into a single preview PNG."""
    if not ldr_images:
        raise ValueError("No images provided for contact sheet")
    rows, cols = grid
    sample = ldr_images[0]
    h, w = sample.shape[:2]
    channels = sample.shape[2] if sample.ndim == 3 else 1
    canvas = np.zeros((rows * h, cols * w, channels), dtype=np.uint8)
    for k, img in enumerate(ldr_images[: rows * cols]):
        r, c = divmod(k, cols)
        i = img
        if i.shape[:2] != (h, w):
            continue
        if i.ndim == 2:
            i = np.repeat(i[..., None], channels, axis=-1)
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = i
    if channels == 1:
        canvas = canvas[..., 0]
    return canvas
