"""Scene loading, bbox, up-axis detection, raycasting helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import mitsuba as mi
import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)

UpAxis = Literal["x", "y", "z"]
_AXIS_INDEX: dict[UpAxis, int] = {"x": 0, "y": 1, "z": 2}


@dataclass(slots=True, frozen=True)
class SceneInfo:
    """Geometric summary of a loaded scene used by camera sampling."""

    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    up_axis: UpAxis
    placement_min: tuple[float, float, float]
    placement_max: tuple[float, float, float]

    @property
    def horizontal_axes(self) -> tuple[int, int]:
        up_idx = _AXIS_INDEX[self.up_axis]
        h = tuple(i for i in range(3) if i != up_idx)
        return cast(tuple[int, int], h)

    @property
    def height_range(self) -> tuple[float, float]:
        up_idx = _AXIS_INDEX[self.up_axis]
        return self.placement_min[up_idx], self.placement_max[up_idx]


def init_mitsuba(prefer: str | None = None) -> str:
    """Select and set a Mitsuba variant. Returns the selected variant name."""
    available = set(mi.variants())
    candidates: list[str] = []
    if prefer is not None:
        candidates.append(prefer)
    candidates += ["cuda_ad_rgb", "llvm_ad_rgb", "scalar_rgb"]
    for v in candidates:
        if v in available:
            mi.set_variant(v)
            log.info("Mitsuba variant: %s", v)
            return v
    raise RuntimeError(
        f"No suitable Mitsuba variant found. Available: {sorted(available)}"
    )


def load_scene(xml_path: Path) -> mi.Scene:
    return mi.load_file(str(xml_path))


def _vec3_to_tuple(v: object) -> tuple[float, float, float]:
    a = np.asarray(v).reshape(-1)
    return (float(a[0]), float(a[1]), float(a[2]))


def compute_bbox(
    scene: mi.Scene,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return the AABB of the scene as (min, max) tuples."""
    bb = scene.bbox()
    return _vec3_to_tuple(bb.min), _vec3_to_tuple(bb.max)


def detect_up_axis(
    bbox_min: tuple[float, float, float],
    bbox_max: tuple[float, float, float],
) -> UpAxis:
    """Heuristic: the up axis is the smallest scene extent.

    For typical indoor scenes (rooms wider than they are tall) this picks Y or Z
    correctly. Override via ``up_axis_override`` when this heuristic is wrong.
    """
    extents = np.asarray(bbox_max, dtype=np.float64) - np.asarray(
        bbox_min, dtype=np.float64
    )
    smallest = int(np.argmin(extents))
    axes: tuple[UpAxis, UpAxis, UpAxis] = ("x", "y", "z")
    return axes[smallest]


def cast_rays(
    scene: mi.Scene,
    origins: NDArray[np.float64],
    directions: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Batch ray intersection.

    Args:
        origins: ``(N, 3)`` array of ray origins.
        directions: ``(N, 3)`` array of ray directions (need not be unit length;
            the returned ``t`` is in the units of ``directions``, so pass unit
            vectors if you want distance in scene units).

    Returns:
        ``(distances, valid)`` with shapes ``(N,)``. ``distances`` is ``inf``
        where ``valid`` is ``False``.
    """
    o = np.ascontiguousarray(origins, dtype=np.float64)
    d = np.ascontiguousarray(directions, dtype=np.float64)
    if o.shape != d.shape or o.ndim != 2 or o.shape[1] != 3:
        raise ValueError(f"origins/directions must be (N, 3); got {o.shape}, {d.shape}")

    pf_o = mi.Point3f(o[:, 0], o[:, 1], o[:, 2])
    pf_d = mi.Vector3f(d[:, 0], d[:, 1], d[:, 2])
    ray = mi.Ray3f(pf_o, pf_d)
    si = scene.ray_intersect(ray)
    t = np.asarray(si.t, dtype=np.float64)
    valid = np.asarray(si.is_valid(), dtype=np.bool_)
    t = np.where(valid, t, np.inf)
    return t, valid


def inside_room_test(
    scene: mi.Scene,
    positions: NDArray[np.float64],
    up_axis: UpAxis,
    max_room_extent: float = 15.0,
    n_horizontal: int = 8,
) -> NDArray[np.bool_]:
    """Per-candidate inside-room raycast test.

    A position passes iff every probe ray hits a surface within
    ``max_room_extent``: one ray straight up, one straight down, and
    ``n_horizontal`` rays evenly spaced in the horizontal plane.

    Args:
        positions: ``(N, 3)`` candidate camera positions.
        up_axis: which axis is "up" in world space.
        max_room_extent: rays escaping past this distance are treated as void.
        n_horizontal: number of horizontal probe rays.

    Returns:
        Boolean array of shape ``(N,)``.
    """
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must be (N, 3); got {positions.shape}")
    n = positions.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.bool_)

    up_idx = _AXIS_INDEX[up_axis]
    horiz_axes = [i for i in range(3) if i != up_idx]

    up_vec = np.zeros(3, dtype=np.float64)
    up_vec[up_idx] = 1.0
    down_vec = -up_vec

    angles = np.linspace(0.0, 2.0 * np.pi, n_horizontal, endpoint=False)
    horiz_dirs = np.zeros((n_horizontal, 3), dtype=np.float64)
    horiz_dirs[:, horiz_axes[0]] = np.cos(angles)
    horiz_dirs[:, horiz_axes[1]] = np.sin(angles)

    all_dirs = np.vstack([up_vec[None, :], down_vec[None, :], horiz_dirs])
    n_dirs = all_dirs.shape[0]

    origins = np.repeat(positions.astype(np.float64), n_dirs, axis=0)
    directions = np.tile(all_dirs, (n, 1))

    distances, valid = cast_rays(scene, origins, directions)
    distances = distances.reshape(n, n_dirs)
    valid = valid.reshape(n, n_dirs)
    inside = valid & (distances < max_room_extent)
    result: NDArray[np.bool_] = inside.all(axis=1)
    return result


def scene_info_from_bbox(
    bb_min: tuple[float, float, float],
    bb_max: tuple[float, float, float],
    height_range: tuple[float, float] = (0.8, 1.8),
    placement_margin: float = 0.3,
    height_margin: float = 0.3,
    up_axis_override: UpAxis | None = None,
) -> SceneInfo:
    """Build a SceneInfo from an axis-aligned bbox + placement/height margins.

    Renderer-agnostic helper: works from any source of (bb_min, bb_max) tuples
    so the Mitsuba and Blender backends can share the same placement logic.
    """
    up = up_axis_override if up_axis_override is not None else detect_up_axis(
        bb_min, bb_max
    )
    up_idx = _AXIS_INDEX[up]

    pmin = list(bb_min)
    pmax = list(bb_max)
    for i in range(3):
        if i == up_idx:
            continue
        pmin[i] = bb_min[i] + placement_margin
        pmax[i] = bb_max[i] - placement_margin
    pmin[up_idx] = max(height_range[0], bb_min[up_idx] + height_margin)
    pmax[up_idx] = min(height_range[1], bb_max[up_idx] - height_margin)

    for i in range(3):
        if pmax[i] <= pmin[i]:
            raise ValueError(
                f"Empty placement region on axis {i}: "
                f"min={pmin[i]:.3f} >= max={pmax[i]:.3f}. "
                f"Scene bbox min={bb_min}, max={bb_max}, up={up}."
            )

    return SceneInfo(
        bbox_min=bb_min,
        bbox_max=bb_max,
        up_axis=up,
        placement_min=cast(tuple[float, float, float], tuple(pmin)),
        placement_max=cast(tuple[float, float, float], tuple(pmax)),
    )


def derive_scene_info(
    scene: mi.Scene,
    height_range: tuple[float, float] = (0.8, 1.8),
    placement_margin: float = 0.3,
    height_margin: float = 0.3,
    up_axis_override: UpAxis | None = None,
) -> SceneInfo:
    """Compute the placement region and height range for camera sampling."""
    bb_min, bb_max = compute_bbox(scene)
    return scene_info_from_bbox(
        bb_min=bb_min,
        bb_max=bb_max,
        height_range=height_range,
        placement_margin=placement_margin,
        height_margin=height_margin,
        up_axis_override=up_axis_override,
    )
