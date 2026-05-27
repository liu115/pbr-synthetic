"""Scene loading, bbox, up-axis detection, raycasting helpers."""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, cast

import mitsuba as mi
import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)

UpAxis = Literal["x", "y", "z"]
_AXIS_INDEX: dict[UpAxis, int] = {"x": 0, "y": 1, "z": 2}

# Shape IDs that we treat as part of the room shell (used by floor-polygon
# extraction). Matched case-insensitively as a substring of the shape ID.
_ROOM_SHELL_PATTERN = re.compile(r"(wall|floor|ceiling|skirting)", re.IGNORECASE)

# Rotation applied by the mitsuba-blender addon when it imports a Mitsuba
# scene: Mitsuba Y-up -> Blender Z-up. ``R @ p_mitsuba == p_blender``.
R_MITSUBA_TO_BLENDER: NDArray[np.float64] = np.array(
    [[1.0, 0.0, 0.0],
     [0.0, 0.0, -1.0],
     [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
R_MITSUBA_TO_BLENDER_4: NDArray[np.float64] = np.eye(4, dtype=np.float64)
R_MITSUBA_TO_BLENDER_4[:3, :3] = R_MITSUBA_TO_BLENDER


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
    """Select and set a Mitsuba variant. Returns the selected variant name.

    If a variant is already set on the process, we keep it unless ``prefer``
    is given and differs — switching variants mid-process triggers Mitsuba's
    "module reloaded" warning and has been observed to produce all-zero
    renders. The unified pipeline relies on this behavior so that the
    scalar variant set by :func:`load_scene_blender` (needed by the
    mitsuba-blender addon) is preserved.
    """
    available = set(mi.variants())
    try:
        current = mi.variant()
    except Exception:
        current = None

    if current and (prefer is None or prefer == current):
        log.info("Mitsuba variant: %s (already set)", current)
        return str(current)

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


# ---- 2D polygon helpers ------------------------------------------------------
#
# Used by the wall-walk camera sampler to test whether candidate camera
# positions lie inside the room footprint, and by ``extract_floor_polygon`` to
# fall back to a minimum-bounding-rectangle when named-wall geometry is not
# available. All polygons here are in the horizontal plane: the up axis is
# dropped and the remaining two axes are taken in their natural order
# (x-up → polygon is (y, z), y-up → polygon is (x, z), z-up → polygon is (x, y)).


def points_in_polygon(
    points: NDArray[np.float64], polygon: NDArray[np.float64]
) -> NDArray[np.bool_]:
    """Vectorised ray-casting point-in-polygon test in the plane.

    Args:
        points: ``(M, 2)`` candidate points.
        polygon: ``(N, 2)`` polygon vertices in any consistent winding. The
            polygon is treated as closed (last vertex implicitly connects to
            the first).

    Returns:
        ``(M,)`` boolean array. Points exactly on an edge are considered inside
        with the usual odd-crossing convention.
    """
    pts = np.asarray(points, dtype=np.float64)
    poly = np.asarray(polygon, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must be (M, 2); got {pts.shape}")
    if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 3:
        raise ValueError(f"polygon must be (N>=3, 2); got {poly.shape}")

    x = pts[:, 0]
    y = pts[:, 1]
    inside = np.zeros(pts.shape[0], dtype=np.bool_)
    n = poly.shape[0]
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        # Edges that straddle the horizontal line through the point.
        cond = ((yi > y) != (yj > y))
        with np.errstate(divide="ignore", invalid="ignore"):
            x_int = (xj - xi) * (y - yi) / (yj - yi) + xi
        crosses = cond & (x < x_int)
        inside ^= crosses
        j = i
    return inside


def point_in_polygon(point: NDArray[np.float64], polygon: NDArray[np.float64]) -> bool:
    """Scalar version of :func:`points_in_polygon` for a single point."""
    return bool(points_in_polygon(np.asarray(point).reshape(1, 2), polygon)[0])


def minimum_bounding_rectangle(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Smallest-area rectangle enclosing 2D ``points``.

    Uses the rotating-calipers algorithm on the convex hull. Returns a ``(4, 2)``
    array of corners in winding order (counter-clockwise starting from the
    bottom-left of the rotated frame).
    """
    from scipy.spatial import ConvexHull

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 3:
        raise ValueError(f"points must be (N>=3, 2); got {pts.shape}")

    hull = ConvexHull(pts)
    hull_pts = pts[hull.vertices]

    edges = np.diff(np.vstack([hull_pts, hull_pts[:1]]), axis=0)
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    # Each unique edge orientation produces a candidate rectangle. Restricting
    # to [0, pi/2) is the standard rotating-calipers reduction.
    angles = np.unique(np.mod(angles, np.pi / 2.0))

    best_area = np.inf
    best_corners = np.zeros((4, 2), dtype=np.float64)
    for theta in angles:
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, s], [-s, c]], dtype=np.float64)
        rotated = hull_pts @ rot.T
        x_min, y_min = rotated.min(axis=0)
        x_max, y_max = rotated.max(axis=0)
        area = (x_max - x_min) * (y_max - y_min)
        if area < best_area:
            best_area = area
            corners_rot = np.array(
                [
                    [x_min, y_min],
                    [x_max, y_min],
                    [x_max, y_max],
                    [x_min, y_max],
                ],
                dtype=np.float64,
            )
            best_corners = corners_rot @ rot
    return best_corners


# ---- Floor polygon extraction -----------------------------------------------


def _horizontal_axes(up_axis: UpAxis) -> tuple[int, int]:
    """Indices of the two non-up axes, in natural order."""
    up_idx = _AXIS_INDEX[up_axis]
    return cast(tuple[int, int], tuple(i for i in range(3) if i != up_idx))


def _load_shape_vertices(scene_xml: Path, only_room_shell: bool) -> NDArray[np.float64]:
    """Concatenate vertices from OBJ shapes referenced in ``scene_xml``.

    Uses :func:`src.mesh_utils.parse_shape_bindings` to discover shapes (it
    already handles ``<transform>`` matrices for us). When ``only_room_shell``
    is True, restricts to shapes whose ID matches ``_ROOM_SHELL_PATTERN``;
    returns an empty array if none match (the caller falls back to all shapes).
    """
    import trimesh  # noqa: F401  (imported for the cast type below)

    from src.mesh_utils import parse_shape_bindings

    bindings = parse_shape_bindings(scene_xml)
    parts: list[NDArray[np.float64]] = []
    for b in bindings:
        if only_room_shell and not _ROOM_SHELL_PATTERN.search(b.shape_id or ""):
            continue
        if not b.obj_path.exists():
            log.warning("Skipping missing OBJ %s for shape %s", b.obj_path, b.shape_id)
            continue
        try:
            loaded = trimesh.load(b.obj_path, force="mesh", process=False)
        except Exception as e:  # pragma: no cover - exercised via integration only
            log.warning("Failed to load %s: %s", b.obj_path, e)
            continue
        # ``force="mesh"`` returns a Trimesh; the static return type is broader.
        mesh = cast(trimesh.Trimesh, loaded)
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        if b.transform is not None:
            t = np.asarray(b.transform, dtype=np.float64).reshape(4, 4)
            verts_h = np.hstack([verts, np.ones((verts.shape[0], 1))])
            verts = (verts_h @ t.T)[:, :3]
        parts.append(verts)

    if not parts:
        return np.zeros((0, 3), dtype=np.float64)
    return np.vstack(parts)


def floor_polygon_from_verts(
    shell_verts: NDArray[np.float64],
    fallback_verts_fn: Callable[[], NDArray[np.float64]],
    up_axis: UpAxis,
) -> NDArray[np.float64]:
    """Build a 2D floor polygon (``(N, 2)``) from 3D vertex sets.

    First tries the convex hull of ``shell_verts`` (presumed to be the
    wall/floor/ceiling/skirting projection). On failure (too few vertices or
    degenerate hull), calls ``fallback_verts_fn()`` and returns the minimum
    bounding rectangle of the result. The 2D coordinates correspond to the
    two non-up axes in natural order.

    ``fallback_verts_fn`` is a thunk so the (potentially expensive) full-mesh
    gather only happens when the shell-hull path fails.
    """
    from scipy.spatial import ConvexHull

    horiz = _horizontal_axes(up_axis)
    if shell_verts.shape[0] >= 3:
        v2 = shell_verts[:, list(horiz)]
        try:
            hull = ConvexHull(v2)
            polygon = v2[hull.vertices]
            log.info(
                "floor_polygon_from_verts: convex hull of room shell (%d points)",
                polygon.shape[0],
            )
            return cast(NDArray[np.float64], polygon.astype(np.float64, copy=False))
        except Exception as e:
            log.warning("ConvexHull failed on room shell (%s); falling back to MBR", e)

    all_verts = fallback_verts_fn()
    if all_verts.shape[0] < 3:
        raise RuntimeError(
            "Cannot extract floor polygon: no fallback geometry available."
        )
    v2 = all_verts[:, list(horiz)]
    polygon = minimum_bounding_rectangle(v2)
    log.info("floor_polygon_from_verts: MBR fallback over all geometry")
    return polygon


def extract_floor_polygon(
    scene_xml: Path, up_axis: UpAxis
) -> NDArray[np.float64]:
    """Return a 2D polygon (``(N, 2)``) approximating the room footprint.

    Strategy:
        1. If any shape in the XML has an ID containing wall/floor/ceiling/
           skirting, use the convex hull of their floor-projected vertices.
        2. Otherwise, fall back to the minimum bounding rectangle of all
           geometry's floor projection (4 corners).

    The 2D coordinates correspond to the two non-up axes in natural order.
    """
    shell_verts = _load_shape_vertices(scene_xml, only_room_shell=True)
    return floor_polygon_from_verts(
        shell_verts,
        lambda: _load_shape_vertices(scene_xml, only_room_shell=False),
        up_axis,
    )


def parse_scene_up_axis_from_xml(scene_xml: Path) -> UpAxis | None:
    """Best-effort up-axis hint from the XML's ``<default name="up_axis">``.

    Returns None if the scene XML doesn't declare one (most Bitterli scenes
    don't — they assume Y up); callers should fall back to bbox heuristics.
    """
    try:
        root = ET.parse(scene_xml).getroot()
    except ET.ParseError:
        return None
    for d in root.findall("default"):
        if d.get("name") in {"up_axis", "axis_up"}:
            value = (d.get("value") or "").lower().strip("+-")
            if value in {"x", "y", "z"}:
                return cast(UpAxis, value)
    return None
