"""Camera placement, orientation, and depth-validity filtering.

The :class:`CameraSampler` ABC is the entry point. Two implementations are
provided:

* :class:`UniformRandomSampler` — uniform 3D box sampling (the previous
  behavior, kept as default for unchanged scripts).
* :class:`WallWalkSampler` — FIPT-style wall traversal: walks along each edge
  of the room footprint and biases cameras to face the room interior. Reduces
  rejection rate on complex layouts and gives more uniform angular coverage.

The free function :func:`sample_cameras` is preserved as a thin shim over
:class:`UniformRandomSampler` so existing call-sites and tests keep working.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Literal, cast

import numpy as np
from numpy.typing import NDArray

from src.scene_utils import (
    SceneInfo,
    UpAxis,
    _AXIS_INDEX,
    points_in_polygon,
)

log = logging.getLogger(__name__)


RejectionReason = Literal[
    "ok",
    "outside_room",
    "outside_polygon",
    "too_much_inf",
    "too_close",
    "too_close_to_wall",
    "too_much_near",
]


# ---- Public dataclasses ------------------------------------------------------


@dataclass(slots=True, frozen=True)
class CameraPose:
    """A camera pose parameterized by world-space position + yaw/pitch.

    Yaw is rotation around the world up axis. Pitch is rotation around the
    camera's right axis (positive pitch tilts the camera upward). Roll is
    always zero.
    """

    position: tuple[float, float, float]
    yaw: float
    pitch: float
    up_axis: UpAxis

    def world_up(self) -> NDArray[np.float64]:
        v = np.zeros(3, dtype=np.float64)
        v[_AXIS_INDEX[self.up_axis]] = 1.0
        return v

    def forward(self) -> NDArray[np.float64]:
        """Unit forward vector in world coordinates."""
        up_idx = _AXIS_INDEX[self.up_axis]
        h0, h1 = [i for i in range(3) if i != up_idx]
        f = np.zeros(3, dtype=np.float64)
        cp = float(np.cos(self.pitch))
        sp = float(np.sin(self.pitch))
        f[h0] = cp * float(np.sin(self.yaw))
        f[h1] = -cp * float(np.cos(self.yaw))
        f[up_idx] = sp
        n = float(np.linalg.norm(f))
        return f / n if n > 0 else f

    def target(self, distance: float = 1.0) -> tuple[float, float, float]:
        p = np.asarray(self.position, dtype=np.float64)
        t = p + self.forward() * distance
        return (float(t[0]), float(t[1]), float(t[2]))


@dataclass(slots=True, frozen=True)
class DepthFilterConfig:
    """Thresholds for the cheap depth-validity check.

    A camera is accepted iff:
      - infinity_ratio <= max_infinity_ratio (no escape to void)
      - median_depth >= min_median_depth (not jammed against a wall)
      - near_ratio   <= max_near_ratio (small fraction of pixels too close)
    """

    max_infinity_ratio: float = 0.01
    min_median_depth: float = 0.3
    near_threshold: float = 0.2
    max_near_ratio: float = 0.05


@dataclass(slots=True, frozen=True)
class DepthFilterResult:
    accepted: bool
    reason: RejectionReason
    infinity_ratio: float
    median_depth: float
    near_ratio: float


@dataclass(slots=True)
class FilterStats:
    """Aggregate stats from a sampling run."""

    n_attempts: int = 0
    n_accepted: int = 0
    by_reason: Counter[str] = field(default_factory=Counter)

    def as_dict(self) -> dict[str, object]:
        return {
            "n_attempts": self.n_attempts,
            "n_accepted": self.n_accepted,
            "by_reason": dict(self.by_reason),
        }


# ---- Module-level helpers (reused by all samplers) ---------------------------


def depth_validity(
    depth: NDArray[np.float64], cfg: DepthFilterConfig
) -> DepthFilterResult:
    """Inspect a depth array and decide whether the camera is acceptable."""
    if depth.size == 0:
        raise ValueError("Empty depth array")

    inf_mask = ~np.isfinite(depth) | (depth <= 0.0)
    n_total = float(depth.size)
    inf_ratio = float(inf_mask.sum()) / n_total
    finite = depth[~inf_mask]
    median = float(np.median(finite)) if finite.size > 0 else 0.0
    near_ratio = float((finite < cfg.near_threshold).sum()) / n_total

    if inf_ratio > cfg.max_infinity_ratio:
        return DepthFilterResult(
            False, "too_much_inf", inf_ratio, median, near_ratio
        )
    if median < cfg.min_median_depth:
        return DepthFilterResult(False, "too_close", inf_ratio, median, near_ratio)
    if near_ratio > cfg.max_near_ratio:
        return DepthFilterResult(
            False, "too_much_near", inf_ratio, median, near_ratio
        )
    return DepthFilterResult(True, "ok", inf_ratio, median, near_ratio)


def sample_position(
    rng: np.random.Generator, info: SceneInfo
) -> NDArray[np.float64]:
    p = np.empty(3, dtype=np.float64)
    for i in range(3):
        p[i] = rng.uniform(info.placement_min[i], info.placement_max[i])
    return p


def sample_orientation(
    rng: np.random.Generator,
    pitch_range_deg: tuple[float, float] = (-25.0, 25.0),
) -> tuple[float, float]:
    yaw = float(rng.uniform(0.0, 2.0 * np.pi))
    p_lo = float(np.deg2rad(pitch_range_deg[0]))
    p_hi = float(np.deg2rad(pitch_range_deg[1]))
    pitch = float(rng.uniform(p_lo, p_hi))
    return yaw, pitch


# Callback signatures shared by both samplers.
DepthRenderFn = Callable[[CameraPose], NDArray[np.float64]]
InsideRoomFn = Callable[[NDArray[np.float64]], NDArray[np.bool_]]


def _yaw_from_horizontal_direction(
    direction_h0: float, direction_h1: float
) -> float:
    """Convert a horizontal direction vector to the yaw angle used by CameraPose.

    ``CameraPose.forward`` is ``(cos(pitch) * sin(yaw), -cos(pitch) * cos(yaw))``
    on the two horizontal axes, so a unit horizontal direction ``(d_h0, d_h1)``
    corresponds to ``yaw = atan2(d_h0, -d_h1)``.
    """
    return float(np.arctan2(direction_h0, -direction_h1))


# ---- Sampler ABC + implementations ------------------------------------------


class CameraSampler(ABC):
    """Base class for camera pose samplers.

    Subclasses implement :meth:`sample`, which must return a list of accepted
    poses and a :class:`FilterStats` summary. The two callbacks
    (``inside_room``, ``render_depth``) keep the sampler decoupled from any
    specific renderer.
    """

    def __init__(
        self,
        info: SceneInfo,
        rng: np.random.Generator,
        inside_room: InsideRoomFn,
        render_depth: DepthRenderFn,
        depth_filter: DepthFilterConfig,
    ) -> None:
        self.info = info
        self.rng = rng
        self.inside_room = inside_room
        self.render_depth = render_depth
        self.depth_filter = depth_filter

    @abstractmethod
    def sample(
        self, num_cameras: int, oversample: int = 4
    ) -> tuple[list[CameraPose], FilterStats]:
        ...

    # Shared helper: run depth filter + return acceptance + reason.
    def _check_depth(self, pose: CameraPose) -> DepthFilterResult:
        depth = self.render_depth(pose)
        return depth_validity(depth, self.depth_filter)


class UniformRandomSampler(CameraSampler):
    """Uniform random placement inside the SceneInfo placement box."""

    def __init__(
        self,
        info: SceneInfo,
        rng: np.random.Generator,
        inside_room: InsideRoomFn,
        render_depth: DepthRenderFn,
        depth_filter: DepthFilterConfig,
        pitch_range_deg: tuple[float, float] = (-25.0, 25.0),
    ) -> None:
        super().__init__(info, rng, inside_room, render_depth, depth_filter)
        self.pitch_range_deg = pitch_range_deg

    def sample(
        self, num_cameras: int, oversample: int = 4
    ) -> tuple[list[CameraPose], FilterStats]:
        if num_cameras <= 0:
            return [], FilterStats()
        max_attempts = oversample * num_cameras
        stats = FilterStats()
        accepted: list[CameraPose] = []

        while stats.n_attempts < max_attempts and len(accepted) < num_cameras:
            stats.n_attempts += 1
            pos = sample_position(self.rng, self.info)
            if not bool(self.inside_room(pos[None, :])[0]):
                stats.by_reason["outside_room"] += 1
                continue
            yaw, pitch = sample_orientation(self.rng, self.pitch_range_deg)
            pose = CameraPose(
                position=cast(
                    tuple[float, float, float], tuple(pos.tolist())
                ),
                yaw=yaw,
                pitch=pitch,
                up_axis=self.info.up_axis,
            )
            result = self._check_depth(pose)
            stats.by_reason[result.reason] += 1
            if result.accepted:
                accepted.append(pose)
                stats.n_accepted += 1

        if len(accepted) < num_cameras:
            log.warning(
                "Only sampled %d / %d cameras after %d attempts. Rejection "
                "counts: %s",
                len(accepted), num_cameras, stats.n_attempts,
                dict(stats.by_reason),
            )
        return accepted, stats


class WallWalkSampler(CameraSampler):
    """FIPT-style: walk each room-edge, offset inward, face the interior.

    For each edge of ``floor_polygon`` the sampler walks along the edge in
    ``samples_per_segment`` evenly-spaced steps. At each step it picks a
    random offset ``[dist_min, dist_max]`` along the inward normal, then a
    random height + pitch + yaw-offset around the inward direction.

    The 2D polygon is in the *horizontal* plane (the two non-up axes, in
    natural order). The polygon winding is auto-detected; inward normals are
    flipped if needed.
    """

    def __init__(
        self,
        info: SceneInfo,
        rng: np.random.Generator,
        inside_room: InsideRoomFn,
        render_depth: DepthRenderFn,
        depth_filter: DepthFilterConfig,
        floor_polygon: NDArray[np.float64],
        dist_min: float = 0.4,
        dist_max: float = 2.5,
        samples_per_segment: int = 8,
        pitch_range_deg: tuple[float, float] = (-25.0, 25.0),
        yaw_offset_range_deg: tuple[float, float] = (-60.0, 60.0),
    ) -> None:
        super().__init__(info, rng, inside_room, render_depth, depth_filter)
        self.floor_polygon = np.asarray(floor_polygon, dtype=np.float64)
        if self.floor_polygon.ndim != 2 or self.floor_polygon.shape[1] != 2:
            raise ValueError(
                f"floor_polygon must be (N, 2); got {self.floor_polygon.shape}"
            )
        if self.floor_polygon.shape[0] < 3:
            raise ValueError("floor_polygon needs at least 3 vertices")
        self.dist_min = float(dist_min)
        self.dist_max = float(dist_max)
        self.samples_per_segment = int(samples_per_segment)
        self.pitch_range_deg = pitch_range_deg
        self.yaw_offset_range_deg = yaw_offset_range_deg

        # Precompute inward normals (one per edge of the polygon).
        self._edge_midpoints, self._edge_normals = self._compute_inward_normals()

    def _compute_inward_normals(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        poly = self.floor_polygon
        n = poly.shape[0]
        midpoints = np.zeros((n, 2), dtype=np.float64)
        normals = np.zeros((n, 2), dtype=np.float64)
        for i in range(n):
            j = (i + 1) % n
            edge = poly[j] - poly[i]
            length = float(np.linalg.norm(edge))
            if length < 1e-9:
                continue
            edge /= length
            # Two candidate perpendiculars; pick the one pointing inward by
            # testing a small step from the edge midpoint.
            mid = 0.5 * (poly[i] + poly[j])
            normal = np.array([-edge[1], edge[0]], dtype=np.float64)
            test_in = mid + normal * 1e-3
            if not bool(points_in_polygon(test_in[None, :], poly)[0]):
                normal = -normal
            midpoints[i] = mid
            normals[i] = normal
        return midpoints, normals

    def _build_pose(
        self,
        position: NDArray[np.float64],
        inward_2d: NDArray[np.float64],
    ) -> CameraPose:
        """Construct a CameraPose looking inward + random pitch + yaw offset."""
        h0, h1 = self.info.horizontal_axes
        base_yaw = _yaw_from_horizontal_direction(
            float(inward_2d[0]), float(inward_2d[1])
        )
        yaw_offset = float(
            np.deg2rad(
                self.rng.uniform(
                    self.yaw_offset_range_deg[0], self.yaw_offset_range_deg[1]
                )
            )
        )
        pitch = float(
            np.deg2rad(
                self.rng.uniform(self.pitch_range_deg[0], self.pitch_range_deg[1])
            )
        )
        return CameraPose(
            position=(float(position[0]), float(position[1]), float(position[2])),
            yaw=base_yaw + yaw_offset,
            pitch=pitch,
            up_axis=self.info.up_axis,
        )

    def sample(
        self, num_cameras: int, oversample: int = 4
    ) -> tuple[list[CameraPose], FilterStats]:
        if num_cameras <= 0:
            return [], FilterStats()
        max_attempts = oversample * num_cameras
        stats = FilterStats()
        accepted: list[CameraPose] = []

        h0, h1 = self.info.horizontal_axes
        up_idx = _AXIS_INDEX[self.info.up_axis]
        h_min = self.info.placement_min[up_idx]
        h_max = self.info.placement_max[up_idx]
        n_edges = self.floor_polygon.shape[0]

        while stats.n_attempts < max_attempts and len(accepted) < num_cameras:
            stats.n_attempts += 1
            edge_idx = int(self.rng.integers(0, n_edges))
            t = float(self.rng.uniform(0.0, 1.0))
            i, j = edge_idx, (edge_idx + 1) % n_edges
            edge_pt = (1.0 - t) * self.floor_polygon[i] + t * self.floor_polygon[j]
            normal = self._edge_normals[edge_idx]
            if float(np.linalg.norm(normal)) < 1e-9:
                # Degenerate edge — no inward direction. Skip.
                stats.by_reason["outside_polygon"] += 1
                continue

            offset = float(self.rng.uniform(self.dist_min, self.dist_max))
            pos_2d = edge_pt + normal * offset
            if not bool(points_in_polygon(pos_2d[None, :], self.floor_polygon)[0]):
                stats.by_reason["outside_polygon"] += 1
                continue

            # Confirm distance from every wall (so we don't pick a point that's
            # close to the *opposite* wall just because we offset from this one).
            dists = np.einsum(
                "ij,ij->i",
                pos_2d[None, :] - self._edge_midpoints,
                self._edge_normals,
            )
            if float(dists.min()) < self.dist_min:
                stats.by_reason["too_close_to_wall"] += 1
                continue

            # Lift to 3D and verify the inside-room raycast still passes.
            pos_3d = np.zeros(3, dtype=np.float64)
            pos_3d[h0] = pos_2d[0]
            pos_3d[h1] = pos_2d[1]
            pos_3d[up_idx] = float(self.rng.uniform(h_min, h_max))
            if not bool(self.inside_room(pos_3d[None, :])[0]):
                stats.by_reason["outside_room"] += 1
                continue

            pose = self._build_pose(pos_3d, normal)
            result = self._check_depth(pose)
            stats.by_reason[result.reason] += 1
            if result.accepted:
                accepted.append(pose)
                stats.n_accepted += 1

        if len(accepted) < num_cameras:
            log.warning(
                "WallWalkSampler: only %d / %d cameras after %d attempts. "
                "Rejection counts: %s",
                len(accepted), num_cameras, stats.n_attempts,
                dict(stats.by_reason),
            )
        return accepted, stats


# ---- Factory + backward-compatible free function ----------------------------


SamplerName = Literal["uniform", "wall_walk"]


def make_sampler(
    name: SamplerName,
    *,
    info: SceneInfo,
    rng: np.random.Generator,
    inside_room: InsideRoomFn,
    render_depth: DepthRenderFn,
    depth_filter: DepthFilterConfig,
    pitch_range_deg: tuple[float, float] = (-25.0, 25.0),
    floor_polygon: NDArray[np.float64] | None = None,
    wall_walk_dist_min: float = 0.4,
    wall_walk_dist_max: float = 2.5,
    yaw_offset_range_deg: tuple[float, float] = (-60.0, 60.0),
) -> CameraSampler:
    """Construct a :class:`CameraSampler` by name."""
    if name == "uniform":
        return UniformRandomSampler(
            info=info,
            rng=rng,
            inside_room=inside_room,
            render_depth=render_depth,
            depth_filter=depth_filter,
            pitch_range_deg=pitch_range_deg,
        )
    if name == "wall_walk":
        if floor_polygon is None:
            raise ValueError("wall_walk sampler requires floor_polygon")
        return WallWalkSampler(
            info=info,
            rng=rng,
            inside_room=inside_room,
            render_depth=render_depth,
            depth_filter=depth_filter,
            floor_polygon=floor_polygon,
            dist_min=wall_walk_dist_min,
            dist_max=wall_walk_dist_max,
            pitch_range_deg=pitch_range_deg,
            yaw_offset_range_deg=yaw_offset_range_deg,
        )
    raise ValueError(f"Unknown sampler name: {name!r}")


def sample_cameras(
    rng: np.random.Generator,
    info: SceneInfo,
    num_cameras: int,
    inside_room: InsideRoomFn,
    render_depth: DepthRenderFn,
    depth_filter: DepthFilterConfig,
    pitch_range_deg: tuple[float, float] = (-25.0, 25.0),
    max_attempts: int | None = None,
    oversample: int = 4,
) -> tuple[list[CameraPose], FilterStats]:
    """Backward-compatible wrapper around :class:`UniformRandomSampler`.

    New code should prefer :func:`make_sampler` so the wall-walk strategy is
    discoverable; existing call-sites continue to work unchanged.
    """
    sampler = UniformRandomSampler(
        info=info,
        rng=rng,
        inside_room=inside_room,
        render_depth=render_depth,
        depth_filter=depth_filter,
        pitch_range_deg=pitch_range_deg,
    )
    if max_attempts is not None:
        oversample = max(oversample, max(1, max_attempts // max(1, num_cameras)))
    return sampler.sample(num_cameras=num_cameras, oversample=oversample)
