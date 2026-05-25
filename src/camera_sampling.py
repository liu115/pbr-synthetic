"""Random camera placement + orientation sampling + depth-validity filtering."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Literal, cast

import numpy as np
from numpy.typing import NDArray

from src.scene_utils import SceneInfo, UpAxis, _AXIS_INDEX

log = logging.getLogger(__name__)


RejectionReason = Literal[
    "ok",
    "outside_room",
    "too_much_inf",
    "too_close",
    "too_much_near",
]


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


def depth_validity(
    depth: NDArray[np.float64], cfg: DepthFilterConfig
) -> DepthFilterResult:
    """Inspect a depth array and decide whether the camera is acceptable.

    Pixels with non-finite depth or depth <= 0 are treated as "no hit" / void.
    """
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


# Function signature for the depth-render callback used during candidate
# filtering. Takes a candidate pose, returns a depth map.
DepthRenderFn = Callable[[CameraPose], NDArray[np.float64]]
# Function signature for the inside-room test (a batch of positions ->
# boolean mask). Returns shape (N,). Inject so this module stays decoupled.
InsideRoomFn = Callable[[NDArray[np.float64]], NDArray[np.bool_]]


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
    """Sample ``num_cameras`` poses that pass both inside-room and depth checks.

    Strategy: for each pose attempt, first reject based on inside-room raycast
    (very cheap), then reject based on a low-resolution depth render. Repeats
    until we have ``num_cameras`` accepted or we hit ``max_attempts`` (default
    ``oversample * num_cameras``).
    """
    if num_cameras <= 0:
        return [], FilterStats()
    if max_attempts is None:
        max_attempts = oversample * num_cameras

    stats = FilterStats()
    accepted: list[CameraPose] = []

    while stats.n_attempts < max_attempts and len(accepted) < num_cameras:
        stats.n_attempts += 1
        pos = sample_position(rng, info)
        ok_inside = bool(inside_room(pos[None, :])[0])
        if not ok_inside:
            stats.by_reason["outside_room"] += 1
            continue

        yaw, pitch = sample_orientation(rng, pitch_range_deg)
        pose = CameraPose(
            position=cast(tuple[float, float, float], tuple(pos.tolist())),
            yaw=yaw,
            pitch=pitch,
            up_axis=info.up_axis,
        )

        depth = render_depth(pose)
        result = depth_validity(depth, depth_filter)
        stats.by_reason[result.reason] += 1
        if result.accepted:
            accepted.append(pose)
            stats.n_accepted += 1

    if len(accepted) < num_cameras:
        log.warning(
            "Only sampled %d / %d cameras after %d attempts. "
            "Rejection counts: %s",
            len(accepted),
            num_cameras,
            stats.n_attempts,
            dict(stats.by_reason),
        )
    return accepted, stats
