"""Unit tests for src/camera_sampling.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.camera_sampling import (
    CameraPose,
    DepthFilterConfig,
    DepthFilterResult,
    UniformRandomSampler,
    WallWalkSampler,
    depth_validity,
    make_sampler,
    sample_cameras,
    sample_orientation,
    sample_position,
)
from src.scene_utils import SceneInfo, point_in_polygon


@pytest.fixture()
def y_up_info() -> SceneInfo:
    return SceneInfo(
        bbox_min=(-3.0, -1.5, -3.0),
        bbox_max=(3.0, 1.5, 3.0),
        up_axis="y",
        placement_min=(-2.5, 0.8, -2.5),
        placement_max=(2.5, 1.6, 2.5),
    )


def test_orientation_distribution_is_within_bounds(rng: np.random.Generator) -> None:
    yaws = np.empty(2000)
    pitches = np.empty(2000)
    for i in range(2000):
        yaws[i], pitches[i] = sample_orientation(rng, pitch_range_deg=(-25.0, 25.0))
    assert yaws.min() >= 0.0 and yaws.max() < 2.0 * np.pi
    assert pitches.min() >= np.deg2rad(-25.0) - 1e-9
    assert pitches.max() <= np.deg2rad(25.0) + 1e-9
    # yaw should be roughly uniform: histogram counts within ~25% of expected.
    counts, _ = np.histogram(yaws, bins=8, range=(0.0, 2.0 * np.pi))
    expected = 2000 / 8
    assert counts.min() > expected * 0.75
    assert counts.max() < expected * 1.25


def test_sample_position_within_placement_box(
    rng: np.random.Generator, y_up_info: SceneInfo
) -> None:
    for _ in range(200):
        p = sample_position(rng, y_up_info)
        for axis in range(3):
            assert y_up_info.placement_min[axis] <= p[axis] <= y_up_info.placement_max[axis]


def test_pose_forward_y_up_yaw0_pitch0_points_minus_z() -> None:
    p = CameraPose(position=(0, 1, 0), yaw=0.0, pitch=0.0, up_axis="y")
    assert np.allclose(p.forward(), (0.0, 0.0, -1.0), atol=1e-9)


def test_pose_forward_y_up_pitch_up() -> None:
    p = CameraPose(position=(0, 1, 0), yaw=0.0, pitch=np.pi / 4, up_axis="y")
    expected = np.array([0.0, np.sin(np.pi / 4), -np.cos(np.pi / 4)])
    assert np.allclose(p.forward(), expected, atol=1e-6)


def test_pose_target_default_distance_one() -> None:
    p = CameraPose(position=(1.0, 2.0, 3.0), yaw=0.0, pitch=0.0, up_axis="y")
    assert np.allclose(p.target(), (1.0, 2.0, 2.0), atol=1e-9)


def test_depth_validity_all_inf_rejected() -> None:
    d = np.full((8, 8), np.inf)
    r = depth_validity(d, DepthFilterConfig())
    assert not r.accepted
    assert r.reason == "too_much_inf"


def test_depth_validity_too_close_rejected() -> None:
    d = np.full((8, 8), 0.1)
    r = depth_validity(d, DepthFilterConfig(min_median_depth=0.3))
    assert not r.accepted
    assert r.reason in ("too_close", "too_much_near")


def test_depth_validity_normal_case_accepted() -> None:
    d = np.full((8, 8), 2.0)
    r = depth_validity(d, DepthFilterConfig())
    assert r.accepted
    assert r.reason == "ok"
    assert r.median_depth == pytest.approx(2.0)


def test_depth_validity_too_many_near_pixels_rejected() -> None:
    d = np.full((10, 10), 2.0)
    d[:, :2] = 0.1  # 20% of pixels closer than 0.2 m, > 5% threshold
    r = depth_validity(d, DepthFilterConfig())
    assert not r.accepted
    assert r.reason in ("too_close", "too_much_near")


def test_sample_cameras_filters_with_callbacks(
    rng: np.random.Generator, y_up_info: SceneInfo
) -> None:
    # An inside_room callback that accepts everything, and a depth render that
    # alternately returns "good" and "bad" depths.
    counter = {"n": 0}

    def inside(positions: np.ndarray) -> np.ndarray:
        return np.ones(positions.shape[0], dtype=bool)

    def render_depth(_pose: CameraPose) -> np.ndarray:
        counter["n"] += 1
        if counter["n"] % 2 == 0:
            return np.full((8, 8), 3.0)
        return np.full((8, 8), np.inf)

    poses, stats = sample_cameras(
        rng,
        y_up_info,
        num_cameras=5,
        inside_room=inside,
        render_depth=render_depth,
        depth_filter=DepthFilterConfig(),
        oversample=4,
    )
    assert len(poses) == 5
    assert stats.n_accepted == 5
    # Half of attempts are rejected -> attempts ~ 10
    assert stats.n_attempts >= 5


# ---- WallWalkSampler ---------------------------------------------------------


def _accept_all(positions: np.ndarray) -> np.ndarray:
    return np.ones(positions.shape[0], dtype=bool)


def _good_depth(_pose: CameraPose) -> np.ndarray:
    return np.full((8, 8), 3.0, dtype=np.float64)


def test_wall_walk_positions_inside_polygon(
    rng: np.random.Generator, y_up_info: SceneInfo
) -> None:
    # Rectangular floor matching y_up_info's horizontal extent.
    polygon = np.array(
        [[-2.5, -2.5], [2.5, -2.5], [2.5, 2.5], [-2.5, 2.5]],
        dtype=np.float64,
    )
    sampler = WallWalkSampler(
        info=y_up_info,
        rng=rng,
        inside_room=_accept_all,
        render_depth=_good_depth,
        depth_filter=DepthFilterConfig(),
        floor_polygon=polygon,
        dist_min=0.3,
        dist_max=1.0,
    )
    poses, stats = sampler.sample(num_cameras=20, oversample=4)
    assert len(poses) == 20
    # Every pose's (x, z) projection must lie strictly inside the polygon.
    for p in poses:
        xz = np.array([p.position[0], p.position[2]])
        assert point_in_polygon(xz, polygon)


def test_wall_walk_orientations_bias_inward(
    rng: np.random.Generator, y_up_info: SceneInfo
) -> None:
    """Inward-facing yaw: the camera's forward should point toward the polygon centroid."""
    polygon = np.array(
        [[-2.5, -2.5], [2.5, -2.5], [2.5, 2.5], [-2.5, 2.5]],
        dtype=np.float64,
    )
    sampler = WallWalkSampler(
        info=y_up_info,
        rng=rng,
        inside_room=_accept_all,
        render_depth=_good_depth,
        depth_filter=DepthFilterConfig(),
        floor_polygon=polygon,
        dist_min=0.3,
        dist_max=0.8,
        yaw_offset_range_deg=(-15.0, 15.0),  # tighten so the test is unambiguous
        pitch_range_deg=(-10.0, 10.0),
    )
    poses, _ = sampler.sample(num_cameras=40, oversample=4)
    centroid = np.array([0.0, 0.0])
    inward_aligned = 0
    for p in poses:
        xz = np.array([p.position[0], p.position[2]])
        to_centroid = centroid - xz
        to_centroid /= np.linalg.norm(to_centroid)
        # Forward vector in horizontal plane (h0=x, h1=z for y-up).
        f = p.forward()
        fxz = np.array([f[0], f[2]])
        if np.linalg.norm(fxz) > 1e-6:
            fxz /= np.linalg.norm(fxz)
            if float(np.dot(fxz, to_centroid)) > 0.5:
                inward_aligned += 1
    # With a 30° total yaw window centered on the inward normal, at least 80%
    # of forward vectors should align with the inward direction.
    assert inward_aligned >= int(0.8 * len(poses))


def test_wall_walk_respects_dist_min(
    rng: np.random.Generator, y_up_info: SceneInfo
) -> None:
    polygon = np.array(
        [[-2.5, -2.5], [2.5, -2.5], [2.5, 2.5], [-2.5, 2.5]],
        dtype=np.float64,
    )
    dist_min = 0.5
    sampler = WallWalkSampler(
        info=y_up_info,
        rng=rng,
        inside_room=_accept_all,
        render_depth=_good_depth,
        depth_filter=DepthFilterConfig(),
        floor_polygon=polygon,
        dist_min=dist_min,
        dist_max=1.5,
    )
    poses, _ = sampler.sample(num_cameras=30, oversample=4)
    # Every accepted pose must be at least dist_min from all four walls.
    for p in poses:
        x, _, z = p.position
        assert 2.5 - abs(x) >= dist_min - 1e-9
        assert 2.5 - abs(z) >= dist_min - 1e-9


def test_make_sampler_wall_walk_requires_polygon(
    rng: np.random.Generator, y_up_info: SceneInfo
) -> None:
    with pytest.raises(ValueError, match="floor_polygon"):
        make_sampler(
            "wall_walk",
            info=y_up_info,
            rng=rng,
            inside_room=_accept_all,
            render_depth=_good_depth,
            depth_filter=DepthFilterConfig(),
        )


def test_make_sampler_returns_correct_class(
    rng: np.random.Generator, y_up_info: SceneInfo
) -> None:
    s = make_sampler(
        "uniform",
        info=y_up_info,
        rng=rng,
        inside_room=_accept_all,
        render_depth=_good_depth,
        depth_filter=DepthFilterConfig(),
    )
    assert isinstance(s, UniformRandomSampler)

    polygon = np.array(
        [[-2.5, -2.5], [2.5, -2.5], [2.5, 2.5], [-2.5, 2.5]],
        dtype=np.float64,
    )
    s2 = make_sampler(
        "wall_walk",
        info=y_up_info,
        rng=rng,
        inside_room=_accept_all,
        render_depth=_good_depth,
        depth_filter=DepthFilterConfig(),
        floor_polygon=polygon,
    )
    assert isinstance(s2, WallWalkSampler)
