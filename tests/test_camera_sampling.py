"""Unit tests for src/camera_sampling.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.camera_sampling import (
    CameraPose,
    DepthFilterConfig,
    DepthFilterResult,
    depth_validity,
    sample_cameras,
    sample_orientation,
    sample_position,
)
from src.scene_utils import SceneInfo


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
