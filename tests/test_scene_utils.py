"""Unit tests for src/scene_utils.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.scene_utils import (
    SceneInfo,
    cast_rays,
    compute_bbox,
    derive_scene_info,
    detect_up_axis,
    inside_room_test,
)


def test_compute_bbox_matches_known_box(box_scene: object) -> None:
    bb_min, bb_max = compute_bbox(box_scene)
    assert np.allclose(bb_min, (-3.0, -1.5, -3.0), atol=1e-3)
    assert np.allclose(bb_max, (3.0, 1.5 + 0.2, 3.0), atol=0.25)
    # (lamp at y=1.3 with radius 0.2 stretches the bbox slightly past 1.5)


@pytest.mark.parametrize(
    "bb_min,bb_max,expected",
    [
        ((-5, -1, -5), (5, 1, 5), "y"),
        ((-1, -5, -5), (1, 5, 5), "x"),
        ((-5, -5, -1), (5, 5, 1), "z"),
    ],
)
def test_detect_up_axis_picks_smallest_extent(
    bb_min: tuple[float, float, float],
    bb_max: tuple[float, float, float],
    expected: str,
) -> None:
    assert detect_up_axis(bb_min, bb_max) == expected


def test_derive_scene_info_y_up(box_scene: object) -> None:
    info = derive_scene_info(
        box_scene,
        height_range=(0.5, 1.2),
        placement_margin=0.3,
        height_margin=0.1,
    )
    assert info.up_axis == "y"
    # XZ shrunk by margin
    assert info.placement_min[0] == pytest.approx(-2.7, abs=0.1)
    assert info.placement_max[0] == pytest.approx(2.7, abs=0.1)
    assert info.placement_min[2] == pytest.approx(-2.7, abs=0.1)
    # Height range intersected with bbox
    assert info.placement_min[1] == pytest.approx(0.5)
    assert info.placement_max[1] == pytest.approx(1.2)


def test_derive_scene_info_rejects_empty_region(box_scene: object) -> None:
    with pytest.raises(ValueError):
        derive_scene_info(
            box_scene,
            height_range=(10.0, 20.0),  # Above the scene.
            placement_margin=0.3,
            height_margin=0.1,
        )


def test_cast_rays_hits_known_wall(box_scene: object) -> None:
    origins = np.array([[0.0, 0.0, 0.0]])
    directions = np.array([[1.0, 0.0, 0.0]])  # Toward +X wall at x=3.
    distances, valid = cast_rays(box_scene, origins, directions)
    assert valid[0]
    assert distances[0] == pytest.approx(3.0, abs=0.05)


def test_inside_room_test_classifies_correctly(box_scene: object) -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],     # Inside.
            [100.0, 0.0, 0.0],   # Outside (far +X).
            [0.0, 100.0, 0.0],   # Outside (far +Y).
        ]
    )
    inside = inside_room_test(box_scene, positions, up_axis="y", max_room_extent=10.0)
    assert inside.tolist() == [True, False, False]


def test_scene_info_horizontal_axes_and_height() -> None:
    info = SceneInfo(
        bbox_min=(-1, -2, -3),
        bbox_max=(1, 2, 3),
        up_axis="y",
        placement_min=(-0.5, -1.5, -2.5),
        placement_max=(0.5, 1.5, 2.5),
    )
    assert info.horizontal_axes == (0, 2)
    assert info.height_range == (-1.5, 1.5)
