"""Tests for the 2D polygon helpers in :mod:`src.scene_utils`."""

from __future__ import annotations

import numpy as np
import pytest

from src.scene_utils import (
    minimum_bounding_rectangle,
    point_in_polygon,
    points_in_polygon,
)


def test_point_in_axis_aligned_square() -> None:
    poly = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    assert point_in_polygon(np.array([0.5, 0.5]), poly)
    assert not point_in_polygon(np.array([1.5, 0.5]), poly)
    assert not point_in_polygon(np.array([-0.1, 0.5]), poly)


def test_points_in_polygon_vectorised_matches_scalar() -> None:
    poly = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    pts = np.array(
        [
            [1.0, 1.0],   # inside
            [3.0, 1.0],   # right of polygon
            [-1.0, 1.0],  # left of polygon
            [1.0, 3.0],   # above
            [0.5, 0.5],   # inside
        ]
    )
    expected = np.array([True, False, False, False, True])
    np.testing.assert_array_equal(points_in_polygon(pts, poly), expected)
    for p, want in zip(pts, expected):
        assert point_in_polygon(p, poly) is bool(want)


def test_point_in_l_shaped_polygon() -> None:
    """Non-convex polygon — verifies the ray-casting algorithm handles concavities."""
    poly = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [0.0, 2.0],
        ]
    )
    assert point_in_polygon(np.array([0.5, 0.5]), poly)   # lower-left arm
    assert point_in_polygon(np.array([1.5, 0.5]), poly)   # lower-right arm
    assert point_in_polygon(np.array([0.5, 1.5]), poly)   # upper-left arm
    assert not point_in_polygon(np.array([1.5, 1.5]), poly)  # in the L's notch


def test_points_in_polygon_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        points_in_polygon(np.zeros((3, 3)), np.zeros((4, 2)))
    with pytest.raises(ValueError):
        points_in_polygon(np.zeros((3, 2)), np.zeros((2, 2)))


def test_mbr_of_axis_aligned_square_recovers_corners() -> None:
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]])
    corners = minimum_bounding_rectangle(pts)
    assert corners.shape == (4, 2)

    # All 4 corners of the unit square must be present (in some order).
    expected = {(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)}
    got = {(round(c[0], 6), round(c[1], 6)) for c in corners}
    assert got == expected


def test_mbr_of_rotated_rectangle_has_tight_area() -> None:
    """A 2x1 rectangle rotated by 30° should give an MBR area very close to 2."""
    theta = np.deg2rad(30.0)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    base = np.array(
        [
            [-1.0, -0.5],
            [1.0, -0.5],
            [1.0, 0.5],
            [-1.0, 0.5],
            # interior sample to make sure ConvexHull still finds the rect.
            [0.0, 0.0],
        ]
    )
    pts = base @ rot.T
    corners = minimum_bounding_rectangle(pts)

    e1 = np.linalg.norm(corners[1] - corners[0])
    e2 = np.linalg.norm(corners[2] - corners[1])
    area = float(e1 * e2)
    assert abs(area - 2.0) < 1e-6


def test_mbr_rejects_degenerate_input() -> None:
    with pytest.raises(ValueError):
        minimum_bounding_rectangle(np.zeros((2, 2)))
