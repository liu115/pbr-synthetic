"""Unit tests for src/viz_utils.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.viz_utils import (
    _turbo_polynomial,
    albedo_to_ldr,
    depth_to_rgb,
    normal_to_rgb,
)


def test_turbo_endpoints_are_dark_purple_and_dark_red() -> None:
    # Turbo starts near (0.19, 0.07, 0.23) and ends near (0.48, 0.02, 0.01).
    t = np.array([0.0, 1.0], dtype=np.float64)
    rgb = _turbo_polynomial(t)
    assert rgb[0, 0] < rgb[0, 2]  # at t=0, blue > red (cool end)
    assert rgb[1, 0] > rgb[1, 2]  # at t=1, red > blue (warm end)
    assert (rgb >= 0.0).all() and (rgb <= 1.0).all()


def test_turbo_monotonic_red_in_main_ramp() -> None:
    # Red ramps monotonically up between anchor t=0.32 (cyan) and t=0.7 (yellow).
    t = np.linspace(0.32, 0.7, 10)
    r = _turbo_polynomial(t)[:, 0]
    diffs = np.diff(r)
    assert (diffs >= -1e-6).all()


def test_normal_to_rgb_endpoints() -> None:
    n = np.array([[[-1.0, 0.0, 1.0]]], dtype=np.float32)
    out = normal_to_rgb(n)
    assert out.shape == (1, 1, 3)
    # -1 -> 0, 0 -> 128 (after rounding via +0.5), +1 -> 255
    assert out[0, 0, 0] == 0
    assert out[0, 0, 2] == 255
    assert 127 <= int(out[0, 0, 1]) <= 128


def test_normal_to_rgb_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        normal_to_rgb(np.zeros((4, 4), dtype=np.float32))


def test_normal_to_rgb_marks_zero_vector_black() -> None:
    n = np.zeros((2, 2, 3), dtype=np.float32)
    out = normal_to_rgb(n)
    assert (out == 0).all()


def test_albedo_to_ldr_gamma_midgray() -> None:
    a = np.full((4, 4, 3), 0.18, dtype=np.float32)
    out = albedo_to_ldr(a, gamma=2.2)
    expected = int(round(0.18 ** (1.0 / 2.2) * 255))
    assert (out == expected).all()


def test_albedo_to_ldr_clips_above_one() -> None:
    a = np.full((2, 2, 3), 5.0, dtype=np.float32)
    out = albedo_to_ldr(a)
    assert (out == 255).all()


def test_albedo_to_ldr_nan_becomes_black() -> None:
    a = np.full((2, 2, 3), np.nan, dtype=np.float32)
    out = albedo_to_ldr(a)
    assert (out == 0).all()


def test_depth_to_rgb_uniform_returns_finite() -> None:
    d = np.full((8, 8), 3.0, dtype=np.float32)
    out = depth_to_rgb(d)
    assert out.shape == (8, 8, 3) and out.dtype == np.uint8


def test_depth_to_rgb_void_pixels_are_white() -> None:
    d = np.full((4, 4), np.inf, dtype=np.float32)
    d[0, 0] = 2.0  # one finite pixel anchors the percentile range
    out = depth_to_rgb(d)
    # Void pixels are explicitly set to white (255, 255, 255).
    assert (out[1, 1] == 255).all()


def test_depth_to_rgb_all_void_is_all_white() -> None:
    d = np.full((4, 4), np.inf, dtype=np.float32)
    out = depth_to_rgb(d)
    assert (out == 255).all()


def test_depth_to_rgb_near_pixel_is_blue_end_of_turbo() -> None:
    d = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    out = depth_to_rgb(d, lo_pct=0.0, hi_pct=100.0)
    # Near (small depth) -> low turbo value -> blue dominant.
    near, far = out[0, 0], out[0, -1]
    assert int(near[2]) > int(near[0])   # blue > red at the near end
    assert int(far[0]) > int(far[2])     # red > blue at the far end
