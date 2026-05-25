"""Unit tests for src/tonemap.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.tonemap import (
    ExposureParams,
    compute_scene_exposure,
    luminance,
    tonemap,
)


def test_luminance_constant_image() -> None:
    img = np.ones((4, 4, 3), dtype=np.float32) * 2.0
    lum = luminance(img)
    assert np.allclose(lum, 2.0, atol=1e-6)


def test_luminance_only_red() -> None:
    img = np.zeros((2, 2, 3), dtype=np.float32)
    img[..., 0] = 1.0
    lum = luminance(img)
    assert np.allclose(lum, 0.2126, atol=1e-6)


def test_compute_scene_exposure_targets_key() -> None:
    # Uniform luminance 4.0 -> exposure such that exposure * 4.0 == key
    img = np.ones((4, 4, 3), dtype=np.float32) * 4.0
    params = compute_scene_exposure([img], key=0.18, percentile=90.0)
    assert params.exposure == pytest.approx(0.18 / 4.0, rel=1e-6)
    assert params.gamma == pytest.approx(2.2)


def test_compute_scene_exposure_uses_percentile() -> None:
    # Stack frames with varying brightness; 90th percentile of luminance ~= 9.
    arr = np.zeros((10, 10, 3), dtype=np.float32)
    arr[:, :9, :] = 1.0
    arr[:, 9:, :] = 100.0  # 10% of pixels are very bright -> 90th pct still ~1
    params = compute_scene_exposure([arr], key=0.18, percentile=90.0)
    # exposure should put the 90th percentile near 0.18.
    assert params.exposure * float(np.percentile(luminance(arr), 90)) == pytest.approx(
        0.18, rel=1e-3
    )


def test_compute_scene_exposure_empty_raises() -> None:
    with pytest.raises(ValueError):
        compute_scene_exposure([])


def test_tonemap_clip_to_white() -> None:
    img = np.ones((4, 4, 3), dtype=np.float32) * 100.0
    params = ExposureParams(key=0.18, percentile=90.0, exposure=1.0, gamma=2.2)
    out = tonemap(img, params)
    assert out.dtype == np.uint8
    assert (out == 255).all()


def test_tonemap_dark_pixel_stays_dark() -> None:
    img = np.zeros((4, 4, 3), dtype=np.float32)
    params = ExposureParams(key=0.18, percentile=90.0, exposure=1.0, gamma=2.2)
    out = tonemap(img, params)
    assert (out == 0).all()


def test_tonemap_midgray_roundtrip() -> None:
    # Linear 0.18 with exposure 1.0 -> srgb-encoded value ~ 0.46 -> ~118.
    img = np.full((4, 4, 3), 0.18, dtype=np.float32)
    params = ExposureParams(key=0.18, percentile=90.0, exposure=1.0, gamma=2.2)
    out = tonemap(img, params)
    expected = int(round(0.18 ** (1 / 2.2) * 255))
    assert (out == expected).all()
