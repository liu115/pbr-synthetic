"""Shared colormap / encoding utilities for debug PNGs and the live viewer."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


# Approximation of Google's "turbo" colormap by piecewise-linear interpolation
# over 7 anchor points sampled from the published LUT
# (https://ai.googleblog.com/2019/08/turbo-improved-rainbow-colormap-for.html).
# Cheaper than embedding the 256-entry LUT and exact at the endpoints.
_TURBO_ANCHOR_T = np.array(
    [0.00, 0.13, 0.32, 0.50, 0.70, 0.85, 1.00], dtype=np.float64
)
_TURBO_ANCHOR_RGB = np.array(
    [
        [0.19, 0.07, 0.23],  # dark purple
        [0.27, 0.32, 0.81],  # blue
        [0.13, 0.79, 0.97],  # cyan
        [0.48, 0.96, 0.41],  # green
        [0.96, 0.79, 0.23],  # yellow
        [0.96, 0.40, 0.10],  # orange
        [0.48, 0.02, 0.01],  # dark red
    ],
    dtype=np.float64,
)


def _turbo_polynomial(t: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apply the turbo colormap to ``t`` in [0, 1]. Returns ``(..., 3)`` in [0, 1]."""
    x = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    r = np.interp(x, _TURBO_ANCHOR_T, _TURBO_ANCHOR_RGB[:, 0])
    g = np.interp(x, _TURBO_ANCHOR_T, _TURBO_ANCHOR_RGB[:, 1])
    b = np.interp(x, _TURBO_ANCHOR_T, _TURBO_ANCHOR_RGB[:, 2])
    return np.stack([r, g, b], axis=-1)


def albedo_to_ldr(albedo: NDArray[Any], gamma: float = 2.2) -> NDArray[np.uint8]:
    """sRGB-encode a linear-light albedo image and return uint8.

    Albedo is a linear reflectance (often in [0, 1]). To view it in a normal
    image viewer we apply a ``1/gamma`` curve and clip. NaN pixels (no-hit
    rays) become black.
    """
    arr = np.asarray(albedo, dtype=np.float64)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.clip(arr, 0.0, 1.0)
    encoded = arr ** (1.0 / float(gamma))
    out: NDArray[np.uint8] = np.clip(encoded * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
    return out


def depth_to_rgb(
    depth: NDArray[Any],
    lo_pct: float = 2.0,
    hi_pct: float = 98.0,
) -> NDArray[np.uint8]:
    """Visualize a depth image as turbo-colormapped RGB uint8.

    Normalization is per-image: the ``lo_pct``..``hi_pct`` percentile of
    finite positive values maps to [0, 1] before applying turbo. Non-finite or
    non-positive pixels (no-hit / void) map to white so they're visually
    distinct from valid depth values.
    """
    d = np.asarray(depth, dtype=np.float64)
    finite_mask = np.isfinite(d) & (d > 0.0)
    finite = d[finite_mask]
    if finite.size == 0:
        return np.full((*d.shape, 3), 255, dtype=np.uint8)
    lo = float(np.percentile(finite, lo_pct))
    hi = float(np.percentile(finite, hi_pct))
    span = max(hi - lo, 1e-6)
    norm = np.clip((d - lo) / span, 0.0, 1.0)
    rgb = _turbo_polynomial(norm)
    out = (rgb * 255.0 + 0.5).astype(np.uint8)
    # Void pixels -> white (255, 255, 255).
    out[~finite_mask] = 255
    return out


def normal_to_rgb(normal: NDArray[Any]) -> NDArray[np.uint8]:
    """Encode a world-space normal map as RGB uint8 via ``(n + 1) / 2``.

    Input shape ``(H, W, 3)``. Pixels with non-finite components or all-zero
    vectors become black.
    """
    n = np.asarray(normal, dtype=np.float64)
    if n.ndim != 3 or n.shape[-1] != 3:
        raise ValueError(f"normal must have shape (H, W, 3); got {n.shape}")
    bad = ~np.isfinite(n).all(axis=-1)
    bad |= (np.abs(n).sum(axis=-1) < 1e-6)
    encoded = np.clip((n * 0.5 + 0.5) * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
    encoded[bad] = 0
    return encoded
