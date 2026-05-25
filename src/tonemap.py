"""Per-scene exposure derivation, gamma curve, and LDR PNG export."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import imageio.v2 as imageio
import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class ExposureParams:
    key: float
    percentile: float
    exposure: float
    gamma: float


_LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


def luminance(hdr: NDArray[Any]) -> NDArray[np.float64]:
    """Return BT.709 luminance of an HDR image. Input shape ``(..., 3)``."""
    arr = np.asarray(hdr, dtype=np.float64)
    if arr.shape[-1] != 3:
        raise ValueError(f"hdr must have last dim 3; got shape {arr.shape}")
    return arr @ _LUMA_WEIGHTS


def compute_scene_exposure(
    hdr_images: Iterable[NDArray[Any]],
    key: float = 0.18,
    percentile: float = 90.0,
    gamma: float = 2.2,
) -> ExposureParams:
    """Compute a single exposure value for an entire scene.

    Stacks luminance across all frames, picks the ``percentile``-th percentile,
    and sets ``exposure = key / percentile_luma``.
    """
    lumas: list[NDArray[np.float64]] = []
    for img in hdr_images:
        lumas.append(luminance(img).reshape(-1))
    if not lumas:
        raise ValueError("compute_scene_exposure: no images provided")
    luma_all = np.concatenate(lumas)
    luma_pos = luma_all[luma_all > 0.0]
    if luma_pos.size == 0:
        raise ValueError("All HDR luminance values are <= 0; check renders.")
    p = float(np.percentile(luma_pos, percentile))
    if p <= 0.0:
        raise ValueError(f"Percentile luminance is non-positive: {p}")
    exposure = float(key / p)
    return ExposureParams(
        key=float(key), percentile=float(percentile), exposure=exposure, gamma=float(gamma)
    )


def tonemap(
    hdr: NDArray[Any], params: ExposureParams
) -> NDArray[np.uint8]:
    """Apply ``exposure -> gamma -> clip -> uint8`` to an HDR image."""
    arr = np.asarray(hdr, dtype=np.float64)
    scaled = np.clip(arr * params.exposure, 0.0, None)
    encoded = scaled ** (1.0 / params.gamma)
    encoded = np.clip(encoded, 0.0, 1.0)
    return (encoded * 255.0 + 0.5).astype(np.uint8)


def save_ldr_png(path: Path, ldr: NDArray[np.uint8]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(path), ldr)
