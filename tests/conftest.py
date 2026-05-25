"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pytest


@pytest.fixture(scope="session")
def mitsuba_variant() -> str:
    from src.scene_utils import init_mitsuba

    return init_mitsuba(prefer="llvm_ad_rgb")


@pytest.fixture(scope="session")
def box_scene(mitsuba_variant: str) -> object:
    """Tiny synthetic Mitsuba scene: a closed inward-facing box with one light."""
    import mitsuba as mi

    return mi.load_dict(
        {
            "type": "scene",
            "integrator": {"type": "path"},
            "walls": {
                "type": "cube",
                "flip_normals": True,
                "to_world": mi.ScalarTransform4f.scale([3.0, 1.5, 3.0]),
                "bsdf": {
                    "type": "twosided",
                    "inner": {
                        "type": "diffuse",
                        "reflectance": {"type": "rgb", "value": [0.8, 0.7, 0.6]},
                    },
                },
            },
            "lamp": {
                "type": "sphere",
                "to_world": mi.ScalarTransform4f.translate([0.0, 1.3, 0.0]).scale(0.2),
                "emitter": {
                    "type": "area",
                    "radiance": {"type": "rgb", "value": [20.0, 20.0, 20.0]},
                },
            },
        }
    )


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=12345)


@pytest.fixture()
def tmp_output_dir(tmp_path: Path) -> Iterator[Path]:
    out = tmp_path / "out"
    out.mkdir()
    yield out
