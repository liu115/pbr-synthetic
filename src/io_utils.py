"""Output directory layout, JSON schemas, EXR helpers, resume logic."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)

# Output subdirectory names. The first block holds dataset outputs referenced
# by ``transforms.json``; the second block holds debug visualizations only.
SUBDIRS: tuple[str, ...] = (
    "rgb",
    "rgb_hdr",
    "depth",
    "normal",
    "albedo",
    "roughness",
    "metallic",
    # Debug-only LDR PNGs for quick visual inspection.
    "albedo_ldr",
    "depth_rgb",
    "normal_rgb",
)

TRANSFORMS_FILENAME = "transforms.json"
METADATA_FILENAME = "metadata.json"
PREVIEW_FILENAME = "preview.png"

# Frame paths follow ``<subdir>/<i:04d>.<ext>``.
_FRAME_RE = re.compile(r"^(?P<idx>\d{4,})\.(exr|png)$")


@dataclass(slots=True, frozen=True)
class Intrinsics:
    fov_x_rad: float
    fl_x: float
    fl_y: float
    cx: float
    cy: float
    width: int
    height: int


def compute_intrinsics(fov_x_deg: float, width: int, height: int) -> Intrinsics:
    """Pinhole intrinsics for a horizontal FOV in degrees, square pixels, centered."""
    fov_x = float(np.deg2rad(fov_x_deg))
    fl_x = width / (2.0 * float(np.tan(fov_x / 2.0)))
    fl_y = fl_x
    return Intrinsics(
        fov_x_rad=fov_x,
        fl_x=fl_x,
        fl_y=fl_y,
        cx=width / 2.0,
        cy=height / 2.0,
        width=width,
        height=height,
    )


def make_output_layout(output_dir: Path) -> None:
    """Create the output directory and all per-AOV subdirectories."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for sub in SUBDIRS:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)


def frame_stem(index: int) -> str:
    return f"{index:04d}"


def existing_rendered_frames(output_dir: Path) -> set[int]:
    """Return frame indices for which the HDR beauty render already exists."""
    out: set[int] = set()
    hdr_dir = output_dir / "rgb_hdr"
    if not hdr_dir.exists():
        return out
    for p in hdr_dir.iterdir():
        m = _FRAME_RE.match(p.name)
        if m is not None and p.suffix == ".exr":
            out.add(int(m.group("idx")))
    return out


# FreeImage EXR plugin flags. EXR_FLOAT = 1 forces 32-bit FP storage instead of
# the default 16-bit half-precision (which would lose precision for HDR
# radiance and for very small depth differences).
_EXR_FLOAT = 1


def write_exr(path: Path, array: NDArray[Any]) -> None:
    """Save a float array (HxW or HxWxC) as a 32-bit OpenEXR file."""
    arr = np.asarray(array, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(path), arr, flags=_EXR_FLOAT)


def read_exr(path: Path) -> NDArray[np.float32]:
    return np.asarray(imageio.imread(str(path)), dtype=np.float32)


def write_png_uint8(path: Path, array: NDArray[np.uint8]) -> None:
    arr = np.asarray(array, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(path), arr)


def write_transforms_json(
    path: Path,
    intrinsics: Intrinsics,
    up_axis: str,
    frames: list[dict[str, Any]],
) -> None:
    payload: dict[str, Any] = {
        "coordinate_convention": "opencv",
        "up_axis": up_axis,
        "camera_angle_x": intrinsics.fov_x_rad,
        "fl_x": intrinsics.fl_x,
        "fl_y": intrinsics.fl_y,
        "cx": intrinsics.cx,
        "cy": intrinsics.cy,
        "w": intrinsics.width,
        "h": intrinsics.height,
        "frames": frames,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def build_frame_record(
    index: int, transform_c2w: NDArray[Any]
) -> dict[str, Any]:
    """Build a single ``frames`` entry for ``transforms.json``."""
    mat = np.asarray(transform_c2w, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"transform must be 4x4; got {mat.shape}")
    stem = frame_stem(index)
    return {
        "file_path": f"rgb/{stem}",
        "transform_matrix": mat.tolist(),
        "depth_path": f"depth/{stem}.exr",
        "normal_path": f"normal/{stem}.exr",
        "albedo_path": f"albedo/{stem}.exr",
        "roughness_path": f"roughness/{stem}.exr",
        "metallic_path": f"metallic/{stem}.exr",
        "hdr_path": f"rgb_hdr/{stem}.exr",
    }


def write_metadata_json(path: Path, metadata: dict[str, Any]) -> None:
    payload = {"render_datetime_utc": datetime.now(timezone.utc).isoformat()}
    payload.update(metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def _json_default(obj: object) -> object:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "as_dict"):
        return obj.as_dict()  # type: ignore[attr-defined]
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
