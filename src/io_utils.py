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

# Per-AOV subdirectories (all rendered by Blender; shared across backends).
#
# ``albedo``         = Cycles DiffCol pass (k_d)
# ``specular_albedo`` = Cycles GlossCol pass (k_s / F0)
# ``emission``       = Cycles Emit pass
# ``material_index`` = Cycles IndexMA pass (per-material segmentation)
# ``roughness`` / ``metallic`` = per-pixel shader-AOV maps
AOV_SUBDIRS: tuple[str, ...] = (
    "depth",
    "normal",
    "albedo",
    "specular_albedo",
    "emission",
    "material_index",
    "roughness",
    "metallic",
)
DEBUG_SUBDIRS: tuple[str, ...] = (
    "albedo_ldr",
    "depth_rgb",
    "normal_rgb",
)
ENVMAP_SUBDIR = "envmap"

# Backwards-compatible legacy alias kept so existing imports continue working
# during the refactor. New code should use ``aov_layout_subdirs()`` and the
# per-backend helpers below.
SUBDIRS: tuple[str, ...] = (
    "rgb",
    "rgb_hdr",
    *AOV_SUBDIRS,
    *DEBUG_SUBDIRS,
)

TRANSFORMS_FILENAME = "transforms.json"
METADATA_FILENAME = "metadata.json"
PREVIEW_FILENAME = "preview.png"

# Frame paths follow ``<subdir>/<i:04d>.<ext>``.
_FRAME_RE = re.compile(r"^(?P<idx>\d{4,})\.(exr|png)$")


# ---- Backend-aware layout helpers -------------------------------------------


def rgb_subdir(backend: str) -> str:
    """Return the LDR RGB subdir name for a beauty backend."""
    return f"rgb_{backend}" if backend not in ("legacy", "") else "rgb"


def rgb_hdr_subdir(backend: str) -> str:
    """Return the HDR EXR subdir name for a beauty backend."""
    return f"rgb_hdr_{backend}" if backend not in ("legacy", "") else "rgb_hdr"


def make_layout(
    output_dir: Path, backends: list[str], with_envmap: bool = False
) -> None:
    """Create the unified output directory tree.

    Per-backend RGB dirs (``rgb_<backend>/`` + ``rgb_hdr_<backend>/``) are
    created for every entry in ``backends``. All AOV dirs are shared
    regardless of backend. Debug-visualization dirs are always created.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for backend in backends:
        (output_dir / rgb_subdir(backend)).mkdir(parents=True, exist_ok=True)
        (output_dir / rgb_hdr_subdir(backend)).mkdir(parents=True, exist_ok=True)
    for sub in AOV_SUBDIRS:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)
    for sub in DEBUG_SUBDIRS:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)
    if with_envmap:
        (output_dir / ENVMAP_SUBDIR).mkdir(parents=True, exist_ok=True)


def existing_rendered_frames_for_backend(
    output_dir: Path, backend: str
) -> set[int]:
    """Frame indices for which the HDR beauty render already exists for ``backend``."""
    out: set[int] = set()
    hdr_dir = output_dir / rgb_hdr_subdir(backend)
    if not hdr_dir.exists():
        return out
    for p in hdr_dir.iterdir():
        m = _FRAME_RE.match(p.name)
        if m is not None and p.suffix == ".exr":
            out.add(int(m.group("idx")))
    return out


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
    # Force FreeImage EXR plugin; otherwise imageio's auto-pick can land on
    # the wrong plugin (e.g. 'SPE') and raise spurious frame-count errors.
    return np.asarray(
        imageio.imread(str(path), format="EXR-FI"),  # type: ignore[arg-type]
        dtype=np.float32,
    )


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
        "world_up_axis": up_axis,
        "depth_format": "z_depth_meters",
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
    index: int,
    transform_c2w: NDArray[Any],
    backends: list[str] | None = None,
    *,
    with_envmap: bool = False,
) -> dict[str, Any]:
    """Build a single ``frames`` entry for ``transforms.json``.

    When ``backends`` is None (legacy mode) the record uses the un-suffixed
    ``rgb/`` and ``rgb_hdr/`` paths. When ``backends`` is provided, per-backend
    paths (``rgb_<backend>/...``) are emitted in addition to the AOV paths.
    """
    mat = np.asarray(transform_c2w, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"transform must be 4x4; got {mat.shape}")
    stem = frame_stem(index)

    record: dict[str, Any] = {
        "transform_matrix": mat.tolist(),
        "depth_path": f"depth/{stem}.exr",
        "normal_path": f"normal/{stem}.exr",
        "albedo_path": f"albedo/{stem}.exr",
        "roughness_path": f"roughness/{stem}.exr",
        "metallic_path": f"metallic/{stem}.exr",
    }
    if backends:
        # Primary file_path / hdr_path use the first backend so single-backend
        # consumers don't need to special-case the format. Extra backends get
        # ``<key>_<backend>`` keys.
        primary = backends[0]
        record["file_path"] = f"{rgb_subdir(primary)}/{stem}"
        record["hdr_path"] = f"{rgb_hdr_subdir(primary)}/{stem}.exr"
        for b in backends:
            record[f"file_path_{b}"] = f"{rgb_subdir(b)}/{stem}"
            record[f"hdr_path_{b}"] = f"{rgb_hdr_subdir(b)}/{stem}.exr"
        # New AOV paths only exist in the unified pipeline.
        record["specular_albedo_path"] = f"specular_albedo/{stem}.exr"
        record["emission_path"] = f"emission/{stem}.exr"
        record["material_index_path"] = f"material_index/{stem}.exr"
        if with_envmap:
            record["envmap_path"] = f"{ENVMAP_SUBDIR}/{stem}.exr"
    else:
        record["file_path"] = f"rgb/{stem}"
        record["hdr_path"] = f"rgb_hdr/{stem}.exr"
    return record


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
