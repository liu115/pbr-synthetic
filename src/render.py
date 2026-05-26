"""Mitsuba sensor construction, beauty + AOV rendering, BSDF param probing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import mitsuba as mi
import numpy as np
from numpy.typing import NDArray

from src.camera_sampling import CameraPose
from src.io_utils import Intrinsics
from src.pose_utils import pose_to_c2w  # re-exported for backward compatibility

__all__ = [
    "AOVImages",
    "MaterialMaps",
    "RenderConfig",
    "derive_material_lut",
    "make_contact_sheet",
    "material_caveat_message",
    "materials_from_shape_index",
    "pose_to_c2w",
    "render_aov",
    "render_beauty",
    "render_depth_for_filter",
]

log = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class RenderConfig:
    intrinsics_full: Intrinsics
    intrinsics_debug: Intrinsics
    spp_beauty_full: int
    spp_beauty_debug: int
    spp_aov: int
    spp_filter: int
    filter_width: int
    filter_height: int
    max_depth: int
    debug: bool

    @property
    def beauty_intrinsics(self) -> Intrinsics:
        return self.intrinsics_debug if self.debug else self.intrinsics_full

    @property
    def spp_beauty(self) -> int:
        return self.spp_beauty_debug if self.debug else self.spp_beauty_full


@dataclass(slots=True, frozen=True)
class AOVImages:
    """Decoded AOV channels for one camera."""

    depth: NDArray[np.float32]
    normal: NDArray[np.float32]
    albedo: NDArray[np.float32]
    shape_index: NDArray[np.int32]


@dataclass(slots=True, frozen=True)
class MaterialMaps:
    roughness: NDArray[np.float32]
    metallic: NDArray[np.float32]


# AOV column layout when using the AOV integrator below: 0..2 = RGB from inner
# integrator, 3 = depth (t-distance), 4..6 = sh_normal, 7..9 = albedo, 10 =
# shape_index. Channels 0..2 are unused by the AOV pass (we get RGB from the
# beauty pass instead) but the AOV plugin still produces them.
_AOV_TOTAL_CHANNELS = 11
_AOV_DEPTH = slice(3, 4)
_AOV_NORMAL = slice(4, 7)
_AOV_ALBEDO = slice(7, 10)
_AOV_SHAPE = slice(10, 11)
_AOV_SPEC = "depth:depth,nn:sh_normal,alb:albedo,sid:shape_index"


def _build_sensor(
    pose: CameraPose,
    intrinsics: Intrinsics,
    spp: int,
    sample_seed_offset: int = 0,
) -> mi.Sensor:
    origin = list(pose.position)
    target = list(pose.target(distance=1.0))
    up = pose.world_up().tolist()
    sensor_dict: dict[str, Any] = {
        "type": "perspective",
        "fov": float(np.rad2deg(intrinsics.fov_x_rad)),
        "fov_axis": "x",
        "to_world": mi.ScalarTransform4f.look_at(
            origin=origin, target=target, up=up
        ),
        "film": {
            "type": "hdrfilm",
            "width": intrinsics.width,
            "height": intrinsics.height,
            "pixel_format": "rgb",
            "file_format": "openexr",
            "rfilter": {"type": "tent"},
        },
        "sampler": {
            "type": "independent",
            "sample_count": spp,
            "seed": sample_seed_offset,
        },
    }
    return mi.load_dict(sensor_dict)


def render_beauty(
    scene: mi.Scene,
    pose: CameraPose,
    intrinsics: Intrinsics,
    spp: int,
    max_depth: int,
    seed: int = 0,
) -> NDArray[np.float32]:
    """Path-traced HDR RGB in linear radiance. Returns ``(H, W, 3)``."""
    sensor = _build_sensor(pose, intrinsics, spp, sample_seed_offset=seed)
    integrator = mi.load_dict({"type": "path", "max_depth": max_depth})
    img = mi.render(scene, sensor=sensor, integrator=integrator, spp=spp, seed=seed)
    arr = np.asarray(img, dtype=np.float32)
    if arr.shape[-1] >= 3:
        return arr[..., :3].copy()
    raise RuntimeError(f"Unexpected beauty image shape {arr.shape}")


def render_aov(
    scene: mi.Scene,
    pose: CameraPose,
    intrinsics: Intrinsics,
    spp: int,
    max_depth: int,
    seed: int = 0,
) -> AOVImages:
    """Render depth, sh_normal, albedo, and shape_index AOVs."""
    sensor = _build_sensor(pose, intrinsics, spp, sample_seed_offset=seed + 1)
    integrator = mi.load_dict(
        {
            "type": "aov",
            "aovs": _AOV_SPEC,
            "integrator": {"type": "path", "max_depth": max_depth},
        }
    )
    img = mi.render(scene, sensor=sensor, integrator=integrator, spp=spp, seed=seed + 1)
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] < _AOV_TOTAL_CHANNELS:
        raise RuntimeError(
            f"AOV image has unexpected shape {arr.shape}; "
            f"expected (H, W, >= {_AOV_TOTAL_CHANNELS})"
        )
    depth = arr[..., _AOV_DEPTH][..., 0].copy()
    # Mitsuba writes 0 for non-hit; convert to +inf for downstream sanity.
    depth = np.where(depth > 0.0, depth, np.float32(np.inf))
    normal = arr[..., _AOV_NORMAL].copy()
    albedo = arr[..., _AOV_ALBEDO].copy()
    shape_index = arr[..., _AOV_SHAPE][..., 0].astype(np.int32)
    return AOVImages(
        depth=depth, normal=normal, albedo=albedo, shape_index=shape_index
    )


def render_depth_for_filter(
    scene: mi.Scene,
    pose: CameraPose,
    intrinsics: Intrinsics,
    spp: int,
    max_depth: int,
    seed: int = 0,
) -> NDArray[np.float32]:
    """Fast low-resolution depth render for the camera-validity filter."""
    sensor = _build_sensor(pose, intrinsics, spp, sample_seed_offset=seed + 2)
    integrator = mi.load_dict(
        {
            "type": "aov",
            "aovs": "depth:depth",
            "integrator": {"type": "path", "max_depth": max_depth},
        }
    )
    img = mi.render(scene, sensor=sensor, integrator=integrator, spp=spp, seed=seed + 2)
    arr = np.asarray(img, dtype=np.float32)
    # Layout: RGB (3) + depth (1) = 4 channels.
    if arr.ndim != 3 or arr.shape[-1] < 4:
        raise RuntimeError(f"Filter AOV image has unexpected shape {arr.shape}")
    depth = arr[..., 3].copy()
    depth = np.where(depth > 0.0, depth, np.float32(np.inf))
    return depth


# ---- Material LUT (per-shape roughness / metallic) ---------------------------

_AOV_CAVEAT_MSG = (
    "Per-shape roughness/metallic values are heuristic best-effort lookups based "
    "on BSDF type strings (diffuse -> (1.0, 0.0), conductor -> (0.1, 1.0), "
    "dielectric -> (0.0, 0.0), plastic -> (0.5, 0.0), principled -> reads "
    "'roughness'/'metallic' params when present). Spatially-varying BSDF "
    "parameters are NOT resolved per pixel. Untyped BSDFs default to (1.0, 0.0)."
)


def _bsdf_type_keyword(bsdf: object) -> str:
    """Return a coarse type keyword inferred from the BSDF's repr."""
    s = repr(bsdf).lower()
    for key in (
        "principled",
        "roughconductor",
        "roughdielectric",
        "roughplastic",
        "conductor",
        "dielectric",
        "plastic",
        "diffuse",
        "twosided",
        "thindielectric",
    ):
        if key in s:
            return key
    return "other"


def _scalar_param(params: object, key: str) -> float | None:
    """Try to read a scalar param value from a SceneParameters-like object."""
    try:
        if key not in params:  # type: ignore[operator]
            return None
        v = params[key]  # type: ignore[index]
    except Exception:
        return None
    try:
        a = np.asarray(v, dtype=np.float64).reshape(-1)
        if a.size == 0:
            return None
        return float(a.mean())
    except Exception:
        return None


def derive_material_lut(scene: mi.Scene) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Build per-shape (roughness, metallic) arrays.

    Returned arrays are 1D of length ``n_shapes + 1``. Index 0 is the "no hit"
    sentinel (NaN values). Index ``i + 1`` corresponds to ``scene.shapes()[i]``,
    matching the convention of Mitsuba's ``shape_index`` AOV.
    """
    shapes = scene.shapes()
    n = len(shapes)
    rough = np.full(n + 1, np.nan, dtype=np.float32)
    metal = np.full(n + 1, np.nan, dtype=np.float32)

    try:
        params = mi.traverse(scene)
    except Exception:
        params = None

    for i, shape in enumerate(shapes):
        bsdf = shape.bsdf()
        if bsdf is None:
            continue
        kw = _bsdf_type_keyword(bsdf)
        r: float | None = None
        m: float | None = None

        # Default by BSDF family.
        if kw == "diffuse" or kw == "twosided":
            r, m = 1.0, 0.0
        elif kw == "principled":
            r, m = 0.5, 0.0
        elif kw == "roughplastic":
            r, m = 0.5, 0.0
        elif kw == "plastic":
            r, m = 0.3, 0.0
        elif kw == "roughdielectric":
            r, m = 0.1, 0.0
        elif kw == "dielectric" or kw == "thindielectric":
            r, m = 0.0, 0.0
        elif kw == "roughconductor":
            r, m = 0.1, 1.0
        elif kw == "conductor":
            r, m = 0.0, 1.0
        else:
            r, m = 1.0, 0.0

        # Try to refine via parameter lookup.
        if params is not None:
            # SceneParameters keys look like "<shape_id>.bsdf.<param>".
            sid = shape.id() if shape.id() else f"shape_{i}"
            for k_metal in (
                f"{sid}.bsdf.metallic",
                f"{sid}.bsdf.brdf_0.metallic",
            ):
                v = _scalar_param(params, k_metal)
                if v is not None:
                    m = v
                    break
            for k_rough in (
                f"{sid}.bsdf.roughness",
                f"{sid}.bsdf.alpha",
                f"{sid}.bsdf.brdf_0.roughness",
                f"{sid}.bsdf.brdf_0.alpha",
            ):
                v = _scalar_param(params, k_rough)
                if v is not None:
                    r = v
                    break

        rough[i + 1] = float(r if r is not None else 1.0)
        metal[i + 1] = float(m if m is not None else 0.0)

    return rough, metal


def materials_from_shape_index(
    shape_index: NDArray[np.int32],
    rough_lut: NDArray[np.float32],
    metal_lut: NDArray[np.float32],
) -> MaterialMaps:
    """Broadcast a per-shape LUT to per-pixel material maps.

    Pixels with no hit (``shape_index <= 0`` or out-of-range) receive NaN.
    """
    si = shape_index.astype(np.int32)
    n_lut = rough_lut.shape[0]
    out_rough = np.full(si.shape, np.nan, dtype=np.float32)
    out_metal = np.full(si.shape, np.nan, dtype=np.float32)
    mask = (si > 0) & (si < n_lut)
    out_rough[mask] = rough_lut[si[mask]]
    out_metal[mask] = metal_lut[si[mask]]
    return MaterialMaps(roughness=out_rough, metallic=out_metal)


def material_caveat_message() -> str:
    return _AOV_CAVEAT_MSG


def make_contact_sheet(
    ldr_images: list[NDArray[np.uint8]], grid: tuple[int, int] = (4, 4)
) -> NDArray[np.uint8]:
    """Tile up to ``grid[0] * grid[1]`` LDR images into a single preview PNG."""
    if not ldr_images:
        raise ValueError("No images provided for contact sheet")
    rows, cols = grid
    sample = ldr_images[0]
    h, w = sample.shape[:2]
    channels = sample.shape[2] if sample.ndim == 3 else 1
    canvas = np.zeros((rows * h, cols * w, channels), dtype=np.uint8)
    for k, img in enumerate(ldr_images[: rows * cols]):
        r, c = divmod(k, cols)
        i = img
        if i.shape[:2] != (h, w):
            continue
        if i.ndim == 2:
            i = np.repeat(i[..., None], channels, axis=-1)
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = i
    if channels == 1:
        canvas = canvas[..., 0]
    return canvas
