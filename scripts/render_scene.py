"""Unified CLI: render a Mitsuba indoor scene with Mitsuba beauty, Blender beauty, or both.

AOVs (depth, normal, albedo, specular_albedo, roughness, metallic, emission,
material_index) and optional per-pixel envmaps are always produced by
Blender Cycles. The ``--backend`` flag controls only which renderer (or
both) produces the photo-real RGB output. ``--sampler`` picks between
uniform and FIPT-style wall-walk camera placement. ``--remove-transparent``
strips dielectric/thindielectric BSDFs from the scene before any rendering.

Output schema (single directory; backend-suffixed RGB, shared AOVs)::

    <output>/
      scene.ply
      preview.png
      transforms.json
      metadata.json
      rgb_<backend>/      0000.png      # one or two of {mitsuba, blender}
      rgb_hdr_<backend>/  0000.exr
      depth/   normal/   albedo/   specular_albedo/   roughness/   metallic/
      emission/   material_index/
      envmap/ (optional)
"""

from __future__ import annotations

import argparse
import logging
import platform
import random
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from numpy.typing import NDArray

from src.camera_sampling import (
    CameraPose,
    DepthFilterConfig,
    make_sampler,
)
from src.envmap_render import EnvmapConfig, render_envmaps_for_frame
from src.io_utils import (
    METADATA_FILENAME,
    PREVIEW_FILENAME,
    TRANSFORMS_FILENAME,
    Intrinsics,
    build_frame_record,
    compute_intrinsics,
    existing_rendered_frames_for_backend,
    frame_stem,
    make_layout,
    rgb_hdr_subdir,
    rgb_subdir,
    write_exr,
    write_metadata_json,
    write_png_uint8,
    write_transforms_json,
)
from src.mesh_utils import export_colored_mesh_ply
from src.pose_utils import pose_to_c2w_opencv
from src.render import (
    RenderConfig,
    make_contact_sheet,
    render_beauty,
)
from src.render_blender import (
    render_aov_blender,
    render_beauty_blender,
    render_depth_for_filter_blender,
)
from src.scene_blender import (
    derive_scene_info_blender,
    init_blender,
    inside_room_test_blender,
    load_scene_blender,
)
from src.scene_utils import (
    SceneInfo,
    UpAxis,
    extract_floor_polygon,
    init_mitsuba,
    load_scene,
    parse_scene_up_axis_from_xml,
)
from src.tonemap import (
    ExposureParams,
    compute_scene_exposure,
    tonemap,
)
from src.transparent_filter import FilterReport, filter_transparent_scene
from src.viz_utils import (
    albedo_to_ldr,
    depth_to_rgb,
    normal_to_rgb,
)

log = logging.getLogger(__name__)


# ---- Defaults ----------------------------------------------------------------

Backend = Literal["mitsuba", "blender", "both"]

DEFAULT_FULL_W, DEFAULT_FULL_H = 640, 480
DEFAULT_DEBUG_W, DEFAULT_DEBUG_H = 160, 120
DEFAULT_FILTER_W, DEFAULT_FILTER_H = 160, 90

DEFAULT_SPP_MITSUBA_FULL = 2048
DEFAULT_SPP_MITSUBA_DEBUG = 64
DEFAULT_SPP_BLENDER_FULL = 256       # Cycles + denoising converges much faster
DEFAULT_SPP_BLENDER_DEBUG = 32
DEFAULT_SPP_AOV = 16                 # AOVs are deterministic; low SPP is fine
DEFAULT_SPP_FILTER = 4               # depth-validity filter
DEFAULT_SPP_ENVMAP = 32

DEFAULT_FOV = 60.0
DEFAULT_NUM_CAMERAS = 200
DEFAULT_MAX_DEPTH = 8

DEFAULT_PLACEMENT_MARGIN = 0.5
DEFAULT_HEIGHT_RANGE = (0.8, 1.8)
DEFAULT_HEIGHT_MARGIN = 0.3
DEFAULT_OVERSAMPLE = 4
DEFAULT_PITCH_RANGE_DEG = (-25.0, 25.0)

DEFAULT_ENVMAP_PATCH_SIZE = 40
DEFAULT_ENVMAP_HEIGHT = 64
DEFAULT_ENVMAP_WIDTH = 128


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render an indoor scene to a multi-view PBR/IR dataset."
    )
    p.add_argument("--scene", type=Path, required=False, help="Path to scene XML.")
    p.add_argument("--output", type=Path, required=False,
                   help="Output directory for this scene.")
    p.add_argument("--num-cameras", type=int, default=DEFAULT_NUM_CAMERAS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--debug", action="store_true",
                   help="Use low resolution + low SPP for fast iteration.")
    p.add_argument("--fov", type=float, default=DEFAULT_FOV)
    p.add_argument(
        "--backend", choices=["mitsuba", "blender", "both"], default="blender",
        help="Which renderer produces the photo-real RGB output. AOVs are "
             "always rendered with Blender Cycles. Default 'blender'.",
    )
    p.add_argument(
        "--sampler", choices=["uniform", "wall_walk"], default="uniform",
        help="Camera-pose sampler. 'wall_walk' is FIPT-style: walk along each "
             "room edge and bias cameras to face the interior.",
    )
    p.add_argument(
        "--remove-transparent", action="store_true",
        help="Strip dielectric/thindielectric/null BSDFs and their shapes from "
             "the scene before rendering (FIPT-style preprocess).",
    )
    p.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    p.add_argument("--oversample", type=int, default=DEFAULT_OVERSAMPLE)
    p.add_argument("--placement-margin", type=float, default=DEFAULT_PLACEMENT_MARGIN)
    p.add_argument("--height-min", type=float, default=DEFAULT_HEIGHT_RANGE[0])
    p.add_argument("--height-max", type=float, default=DEFAULT_HEIGHT_RANGE[1])
    p.add_argument("--height-margin", type=float, default=DEFAULT_HEIGHT_MARGIN)
    p.add_argument("--pitch-min-deg", type=float, default=DEFAULT_PITCH_RANGE_DEG[0])
    p.add_argument("--pitch-max-deg", type=float, default=DEFAULT_PITCH_RANGE_DEG[1])
    p.add_argument("--up-axis", choices=["x", "y", "z"], default=None)
    # WallWalk-specific
    p.add_argument("--wall-walk-dist-min", type=float, default=0.4)
    p.add_argument("--wall-walk-dist-max", type=float, default=2.5)
    p.add_argument("--yaw-offset-min-deg", type=float, default=-60.0)
    p.add_argument("--yaw-offset-max-deg", type=float, default=60.0)
    # Depth filter thresholds
    p.add_argument("--filter-max-inf", type=float, default=0.01)
    p.add_argument("--filter-min-median", type=float, default=0.8)
    p.add_argument("--filter-near-threshold", type=float, default=0.3)
    p.add_argument("--filter-max-near", type=float, default=0.05)
    # Tonemap params
    p.add_argument("--tonemap-key", type=float, default=0.18)
    p.add_argument("--tonemap-percentile", type=float, default=90.0)
    p.add_argument("--tonemap-gamma", type=float, default=2.2)
    p.add_argument("--scene-config", type=Path, default=None)
    p.add_argument("--no-ply", action="store_true")
    p.add_argument("--only-ply", action="store_true")
    p.add_argument("--tessellate-spacing", type=float, default=0.10)
    p.add_argument("--no-tessellate", action="store_true")
    p.add_argument("--simplify-dense-meshes", action="store_true")
    p.add_argument("--simplify-vertex-threshold", type=int, default=100_000)
    # Mitsuba-specific
    p.add_argument("--mitsuba-variant", type=str, default=None)
    # Blender-specific
    p.add_argument("--cycles-device", choices=["OPTIX", "CUDA", "CPU"], default="OPTIX")
    p.add_argument("--denoiser", choices=["OPTIX", "OPENIMAGEDENOISE", "NONE"], default="OPTIX")
    p.add_argument("--roughplastic-coat-weight", type=float, default=0.8,
                   help="Coat weight for the mitsuba-blender Principled-BSDF mapping "
                        "of roughplastic BSDFs. Lower values reduce specular gloss.")
    # Envmap
    p.add_argument("--no-envmap", action="store_true",
                   help="Skip per-pixel envmap rendering (default off in --debug).")
    p.add_argument("--envmap", action="store_true",
                   help="Force envmap rendering even in --debug.")
    p.add_argument("--envmap-patch-size", type=int, default=DEFAULT_ENVMAP_PATCH_SIZE)
    p.add_argument("--envmap-resolution", type=int, nargs=2,
                   default=[DEFAULT_ENVMAP_HEIGHT, DEFAULT_ENVMAP_WIDTH],
                   metavar=("H", "W"))
    p.add_argument("--envmap-spp", type=int, default=DEFAULT_SPP_ENVMAP)
    p.add_argument("--install-addon-only", action="store_true",
                   help="Install/upgrade the mitsuba-blender addon, then exit.")
    return p.parse_args(argv)


def _load_scene_config(path: Path | None, args: argparse.Namespace) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    candidates: list[Path] = []
    if path is not None:
        candidates.append(path)
    if args.scene is not None:
        auto = args.scene.parent / "scene_config.yaml"
        if auto.exists():
            candidates.append(auto)
    for p in candidates:
        if p.exists():
            with p.open() as f:
                data = yaml.safe_load(f) or {}
            cfg.update(data)
            log.info("Loaded scene config: %s", p)
    return cfg


def _apply_config_overrides(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    if not cfg:
        return
    mapping = {
        "fov": "fov",
        "up_axis": "up_axis",
        "placement_margin": "placement_margin",
        "height_min": "height_min",
        "height_max": "height_max",
        "height_margin": "height_margin",
        "pitch_min_deg": "pitch_min_deg",
        "pitch_max_deg": "pitch_max_deg",
        "max_depth": "max_depth",
    }
    for cfg_key, attr in mapping.items():
        if cfg_key in cfg:
            setattr(args, attr, cfg[cfg_key])
            log.info("Override %s = %r", attr, cfg[cfg_key])


def _backends_active(args: argparse.Namespace) -> list[str]:
    if args.backend == "both":
        return ["mitsuba", "blender"]
    return [args.backend]


def _envmap_enabled(args: argparse.Namespace) -> bool:
    if args.no_envmap:
        return False
    if args.envmap:
        return True
    return not args.debug


def _set_seeds(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    return np.random.default_rng(seed)


def _build_render_config(args: argparse.Namespace) -> RenderConfig:
    intr_full = compute_intrinsics(args.fov, DEFAULT_FULL_W, DEFAULT_FULL_H)
    intr_debug = compute_intrinsics(args.fov, DEFAULT_DEBUG_W, DEFAULT_DEBUG_H)
    # We piggy-back on RenderConfig for resolution + max_depth; the SPP fields
    # here are unused by the unified loop (it picks per-backend SPP below).
    return RenderConfig(
        intrinsics_full=intr_full,
        intrinsics_debug=intr_debug,
        spp_beauty_full=DEFAULT_SPP_MITSUBA_FULL,
        spp_beauty_debug=DEFAULT_SPP_MITSUBA_DEBUG,
        spp_aov=DEFAULT_SPP_AOV,
        spp_filter=DEFAULT_SPP_FILTER,
        filter_width=DEFAULT_FILTER_W,
        filter_height=DEFAULT_FILTER_H,
        max_depth=args.max_depth,
        debug=args.debug,
    )


def _spp_beauty(backend: str, args: argparse.Namespace) -> int:
    if backend == "mitsuba":
        return DEFAULT_SPP_MITSUBA_DEBUG if args.debug else DEFAULT_SPP_MITSUBA_FULL
    if backend == "blender":
        return DEFAULT_SPP_BLENDER_DEBUG if args.debug else DEFAULT_SPP_BLENDER_FULL
    raise ValueError(f"unknown backend {backend!r}")


# ---- Sampling ---------------------------------------------------------------


def _sample_phase(
    info: SceneInfo,
    rng: np.random.Generator,
    rcfg: RenderConfig,
    args: argparse.Namespace,
    scene_xml_for_polygon: Path,
) -> tuple[list[CameraPose], dict[str, Any]]:
    filter_intr = compute_intrinsics(args.fov, rcfg.filter_width, rcfg.filter_height)

    def inside(positions: NDArray[np.float64]) -> NDArray[np.bool_]:
        return inside_room_test_blender(positions, up_axis=info.up_axis)

    def render_depth_cb(pose: CameraPose) -> NDArray[np.float64]:
        depth = render_depth_for_filter_blender(
            pose, filter_intr,
            spp=rcfg.spp_filter, max_depth=rcfg.max_depth,
        )
        return depth.astype(np.float64)

    filter_cfg = DepthFilterConfig(
        max_infinity_ratio=args.filter_max_inf,
        min_median_depth=args.filter_min_median,
        near_threshold=args.filter_near_threshold,
        max_near_ratio=args.filter_max_near,
    )

    floor_polygon: NDArray[np.float64] | None = None
    if args.sampler == "wall_walk":
        floor_polygon = extract_floor_polygon(scene_xml_for_polygon, info.up_axis)

    sampler = make_sampler(
        args.sampler,
        info=info,
        rng=rng,
        inside_room=inside,
        render_depth=render_depth_cb,
        depth_filter=filter_cfg,
        pitch_range_deg=(args.pitch_min_deg, args.pitch_max_deg),
        floor_polygon=floor_polygon,
        wall_walk_dist_min=args.wall_walk_dist_min,
        wall_walk_dist_max=args.wall_walk_dist_max,
        yaw_offset_range_deg=(args.yaw_offset_min_deg, args.yaw_offset_max_deg),
    )

    t0 = time.time()
    poses, stats = sampler.sample(
        num_cameras=args.num_cameras, oversample=args.oversample,
    )
    elapsed = time.time() - t0
    log.info(
        "Sampled %d/%d cameras with sampler=%s in %.1fs. Rejections: %s",
        len(poses), args.num_cameras, args.sampler, elapsed, dict(stats.by_reason),
    )
    return poses, {
        "n_attempts": stats.n_attempts,
        "n_accepted": stats.n_accepted,
        "by_reason": dict(stats.by_reason),
        "sampling_seconds": round(elapsed, 2),
    }


# ---- Preview ----------------------------------------------------------------


def _render_preview(
    poses: list[CameraPose],
    rcfg: RenderConfig,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    if not poses:
        return
    placeholder = ExposureParams(key=0.18, percentile=90.0, exposure=0.5, gamma=2.2)
    thumbs: list[NDArray[np.uint8]] = []
    for i, pose in enumerate(poses[:16]):
        hdr = render_beauty_blender(
            pose, rcfg.intrinsics_debug,
            spp=DEFAULT_SPP_BLENDER_DEBUG, max_depth=rcfg.max_depth,
            seed=args.seed + i,
        )
        thumbs.append(tonemap(hdr, placeholder))
    sheet = make_contact_sheet(thumbs, grid=(4, 4))
    write_png_uint8(output_dir / PREVIEW_FILENAME, sheet)
    log.info("Preview written: %s", output_dir / PREVIEW_FILENAME)


# ---- Per-frame loop ---------------------------------------------------------


def _render_frame_beauty(
    backend: str, pose: CameraPose, intrinsics: Intrinsics,
    args: argparse.Namespace, rcfg: RenderConfig,
    mi_scene: Any, seed: int,
) -> NDArray[np.float32]:
    if backend == "mitsuba":
        if mi_scene is None:
            raise RuntimeError("Mitsuba scene not initialised but backend=mitsuba")
        # Poses come from Blender's scene info (Z-up after the addon's axis
        # flip). The Mitsuba scene was loaded directly from the original
        # Y-up XML, so we tell render_beauty to undo the flip on the camera.
        return render_beauty(
            mi_scene, pose, intrinsics,
            spp=_spp_beauty("mitsuba", args), max_depth=rcfg.max_depth, seed=seed,
            pose_frame="blender",
        )
    if backend == "blender":
        return render_beauty_blender(
            pose, intrinsics,
            spp=_spp_beauty("blender", args), max_depth=rcfg.max_depth, seed=seed,
        )
    raise ValueError(f"unknown backend {backend!r}")


def _render_phase(
    backends: list[str],
    poses: list[CameraPose],
    rcfg: RenderConfig,
    args: argparse.Namespace,
    output_dir: Path,
    mi_scene: Any,
    envmap_cfg: EnvmapConfig | None,
) -> dict[str, list[NDArray[np.float32]]]:
    """Render each pose: beauty per backend + shared AOVs from Blender."""
    beauty_intr = rcfg.beauty_intrinsics

    # Resume bookkeeping: per-backend rgb_hdr_<backend> dirs.
    skip_per_backend = {
        b: existing_rendered_frames_for_backend(output_dir, b) for b in backends
    }
    for b in backends:
        if skip_per_backend[b]:
            log.info(
                "Resuming: %d existing %s frames will be skipped.",
                len(skip_per_backend[b]), b,
            )

    placeholder_exposure = ExposureParams(
        key=args.tonemap_key, percentile=args.tonemap_percentile,
        exposure=1.0, gamma=args.tonemap_gamma,
    )

    hdr_per_backend: dict[str, list[NDArray[np.float32]]] = {b: [] for b in backends}

    from src.io_utils import read_exr  # local import to avoid top-level cost

    for i, pose in enumerate(poses):
        stem = frame_stem(i)
        seed = args.seed + i * 17 + 1
        t0 = time.time()

        # Beauty (per backend).
        for b in backends:
            hdr_path = output_dir / rgb_hdr_subdir(b) / f"{stem}.exr"
            if i in skip_per_backend[b] and hdr_path.exists():
                hdr_per_backend[b].append(read_exr(hdr_path))
                continue
            hdr = _render_frame_beauty(b, pose, beauty_intr, args, rcfg, mi_scene, seed)
            write_exr(hdr_path, hdr)
            write_png_uint8(
                output_dir / rgb_subdir(b) / f"{stem}.png",
                tonemap(hdr, placeholder_exposure),
            )
            hdr_per_backend[b].append(hdr)

        # AOVs (shared, always Blender).
        aov = render_aov_blender(
            pose, beauty_intr,
            spp=rcfg.spp_aov, max_depth=rcfg.max_depth, seed=seed,
        )
        write_exr(output_dir / "depth" / f"{stem}.exr", aov.depth)
        write_exr(output_dir / "normal" / f"{stem}.exr", aov.normal)
        write_exr(output_dir / "albedo" / f"{stem}.exr", aov.albedo)
        if aov.glossy_color is not None:
            write_exr(output_dir / "specular_albedo" / f"{stem}.exr", aov.glossy_color)
        if aov.emission is not None:
            write_exr(output_dir / "emission" / f"{stem}.exr", aov.emission)
        if aov.material_index is not None:
            # Material indices are small integers — store as float EXR to match
            # everything else (FreeImage round-trips fine).
            write_exr(
                output_dir / "material_index" / f"{stem}.exr",
                aov.material_index.astype(np.float32),
            )
        if aov.roughness_per_pixel is not None:
            write_exr(output_dir / "roughness" / f"{stem}.exr", aov.roughness_per_pixel)
        if aov.metallic_per_pixel is not None:
            write_exr(output_dir / "metallic" / f"{stem}.exr", aov.metallic_per_pixel)

        # Debug LDR PNGs.
        write_png_uint8(
            output_dir / "albedo_ldr" / f"{stem}.png", albedo_to_ldr(aov.albedo)
        )
        write_png_uint8(
            output_dir / "depth_rgb" / f"{stem}.png", depth_to_rgb(aov.depth)
        )
        write_png_uint8(
            output_dir / "normal_rgb" / f"{stem}.png", normal_to_rgb(aov.normal)
        )

        # Per-pixel envmap (slow — only when enabled).
        if envmap_cfg is not None:
            envmap = render_envmaps_for_frame(
                pose=pose,
                intrinsics=beauty_intr,
                depth=aov.depth,
                normal=aov.normal,
                cfg=envmap_cfg,
                seed=seed + 7,
            )
            write_exr(output_dir / "envmap" / f"{stem}.exr", envmap)

        log.info("Frame %s / %s rendered in %.1fs",
                 stem, frame_stem(len(poses) - 1), time.time() - t0)
    return hdr_per_backend


# ---- Tonemap ----------------------------------------------------------------


def _tonemap_phase(
    hdr_per_backend: dict[str, list[NDArray[np.float32]]],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, ExposureParams]:
    out: dict[str, ExposureParams] = {}
    for backend, hdr_images in hdr_per_backend.items():
        if not hdr_images:
            continue
        params = compute_scene_exposure(
            hdr_images,
            key=args.tonemap_key,
            percentile=args.tonemap_percentile,
            gamma=args.tonemap_gamma,
        )
        log.info(
            "Backend %s exposure: key=%.3f exposure=%.4f gamma=%.2f",
            backend, params.key, params.exposure, params.gamma,
        )
        for i, hdr in enumerate(hdr_images):
            ldr = tonemap(hdr, params)
            write_png_uint8(
                output_dir / rgb_subdir(backend) / f"{frame_stem(i)}.png", ldr
            )
        out[backend] = params
    return out


# ---- JSON output ------------------------------------------------------------


def _write_dataset_json(
    output_dir: Path,
    poses: list[CameraPose],
    rcfg: RenderConfig,
    info: SceneInfo,
    exposures: dict[str, ExposureParams],
    args: argparse.Namespace,
    backends: list[str],
    extra_metadata: dict[str, Any],
    *,
    blender_version: str | None,
    mitsuba_variant: str | None,
    with_envmap: bool,
) -> None:
    beauty_intr = rcfg.beauty_intrinsics
    frames = [
        build_frame_record(
            i, pose_to_c2w_opencv(pose),
            backends=backends, with_envmap=with_envmap,
        )
        for i, pose in enumerate(poses)
    ]
    write_transforms_json(
        output_dir / TRANSFORMS_FILENAME, beauty_intr, info.up_axis, frames
    )

    metadata: dict[str, Any] = {
        "scene_xml": str(args.scene.resolve()) if args.scene else None,
        "scene_name": args.scene.parent.name if args.scene else None,
        "backends": backends,
        "blender_version": blender_version,
        "mitsuba_variant": mitsuba_variant,
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "seed": args.seed,
        "num_cameras_requested": args.num_cameras,
        "num_cameras_kept": len(poses),
        "oversample": args.oversample,
        "debug": args.debug,
        "fov_x_deg": args.fov,
        "resolution_full": [DEFAULT_FULL_W, DEFAULT_FULL_H],
        "resolution_debug": [DEFAULT_DEBUG_W, DEFAULT_DEBUG_H],
        "resolution_filter": [rcfg.filter_width, rcfg.filter_height],
        "resolution_used": [beauty_intr.width, beauty_intr.height],
        "spp": {
            "beauty_mitsuba": _spp_beauty("mitsuba", args) if "mitsuba" in backends else None,
            "beauty_blender": _spp_beauty("blender", args) if "blender" in backends else None,
            "aov": rcfg.spp_aov,
            "filter": rcfg.spp_filter,
            "envmap": args.envmap_spp if with_envmap else None,
        },
        "max_depth": rcfg.max_depth,
        "tonemap": {k: asdict(v) for k, v in exposures.items()},
        "depth_filter": {
            "max_infinity_ratio": args.filter_max_inf,
            "min_median_depth": args.filter_min_median,
            "near_threshold": args.filter_near_threshold,
            "max_near_ratio": args.filter_max_near,
        },
        "placement": {
            "placement_margin": args.placement_margin,
            "height_margin": args.height_margin,
            "height_range_requested": [args.height_min, args.height_max],
            "height_range_used": list(info.height_range),
            "bbox_min": list(info.bbox_min),
            "bbox_max": list(info.bbox_max),
            "placement_min": list(info.placement_min),
            "placement_max": list(info.placement_max),
            "up_axis": info.up_axis,
        },
        "sampler": {
            "name": args.sampler,
            "pitch_range_deg": [args.pitch_min_deg, args.pitch_max_deg],
            "yaw_offset_range_deg": [args.yaw_offset_min_deg, args.yaw_offset_max_deg]
                if args.sampler == "wall_walk" else None,
            "wall_walk_dist_min": args.wall_walk_dist_min
                if args.sampler == "wall_walk" else None,
            "wall_walk_dist_max": args.wall_walk_dist_max
                if args.sampler == "wall_walk" else None,
        },
        "aov_source": "blender",
        "aov_aliases": {
            "albedo": "FIPT DiffCol (k_d)",
            "specular_albedo": "FIPT GlossCol (k_s)",
            "material_index": "FIPT IndexMA",
            "emission": "FIPT Emit",
            "roughness": "per-pixel shader AOV (Principled BSDF Roughness)",
            "metallic": "per-pixel shader AOV (Principled BSDF Metallic)",
        },
        "depth_convention": (
            "Per-pixel value is the perspective z-depth (camera-space +Z, in "
            "meters) from the camera plane to the first surface hit. Multiply "
            "by sqrt(1 + ((u-cx)/fx)^2 + ((v-cy)/fy)^2) to convert to ray-t."
        ),
    }
    metadata.update(extra_metadata)
    write_metadata_json(output_dir / METADATA_FILENAME, metadata)


# ---- Main -------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    args = parse_args(argv)

    if args.install_addon_only:
        from src.scene_blender import ensure_mitsuba_addon
        module = ensure_mitsuba_addon()
        log.info("mitsuba-blender addon ready (module=%s)", module)
        return 0

    if args.scene is None or args.output is None:
        log.error("--scene and --output are required unless --install-addon-only.")
        return 2

    cfg = _load_scene_config(args.scene_config, args)
    _apply_config_overrides(args, cfg)
    args.output.mkdir(parents=True, exist_ok=True)

    # Transparent filter: pick the XML we'll use everywhere downstream.
    work_dir = args.output / "_work"
    transparent_report: FilterReport | None = None
    if args.remove_transparent:
        scene_for_load, transparent_report = filter_transparent_scene(
            args.scene, work_dir
        )
        log.info(
            "Transparent filter dropped %d BSDF(s), %d shape(s)",
            len(transparent_report.dropped_bsdf_ids),
            len(transparent_report.dropped_shape_ids),
        )
    else:
        scene_for_load = args.scene

    # PLY-only mode.
    if args.only_ply:
        ply_path = args.output / "scene.ply"
        spacing = 0.0 if args.no_tessellate else float(args.tessellate_spacing)
        result = export_colored_mesh_ply(
            scene_for_load, ply_path,
            tessellate_spacing=(spacing if spacing > 0.0 else None),
            simplify_dense_meshes=args.simplify_dense_meshes,
            simplify_vertex_threshold=args.simplify_vertex_threshold,
        )
        if result is None:
            log.error("PLY export failed: no geometry found in %s", scene_for_load)
            return 1
        log.info("Wrote %s (%d verts, %d faces).",
                 result.path, result.total_verts, result.total_faces)
        return 0

    rng = _set_seeds(args.seed)

    # ---- Backend init ----
    backends = _backends_active(args)
    blender_version = init_blender(
        ensure_addon=True,
        device=args.cycles_device,
        denoiser=args.denoiser,
        roughplastic_coat_weight=args.roughplastic_coat_weight,
    )
    load_scene_blender(scene_for_load)

    mi_scene: Any = None
    mitsuba_variant: str | None = None
    if "mitsuba" in backends:
        mitsuba_variant = init_mitsuba(prefer=args.mitsuba_variant)
        mi_scene = load_scene(scene_for_load)

    # ---- Scene info + output layout ----
    up_axis_hint: UpAxis | None = args.up_axis
    if up_axis_hint is None:
        up_axis_hint = parse_scene_up_axis_from_xml(scene_for_load)
    info = derive_scene_info_blender(
        height_range=(args.height_min, args.height_max),
        placement_margin=args.placement_margin,
        height_margin=args.height_margin,
        up_axis_override=up_axis_hint,
    )
    log.info("Scene info: %s", info)
    with_envmap = _envmap_enabled(args)
    make_layout(args.output, backends=backends, with_envmap=with_envmap)

    # ---- PLY export ----
    ply_relpath: str | None = None
    mesh_export_meta: dict[str, Any] | None = None
    if not args.no_ply:
        ply_path = args.output / "scene.ply"
        spacing = 0.0 if args.no_tessellate else float(args.tessellate_spacing)
        try:
            result = export_colored_mesh_ply(
                scene_for_load, ply_path,
                tessellate_spacing=(spacing if spacing > 0.0 else None),
                simplify_dense_meshes=args.simplify_dense_meshes,
                simplify_vertex_threshold=args.simplify_vertex_threshold,
            )
            if result is not None:
                ply_relpath = ply_path.name
                mesh_export_meta = {
                    "tessellate_spacing": spacing if spacing > 0.0 else None,
                    "simplify_dense_meshes": args.simplify_dense_meshes,
                    "simplify_vertex_threshold": args.simplify_vertex_threshold,
                    "total_verts": result.total_verts,
                    "total_faces": result.total_faces,
                    "per_shape": result.per_shape,
                }
        except Exception as e:
            log.warning("PLY export failed: %s", e)

    # ---- Camera sampling ----
    rcfg = _build_render_config(args)
    log.info(
        "RenderConfig: resolution=%s spp_aov=%d max_depth=%d debug=%s backends=%s",
        (rcfg.beauty_intrinsics.width, rcfg.beauty_intrinsics.height),
        rcfg.spp_aov, rcfg.max_depth, rcfg.debug, backends,
    )
    poses, sampling_meta = _sample_phase(
        info, rng, rcfg, args, scene_for_load
    )
    if not poses:
        log.error("No cameras passed filtering; aborting before render.")
        return 1

    _render_preview(poses, rcfg, args, args.output)

    # ---- Render ----
    envmap_cfg: EnvmapConfig | None = None
    if with_envmap:
        envmap_cfg = EnvmapConfig(
            patch_size=args.envmap_patch_size,
            env_height=int(args.envmap_resolution[0]),
            env_width=int(args.envmap_resolution[1]),
            spp=args.envmap_spp,
            max_depth=rcfg.max_depth,
        )
        log.info(
            "Envmap enabled: patch=%d, %dx%d per cell, spp=%d",
            envmap_cfg.patch_size, envmap_cfg.env_height, envmap_cfg.env_width,
            envmap_cfg.spp,
        )

    t0 = time.time()
    hdr_per_backend = _render_phase(
        backends, poses, rcfg, args, args.output, mi_scene, envmap_cfg
    )
    render_seconds = time.time() - t0
    log.info("Rendered %d frames in %.1fs.", len(poses), render_seconds)

    exposures = _tonemap_phase(hdr_per_backend, args.output, args)

    extra: dict[str, Any] = {
        "sampling": sampling_meta,
        "render_seconds": round(render_seconds, 2),
    }
    if ply_relpath is not None:
        extra["scene_ply"] = ply_relpath
    if mesh_export_meta is not None:
        extra["mesh_export"] = mesh_export_meta
    if transparent_report is not None:
        extra["transparent_filter"] = transparent_report.to_dict()
    if envmap_cfg is not None:
        env_row, env_col = envmap_cfg.grid_shape(
            rcfg.beauty_intrinsics.height, rcfg.beauty_intrinsics.width
        )
        extra["envmap"] = {
            "patch_size": envmap_cfg.patch_size,
            "env_height": envmap_cfg.env_height,
            "env_width": envmap_cfg.env_width,
            "env_row": env_row,
            "env_col": env_col,
            "mosaic_shape": list(
                envmap_cfg.mosaic_shape(
                    rcfg.beauty_intrinsics.height, rcfg.beauty_intrinsics.width,
                )
            ),
            "spp": envmap_cfg.spp,
        }

    _write_dataset_json(
        args.output, poses, rcfg, info, exposures, args,
        backends=backends, extra_metadata=extra,
        blender_version=blender_version, mitsuba_variant=mitsuba_variant,
        with_envmap=with_envmap,
    )

    # Clean up the transient work dir.
    if work_dir.exists():
        try:
            shutil.rmtree(work_dir)
        except OSError as e:
            log.warning("Could not remove work dir %s: %s", work_dir, e)

    log.info("Done. Output: %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
