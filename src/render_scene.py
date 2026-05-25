"""CLI entry point: render a single Mitsuba scene into a multi-view dataset."""

from __future__ import annotations

import argparse
import logging
import platform
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mitsuba as mi
import numpy as np
import yaml
from numpy.typing import NDArray

from src.camera_sampling import (
    CameraPose,
    DepthFilterConfig,
    sample_cameras,
)
from src.io_utils import (
    METADATA_FILENAME,
    PREVIEW_FILENAME,
    SUBDIRS,
    TRANSFORMS_FILENAME,
    build_frame_record,
    compute_intrinsics,
    existing_rendered_frames,
    frame_stem,
    make_output_layout,
    write_exr,
    write_metadata_json,
    write_png_uint8,
    write_transforms_json,
)
from src.render import (
    RenderConfig,
    derive_material_lut,
    make_contact_sheet,
    material_caveat_message,
    materials_from_shape_index,
    pose_to_c2w,
    render_aov,
    render_beauty,
    render_depth_for_filter,
)
from src.scene_utils import (
    SceneInfo,
    UpAxis,
    derive_scene_info,
    init_mitsuba,
    inside_room_test,
    load_scene,
)
from src.tonemap import (
    ExposureParams,
    compute_scene_exposure,
    tonemap,
)
from src.viz_utils import (
    albedo_to_ldr,
    depth_to_rgb,
    normal_to_rgb,
)
from src.mesh_utils import export_colored_mesh_ply

log = logging.getLogger(__name__)


# ---- Argument parsing --------------------------------------------------------

DEFAULT_FULL_W, DEFAULT_FULL_H = 640, 480
DEFAULT_DEBUG_W, DEFAULT_DEBUG_H = 160, 120
DEFAULT_FILTER_W, DEFAULT_FILTER_H = 160, 90
DEFAULT_SPP_BEAUTY_FULL = 256
DEFAULT_SPP_BEAUTY_DEBUG = 64
DEFAULT_SPP_AOV = 16
DEFAULT_SPP_FILTER = 4
DEFAULT_FOV = 60.0
DEFAULT_NUM_CAMERAS = 200
DEFAULT_MAX_DEPTH = 16  # Conservative; scene XML can override only if larger.
DEFAULT_PLACEMENT_MARGIN = 0.5
DEFAULT_HEIGHT_RANGE = (0.8, 1.8)
DEFAULT_HEIGHT_MARGIN = 0.3
DEFAULT_OVERSAMPLE = 4
DEFAULT_PITCH_RANGE_DEG = (-25.0, 25.0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render a Mitsuba indoor scene into a multi-view PBR/IR test set."
    )
    p.add_argument("--scene", type=Path, required=True, help="Path to scene XML.")
    p.add_argument(
        "--output", type=Path, required=True, help="Output directory for this scene."
    )
    p.add_argument(
        "--num-cameras", type=int, default=DEFAULT_NUM_CAMERAS,
        help=f"Cameras to keep (default {DEFAULT_NUM_CAMERAS}).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--debug", action="store_true",
                   help="Use low resolution + low spp for fast iteration.")
    p.add_argument("--fov", type=float, default=DEFAULT_FOV,
                   help="Horizontal FOV in degrees (default %(default)s).")
    p.add_argument("--mitsuba-variant", type=str, default=None,
                   help="Override variant choice (default: cuda_ad_rgb -> llvm_ad_rgb).")
    p.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    p.add_argument("--oversample", type=int, default=DEFAULT_OVERSAMPLE)
    p.add_argument("--placement-margin", type=float, default=DEFAULT_PLACEMENT_MARGIN)
    p.add_argument("--height-min", type=float, default=DEFAULT_HEIGHT_RANGE[0])
    p.add_argument("--height-max", type=float, default=DEFAULT_HEIGHT_RANGE[1])
    p.add_argument("--height-margin", type=float, default=DEFAULT_HEIGHT_MARGIN)
    p.add_argument("--pitch-min-deg", type=float, default=DEFAULT_PITCH_RANGE_DEG[0])
    p.add_argument("--pitch-max-deg", type=float, default=DEFAULT_PITCH_RANGE_DEG[1])
    p.add_argument("--up-axis", choices=["x", "y", "z"], default=None,
                   help="Override the detected up axis.")
    # Depth filter thresholds
    p.add_argument("--filter-max-inf", type=float, default=0.01)
    p.add_argument("--filter-min-median", type=float, default=0.8)
    p.add_argument("--filter-near-threshold", type=float, default=0.3)
    p.add_argument("--filter-max-near", type=float, default=0.05)
    # Tonemap params
    p.add_argument("--tonemap-key", type=float, default=0.18)
    p.add_argument("--tonemap-percentile", type=float, default=90.0)
    p.add_argument("--tonemap-gamma", type=float, default=2.2)
    p.add_argument("--scene-config", type=Path, default=None,
                   help="Optional YAML overrides for placement/height/up_axis/fov.")
    p.add_argument("--no-ply", action="store_true",
                   help="Skip exporting the colored mesh as <output>/scene.ply.")
    p.add_argument("--tessellate-spacing", type=float, default=0.10,
                   help="Adaptively subdivide textured faces so world-space "
                        "edges don't exceed this spacing (meters), then "
                        "bilinear-bake the texture into per-vertex colors. "
                        "Set to 0 (or use --no-tessellate) to disable. "
                        "Default 0.10.")
    p.add_argument("--no-tessellate", action="store_true",
                   help="Disable tessellate-then-bake (equivalent to "
                        "--tessellate-spacing 0). Falls back to one mean color "
                        "per textured shape.")
    p.add_argument("--simplify-dense-meshes", action="store_true",
                   help="Decimate any single mesh with > "
                        "--simplify-vertex-threshold vertices via quadric "
                        "decimation before coloring. Useful when one shape "
                        "(e.g. a heavily subdivided carpet) dominates the "
                        "PLY size.")
    p.add_argument("--simplify-vertex-threshold", type=int, default=100_000,
                   help="Vertex threshold above which decimation kicks in. "
                        "Only takes effect with --simplify-dense-meshes. "
                        "Default 100000.")
    return p.parse_args(argv)


def _load_scene_config(path: Path | None, args: argparse.Namespace) -> dict[str, Any]:
    """Apply optional YAML overrides next to the scene XML."""
    cfg: dict[str, Any] = {}
    candidates: list[Path] = []
    if path is not None:
        candidates.append(path)
    # Auto-detect: scene_config.yaml next to the scene XML.
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


# ---- Orchestration helpers ---------------------------------------------------


def _set_seeds(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    return np.random.default_rng(seed)


def _build_render_config(args: argparse.Namespace) -> RenderConfig:
    intr_full = compute_intrinsics(args.fov, DEFAULT_FULL_W, DEFAULT_FULL_H)
    intr_debug = compute_intrinsics(args.fov, DEFAULT_DEBUG_W, DEFAULT_DEBUG_H)
    return RenderConfig(
        intrinsics_full=intr_full,
        intrinsics_debug=intr_debug,
        spp_beauty_full=DEFAULT_SPP_BEAUTY_FULL,
        spp_beauty_debug=DEFAULT_SPP_BEAUTY_DEBUG,
        spp_aov=DEFAULT_SPP_AOV,
        spp_filter=DEFAULT_SPP_FILTER,
        filter_width=DEFAULT_FILTER_W,
        filter_height=DEFAULT_FILTER_H,
        max_depth=args.max_depth,
        debug=args.debug,
    )


def _sample_phase(
    scene: mi.Scene,
    info: SceneInfo,
    rng: np.random.Generator,
    rcfg: RenderConfig,
    args: argparse.Namespace,
) -> tuple[list[CameraPose], dict[str, Any]]:
    filter_intr = compute_intrinsics(args.fov, rcfg.filter_width, rcfg.filter_height)

    def inside(positions: NDArray[np.float64]) -> NDArray[np.bool_]:
        return inside_room_test(scene, positions, up_axis=info.up_axis)

    def render_depth_cb(pose: CameraPose) -> NDArray[np.float64]:
        depth = render_depth_for_filter(
            scene, pose, filter_intr,
            spp=rcfg.spp_filter, max_depth=rcfg.max_depth,
        )
        return depth.astype(np.float64)

    filter_cfg = DepthFilterConfig(
        max_infinity_ratio=args.filter_max_inf,
        min_median_depth=args.filter_min_median,
        near_threshold=args.filter_near_threshold,
        max_near_ratio=args.filter_max_near,
    )
    t0 = time.time()
    poses, stats = sample_cameras(
        rng,
        info,
        num_cameras=args.num_cameras,
        inside_room=inside,
        render_depth=render_depth_cb,
        depth_filter=filter_cfg,
        pitch_range_deg=(args.pitch_min_deg, args.pitch_max_deg),
        oversample=args.oversample,
    )
    elapsed = time.time() - t0
    log.info(
        "Sampled %d/%d cameras in %.1fs. Rejections: %s",
        len(poses), args.num_cameras, elapsed, dict(stats.by_reason),
    )
    return poses, {
        "n_attempts": stats.n_attempts,
        "n_accepted": stats.n_accepted,
        "by_reason": dict(stats.by_reason),
        "sampling_seconds": round(elapsed, 2),
    }


def _render_preview(
    scene: mi.Scene,
    poses: list[CameraPose],
    rcfg: RenderConfig,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    """Render up to 16 debug-quality thumbnails with a placeholder exposure."""
    if not poses:
        return
    preview_intr = rcfg.intrinsics_debug
    placeholder = ExposureParams(key=0.18, percentile=90.0, exposure=0.5, gamma=2.2)
    thumbs: list[NDArray[np.uint8]] = []
    for i, pose in enumerate(poses[:16]):
        hdr = render_beauty(
            scene, pose, preview_intr,
            spp=max(8, rcfg.spp_beauty_debug // 2),
            max_depth=rcfg.max_depth,
            seed=args.seed + i,
        )
        ldr = tonemap(hdr, placeholder)
        thumbs.append(ldr)
    sheet = make_contact_sheet(thumbs, grid=(4, 4))
    write_png_uint8(output_dir / PREVIEW_FILENAME, sheet)
    log.info("Preview written: %s", output_dir / PREVIEW_FILENAME)


def _render_phase(
    scene: mi.Scene,
    poses: list[CameraPose],
    rcfg: RenderConfig,
    args: argparse.Namespace,
    output_dir: Path,
    rough_lut: NDArray[np.float32],
    metal_lut: NDArray[np.float32],
) -> list[NDArray[np.float32]]:
    """Render each kept camera's beauty + AOV passes. Returns HDR images."""
    beauty_intr = rcfg.beauty_intrinsics
    aov_intr = beauty_intr  # AOVs use the same resolution as beauty.
    skip = existing_rendered_frames(output_dir)
    if skip:
        log.info("Resuming: %d frames already have HDR EXRs and will be skipped.",
                 len(skip))

    hdr_images: list[NDArray[np.float32]] = []
    for i, pose in enumerate(poses):
        stem = frame_stem(i)
        if i in skip:
            hdr_path = output_dir / "rgb_hdr" / f"{stem}.exr"
            if hdr_path.exists():
                # Read back HDR so the tonemap pass can use it.
                from src.io_utils import read_exr
                hdr_images.append(read_exr(hdr_path))
                continue

        seed = args.seed + i * 17 + 1
        t0 = time.time()
        hdr = render_beauty(
            scene, pose, beauty_intr,
            spp=rcfg.spp_beauty, max_depth=rcfg.max_depth, seed=seed,
        )
        write_exr(output_dir / "rgb_hdr" / f"{stem}.exr", hdr)

        aov = render_aov(
            scene, pose, aov_intr,
            spp=rcfg.spp_aov, max_depth=rcfg.max_depth, seed=seed,
        )
        write_exr(output_dir / "depth" / f"{stem}.exr", aov.depth)
        write_exr(output_dir / "normal" / f"{stem}.exr", aov.normal)
        write_exr(output_dir / "albedo" / f"{stem}.exr", aov.albedo)

        mats = materials_from_shape_index(aov.shape_index, rough_lut, metal_lut)
        write_exr(output_dir / "roughness" / f"{stem}.exr", mats.roughness)
        write_exr(output_dir / "metallic" / f"{stem}.exr", mats.metallic)

        # Debug LDR PNGs (not referenced by transforms.json).
        write_png_uint8(
            output_dir / "albedo_ldr" / f"{stem}.png", albedo_to_ldr(aov.albedo)
        )
        write_png_uint8(
            output_dir / "depth_rgb" / f"{stem}.png", depth_to_rgb(aov.depth)
        )
        write_png_uint8(
            output_dir / "normal_rgb" / f"{stem}.png", normal_to_rgb(aov.normal)
        )

        hdr_images.append(hdr)
        log.info("Frame %s/%s rendered in %.1fs", stem, frame_stem(len(poses) - 1),
                 time.time() - t0)
    return hdr_images


def _tonemap_phase(
    hdr_images: list[NDArray[np.float32]],
    output_dir: Path,
    args: argparse.Namespace,
) -> ExposureParams:
    params = compute_scene_exposure(
        hdr_images,
        key=args.tonemap_key,
        percentile=args.tonemap_percentile,
        gamma=args.tonemap_gamma,
    )
    log.info(
        "Scene exposure: key=%.3f p%d=%.4f -> exposure=%.4f, gamma=%.2f",
        params.key, int(params.percentile), params.key / params.exposure,
        params.exposure, params.gamma,
    )
    for i, hdr in enumerate(hdr_images):
        ldr = tonemap(hdr, params)
        write_png_uint8(output_dir / "rgb" / f"{frame_stem(i)}.png", ldr)
    return params


def _write_dataset_json(
    output_dir: Path,
    poses: list[CameraPose],
    rcfg: RenderConfig,
    info: SceneInfo,
    exposure: ExposureParams,
    args: argparse.Namespace,
    extra_metadata: dict[str, Any],
    variant: str,
) -> None:
    beauty_intr = rcfg.beauty_intrinsics
    frames = [
        build_frame_record(i, pose_to_c2w(pose)) for i, pose in enumerate(poses)
    ]
    write_transforms_json(
        output_dir / TRANSFORMS_FILENAME, beauty_intr, info.up_axis, frames
    )

    metadata: dict[str, Any] = {
        "scene_xml": str(args.scene.resolve()),
        "scene_name": args.scene.parent.name,
        "mitsuba_version": mi.__version__,
        "mitsuba_variant": variant,
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
            "beauty": rcfg.spp_beauty,
            "aov": rcfg.spp_aov,
            "filter": rcfg.spp_filter,
        },
        "max_depth": rcfg.max_depth,
        "tonemap": asdict(exposure),
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
        "aov_caveats": material_caveat_message(),
        "depth_convention": (
            "Per-pixel value is the ray-distance t (in meters) from the camera "
            "origin to the first surface hit. NOT perspective Z-depth."
        ),
        "debug_outputs": {
            "albedo_ldr": {
                "dir": "albedo_ldr",
                "encoding": "sRGB gamma 2.2 of linear albedo, clip [0, 1] -> uint8",
            },
            "depth_rgb": {
                "dir": "depth_rgb",
                "colormap": "turbo",
                "normalization": "per-frame percentile (2, 98); void pixels white",
            },
            "normal_rgb": {
                "dir": "normal_rgb",
                "encoding": "(world_normal * 0.5 + 0.5) * 255 -> uint8 RGB",
            },
        },
    }
    metadata.update(extra_metadata)
    write_metadata_json(output_dir / METADATA_FILENAME, metadata)


# ---- main --------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    args = parse_args(argv)
    cfg = _load_scene_config(args.scene_config, args)
    _apply_config_overrides(args, cfg)

    rng = _set_seeds(args.seed)
    variant = init_mitsuba(prefer=args.mitsuba_variant)
    scene = load_scene(args.scene)

    info = derive_scene_info(
        scene,
        height_range=(args.height_min, args.height_max),
        placement_margin=args.placement_margin,
        height_margin=args.height_margin,
        up_axis_override=args.up_axis,  # type: ignore[arg-type]
    )
    log.info("Scene info: %s", info)
    make_output_layout(args.output)

    ply_relpath: str | None = None
    mesh_export_meta: dict[str, Any] | None = None
    if not args.no_ply:
        ply_path = args.output / "scene.ply"
        spacing = 0.0 if args.no_tessellate else float(args.tessellate_spacing)
        try:
            result = export_colored_mesh_ply(
                args.scene, ply_path,
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

    rcfg = _build_render_config(args)
    log.info(
        "RenderConfig: resolution=%s spp_beauty=%d spp_aov=%d max_depth=%d debug=%s",
        (rcfg.beauty_intrinsics.width, rcfg.beauty_intrinsics.height),
        rcfg.spp_beauty, rcfg.spp_aov, rcfg.max_depth, rcfg.debug,
    )

    poses, sampling_meta = _sample_phase(scene, info, rng, rcfg, args)
    if not poses:
        log.error("No cameras passed filtering; aborting before render.")
        return 1

    _render_preview(scene, poses, rcfg, args, args.output)

    rough_lut, metal_lut = derive_material_lut(scene)
    t0 = time.time()
    hdr_images = _render_phase(
        scene, poses, rcfg, args, args.output, rough_lut, metal_lut
    )
    render_seconds = time.time() - t0
    log.info("Rendered %d frames in %.1fs.", len(hdr_images), render_seconds)

    exposure = _tonemap_phase(hdr_images, args.output, args)

    extra: dict[str, Any] = {
        "sampling": sampling_meta,
        "render_seconds": round(render_seconds, 2),
    }
    if ply_relpath is not None:
        extra["scene_ply"] = ply_relpath
    if mesh_export_meta is not None:
        extra["mesh_export"] = mesh_export_meta
    _write_dataset_json(
        args.output, poses, rcfg, info, exposure, args, extra, variant
    )
    log.info("Done. Output: %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
