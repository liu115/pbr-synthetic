"""CLI entry point: render a Mitsuba-XML scene with Blender's Cycles backend.

Mirrors ``src/render_scene.py`` so users can switch backends by changing one
module name on the command line. The output schema (directory layout,
``transforms.json``, ``metadata.json``, ``scene.ply``) is identical; only
``metadata.renderer`` distinguishes the two pipelines.
"""

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
from src.mesh_utils import export_colored_mesh_ply
from src.pose_utils import pose_to_c2w
from src.render import (
    RenderConfig,
    make_contact_sheet,
    material_caveat_message,
    materials_from_shape_index,
)
from src.render_blender import (
    derive_material_lut_blender,
    render_aov_blender,
    render_beauty_blender,
    render_depth_for_filter_blender,
)
from src.scene_blender import (
    derive_scene_info_blender,
    ensure_mitsuba_addon,
    init_blender,
    inside_room_test_blender,
    load_scene_blender,
)
from src.scene_utils import SceneInfo
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

log = logging.getLogger(__name__)


# ---- Defaults (mirror render_scene.py) --------------------------------------

DEFAULT_FULL_W, DEFAULT_FULL_H = 640, 480
DEFAULT_DEBUG_W, DEFAULT_DEBUG_H = 160, 120
DEFAULT_FILTER_W, DEFAULT_FILTER_H = 160, 90
# Cycles + OptiX denoising reaches clean images at much lower spp than raw
# Mitsuba; default lower so a default-flag run is comparable in wall time.
DEFAULT_SPP_BEAUTY_FULL = 256
DEFAULT_SPP_BEAUTY_DEBUG = 32
DEFAULT_SPP_AOV = 16
DEFAULT_SPP_FILTER = 4
DEFAULT_FOV = 60.0
DEFAULT_NUM_CAMERAS = 200
DEFAULT_MAX_DEPTH = 8
DEFAULT_PLACEMENT_MARGIN = 0.5
DEFAULT_HEIGHT_RANGE = (0.8, 1.8)
DEFAULT_HEIGHT_MARGIN = 0.3
DEFAULT_OVERSAMPLE = 4
DEFAULT_PITCH_RANGE_DEG = (-25.0, 25.0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render a Mitsuba scene XML into a multi-view dataset using Blender's Cycles."
    )
    p.add_argument("--scene", type=Path, required=False, help="Path to scene XML.")
    p.add_argument("--output", type=Path, required=False,
                   help="Output directory for this scene.")
    p.add_argument("--num-cameras", type=int, default=DEFAULT_NUM_CAMERAS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--debug", action="store_true",
                   help="Use low resolution + low spp for fast iteration.")
    p.add_argument("--fov", type=float, default=DEFAULT_FOV)
    p.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    p.add_argument("--oversample", type=int, default=DEFAULT_OVERSAMPLE)
    p.add_argument("--placement-margin", type=float, default=DEFAULT_PLACEMENT_MARGIN)
    p.add_argument("--height-min", type=float, default=DEFAULT_HEIGHT_RANGE[0])
    p.add_argument("--height-max", type=float, default=DEFAULT_HEIGHT_RANGE[1])
    p.add_argument("--height-margin", type=float, default=DEFAULT_HEIGHT_MARGIN)
    p.add_argument("--pitch-min-deg", type=float, default=DEFAULT_PITCH_RANGE_DEG[0])
    p.add_argument("--pitch-max-deg", type=float, default=DEFAULT_PITCH_RANGE_DEG[1])
    p.add_argument("--up-axis", choices=["x", "y", "z"], default=None)
    # Depth filter thresholds (same as Mitsuba)
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
    # Blender-specific
    p.add_argument("--cycles-device", choices=["OPTIX", "CUDA", "CPU"], default="OPTIX",
                   help="Cycles compute device (default OPTIX).")
    p.add_argument("--denoiser", choices=["OPTIX", "OPENIMAGEDENOISE", "NONE"],
                   default="OPTIX",
                   help="Cycles denoiser for beauty renders (default OPTIX).")
    p.add_argument("--install-addon-only", action="store_true",
                   help="Install and enable the mitsuba-blender addon, then exit.")
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


# ---- Orchestration ----------------------------------------------------------


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
    info: SceneInfo,
    rng: np.random.Generator,
    rcfg: RenderConfig,
    args: argparse.Namespace,
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
    poses: list[CameraPose],
    rcfg: RenderConfig,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    if not poses:
        return
    preview_intr = rcfg.intrinsics_debug
    placeholder = ExposureParams(key=0.18, percentile=90.0, exposure=0.5, gamma=2.2)
    thumbs: list[NDArray[np.uint8]] = []
    for i, pose in enumerate(poses[:16]):
        hdr = render_beauty_blender(
            pose, preview_intr,
            spp=max(8, rcfg.spp_beauty_debug // 2),
            max_depth=rcfg.max_depth,
            seed=args.seed + i,
        )
        thumbs.append(tonemap(hdr, placeholder))
    sheet = make_contact_sheet(thumbs, grid=(4, 4))
    write_png_uint8(output_dir / PREVIEW_FILENAME, sheet)
    log.info("Preview written: %s", output_dir / PREVIEW_FILENAME)


def _render_phase(
    poses: list[CameraPose],
    rcfg: RenderConfig,
    args: argparse.Namespace,
    output_dir: Path,
    rough_lut: NDArray[np.float32],
    metal_lut: NDArray[np.float32],
) -> list[NDArray[np.float32]]:
    beauty_intr = rcfg.beauty_intrinsics
    aov_intr = beauty_intr
    skip = existing_rendered_frames(output_dir)
    if skip:
        log.info("Resuming: %d frames already have HDR EXRs and will be skipped.",
                 len(skip))

    placeholder_exposure = ExposureParams(
        key=args.tonemap_key, percentile=args.tonemap_percentile,
        exposure=1.0, gamma=args.tonemap_gamma,
    )

    hdr_images: list[NDArray[np.float32]] = []
    for i, pose in enumerate(poses):
        stem = frame_stem(i)
        if i in skip:
            hdr_path = output_dir / "rgb_hdr" / f"{stem}.exr"
            if hdr_path.exists():
                from src.io_utils import read_exr
                hdr_images.append(read_exr(hdr_path))
                continue

        seed = args.seed + i * 17 + 1
        t0 = time.time()
        hdr = render_beauty_blender(
            pose, beauty_intr,
            spp=rcfg.spp_beauty, max_depth=rcfg.max_depth, seed=seed,
        )
        write_exr(output_dir / "rgb_hdr" / f"{stem}.exr", hdr)
        write_png_uint8(
            output_dir / "rgb" / f"{stem}.png", tonemap(hdr, placeholder_exposure),
        )

        aov = render_aov_blender(
            pose, aov_intr,
            spp=rcfg.spp_aov, max_depth=rcfg.max_depth, seed=seed,
        )
        write_exr(output_dir / "depth" / f"{stem}.exr", aov.depth)
        write_exr(output_dir / "normal" / f"{stem}.exr", aov.normal)
        write_exr(output_dir / "albedo" / f"{stem}.exr", aov.albedo)

        mats = materials_from_shape_index(aov.shape_index, rough_lut, metal_lut)
        write_exr(output_dir / "roughness" / f"{stem}.exr", mats.roughness)
        write_exr(output_dir / "metallic" / f"{stem}.exr", mats.metallic)

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
    blender_version: str,
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
        "renderer": "blender",
        "blender_version": blender_version,
        "cycles": {
            "device": args.cycles_device,
            "denoiser": args.denoiser,
            "samples": rcfg.spp_beauty,
            "max_bounces": rcfg.max_depth,
        },
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
            "origin to the first surface hit. NOT perspective Z-depth. "
            "(Cycles Z pass converted via sqrt(1 + u^2 + v^2) where (u, v) are "
            "normalized image-plane coordinates.)"
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


# ---- main -------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    args = parse_args(argv)

    if args.install_addon_only:
        module = ensure_mitsuba_addon()
        log.info("mitsuba-blender addon ready (module=%s)", module)
        return 0

    if args.scene is None or args.output is None:
        log.error("--scene and --output are required unless --install-addon-only.")
        return 2

    cfg = _load_scene_config(args.scene_config, args)
    _apply_config_overrides(args, cfg)

    args.output.mkdir(parents=True, exist_ok=True)

    if args.only_ply:
        ply_path = args.output / "scene.ply"
        spacing = 0.0 if args.no_tessellate else float(args.tessellate_spacing)
        result = export_colored_mesh_ply(
            args.scene, ply_path,
            tessellate_spacing=(spacing if spacing > 0.0 else None),
            simplify_dense_meshes=args.simplify_dense_meshes,
            simplify_vertex_threshold=args.simplify_vertex_threshold,
        )
        if result is None:
            log.error("PLY export failed: no geometry found in %s", args.scene)
            return 1
        log.info(
            "Wrote %s (%d verts, %d faces). Done.",
            result.path, result.total_verts, result.total_faces,
        )
        return 0

    rng = _set_seeds(args.seed)
    blender_version = init_blender(
        ensure_addon=True, device=args.cycles_device, denoiser=args.denoiser,
    )
    pass_index_to_object = load_scene_blender(args.scene)

    info = derive_scene_info_blender(
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

    poses, sampling_meta = _sample_phase(info, rng, rcfg, args)
    if not poses:
        log.error("No cameras passed filtering; aborting before render.")
        return 1

    _render_preview(poses, rcfg, args, args.output)

    rough_lut, metal_lut = derive_material_lut_blender(pass_index_to_object)
    t0 = time.time()
    hdr_images = _render_phase(
        poses, rcfg, args, args.output, rough_lut, metal_lut
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
        args.output, poses, rcfg, info, exposure, args, extra, blender_version,
    )
    log.info("Done. Output: %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
