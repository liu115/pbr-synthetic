"""Blender↔Mitsuba HDR parity harness.

Render ONE scene from ONE camera with both backends and print the per-channel mean
radiance ratio ``blender / mitsuba``. Run after each parity fix (mirror, world, coat,
max_depth) to watch the ratio drift toward ``(1, 1, 1)``; the acceptance band is
``[0.97, 1.05]`` per channel.

It reuses the production render entry points (``render_beauty_blender``,
``render_beauty``) and the same ``pose_frame='blender'`` convention the pipeline uses,
so the two backends see the *same* camera.

Variant note: ``load_scene_blender`` forces Mitsuba's ``scalar_rgb`` variant for the
addon. ``scene_utils.init_mitsuba`` warns that switching variants mid-process can yield
all-zero renders, so we render Blender first and keep ``scalar_rgb`` for Mitsuba unless
``--mitsuba-variant`` overrides it.

Usage::

    python scripts/hdr_ratio_parity.py --scene scene_v3.xml \
        [--pose X Y Z YAW_DEG PITCH_DEG] [--width 160 --height 120 --fov 60] \
        [--spp-mitsuba 256 --spp-blender 256 --max-depth -1 --device CPU] \
        [--json out.json] [--assert]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

IN_BAND = (0.97, 1.05)


def _look_at_pose(eye: np.ndarray, target: np.ndarray, up_axis: str):
    """Build a CameraPose (position + yaw/pitch) looking from ``eye`` to ``target``.

    Inverts CameraPose.forward(): f[h0]=cos(pitch)sin(yaw), f[h1]=-cos(pitch)cos(yaw),
    f[up]=sin(pitch).
    """
    from src.camera_sampling import CameraPose

    axis_index = {"x": 0, "y": 1, "z": 2}
    up_idx = axis_index[up_axis]
    h0, h1 = (i for i in range(3) if i != up_idx)
    f = np.asarray(target, float) - np.asarray(eye, float)
    n = float(np.linalg.norm(f))
    f = f / n if n > 0 else f
    pitch = math.asin(float(np.clip(f[up_idx], -1.0, 1.0)))
    cp = math.cos(pitch)
    yaw = math.atan2(f[h0] / cp, -f[h1] / cp) if abs(cp) > 1e-9 else 0.0
    return CameraPose(
        position=(float(eye[0]), float(eye[1]), float(eye[2])),
        yaw=float(yaw), pitch=float(pitch), up_axis=up_axis,  # type: ignore[arg-type]
    )


def _default_pose():
    """A shared camera looking across the loaded scene's bounding box."""
    from src.scene_blender import derive_scene_info_blender

    info = derive_scene_info_blender(
        height_range=(0.3, 1.8), placement_margin=0.2, height_margin=0.1
    )
    lo = np.asarray(info.bbox_min, float)
    hi = np.asarray(info.bbox_max, float)
    center = 0.5 * (lo + hi)
    extent = hi - lo
    up_idx = {"x": 0, "y": 1, "z": 2}[info.up_axis]
    h0, h1 = (i for i in range(3) if i != up_idx)
    eye = center.copy()
    eye[h0] = lo[h0] + 0.2 * extent[h0]
    eye[h1] = lo[h1] + 0.2 * extent[h1]
    eye[up_idx] = center[up_idx]
    return _look_at_pose(eye, center, info.up_axis)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--pose", type=float, nargs=5, default=None,
                    metavar=("X", "Y", "Z", "YAW_DEG", "PITCH_DEG"),
                    help="Explicit pose; default looks across the scene bbox.")
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--height", type=int, default=120)
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--spp-mitsuba", type=int, default=256)
    ap.add_argument("--spp-blender", type=int, default=256)
    ap.add_argument("--max-depth", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", choices=["OPTIX", "CUDA", "CPU"], default="CPU")
    ap.add_argument("--mitsuba-variant", type=str, default=None)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--assert", dest="do_assert", action="store_true",
                    help="Exit non-zero if any channel ratio is outside [0.97, 1.05] "
                         "or structural mismatch >= 2%%.")
    args = ap.parse_args()
    if not args.scene.exists():
        ap.error(f"scene not found: {args.scene}")

    from src.io_utils import compute_intrinsics
    from src.scene_blender import init_blender, load_scene_blender
    from src.render_blender import render_beauty_blender
    from src.scene_utils import init_mitsuba, load_scene
    from src.render import render_beauty
    from src.camera_sampling import CameraPose

    intr = compute_intrinsics(args.fov, args.width, args.height)

    # ---- Blender first (its import forces Mitsuba's scalar_rgb variant) ----
    init_blender(ensure_addon=True, device=args.device, denoiser="NONE")
    load_scene_blender(args.scene)

    if args.pose is not None:
        from src.scene_blender import derive_scene_info_blender
        up_axis = derive_scene_info_blender().up_axis
        x, y, z, yaw_d, pitch_d = args.pose
        pose = CameraPose(position=(x, y, z), yaw=math.radians(yaw_d),
                          pitch=math.radians(pitch_d), up_axis=up_axis)  # type: ignore[arg-type]
    else:
        pose = _default_pose()

    hdr_b = render_beauty_blender(pose, intr, spp=args.spp_blender,
                                  max_depth=args.max_depth, seed=args.seed)

    # ---- Mitsuba (keep scalar_rgb unless overridden) ----
    init_mitsuba(prefer=args.mitsuba_variant)
    scene = load_scene(args.scene)
    hdr_m = render_beauty(scene, pose, intr, spp=args.spp_mitsuba,
                          max_depth=args.max_depth, seed=args.seed,
                          pose_frame="blender")

    # ---- Metrics ----
    b = hdr_b.astype(np.float64)
    m = hdr_m.astype(np.float64)
    finite = np.isfinite(b).all(-1) & np.isfinite(m).all(-1)
    pos = finite & (b > 0).all(-1) & (m > 0).all(-1)
    # structural mismatch: exactly one backend ~black at a pixel
    bdark = (b.max(-1) < 1e-6)
    mdark = (m.max(-1) < 1e-6)
    struct_mismatch = float(np.mean(finite & (bdark ^ mdark)))

    mean_b = b[pos].mean(0)
    mean_m = m[pos].mean(0)
    ratio = mean_b / np.clip(mean_m, 1e-9, None)

    def _lum(x):  # Rec.709
        return x @ np.array([0.2126, 0.7152, 0.0722])
    lum_ratio = float(_lum(mean_b) / max(_lum(mean_m), 1e-9))
    median_ratio = list(np.median(
        b[pos] / np.clip(m[pos], 1e-9, None), axis=0))

    in_band = [bool(IN_BAND[0] <= r <= IN_BAND[1]) for r in ratio]
    report = {
        "scene": str(args.scene),
        "resolution": [args.width, args.height],
        "spp": {"mitsuba": args.spp_mitsuba, "blender": args.spp_blender},
        "max_depth": args.max_depth,
        "mean_blender": list(map(float, mean_b)),
        "mean_mitsuba": list(map(float, mean_m)),
        "ratio_rgb": list(map(float, ratio)),
        "ratio_luma": lum_ratio,
        "median_ratio_rgb": list(map(float, median_ratio)),
        "in_band": in_band,
        "structural_mismatch_frac": struct_mismatch,
    }

    print(f"\n=== HDR parity: {args.scene.name} ===")
    print(f"  channel   blender    mitsuba    ratio   in[{IN_BAND[0]},{IN_BAND[1]}]")
    for i, ch in enumerate("RGB"):
        print(f"   {ch}      {mean_b[i]:.5f}   {mean_m[i]:.5f}   {ratio[i]:.4f}   {in_band[i]}")
    print(f"  luminance ratio: {lum_ratio:.4f}")
    print(f"  median ratio   : {[round(x,4) for x in median_ratio]}")
    print(f"  structural mismatch (one backend black): {struct_mismatch:.3%}")
    if struct_mismatch >= 0.02:
        print("  ! high structural mismatch -> cameras may be misaligned; ratio is unreliable.")

    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
        print(f"  wrote {args.json}")

    if args.do_assert:
        ok = all(in_band) and struct_mismatch < 0.02
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
