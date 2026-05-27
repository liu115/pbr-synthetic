"""TSDF-fuse rendered RGB + depth into a colored mesh for visual verification.

The fused mesh is a sanity check on the camera-coordinate-system chain
end-to-end: if the depth values, c2w poses, and intrinsics in
``transforms.json`` line up, dropping every frame into a TSDF volume should
reconstruct a recognisable copy of the room. Compare against the renderer's
own ``scene.ply`` (which is exported via the same pipeline) — both live in
the Blender Z-up world frame, so they overlap directly in MeshLab.

Usage::

    python scripts/tsdf_fuse.py --data /path/to/output [--backend blender]

The default output is ``<data>/fused.ply``.

Conventions (as of the post-OpenCV-cleanup output schema):

* ``depth/{idx:04d}.exr`` is perspective z-depth in meters (camera-space +Z).
  Open3D's TSDF wants exactly this — no conversion needed.
* ``transforms.json[frames][i].transform_matrix`` is c2w in OpenCV
  convention (+X right, +Y down, +Z forward). Invert directly to get the
  world-to-camera extrinsic.
* ``scene.ply`` and the c2w poses share the Blender Z-up world. Pass
  ``--world-frame mitsuba`` to rotate the output mesh to the original
  Mitsuba Y-up frame instead.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import open3d as o3d  # type: ignore[import-not-found]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.scene_utils import R_MITSUBA_TO_BLENDER_4  # noqa: E402

log = logging.getLogger(__name__)

# Rotation from the Blender Z-up world back to the Mitsuba Y-up world (used
# only when --world-frame mitsuba is requested).
_R_BLENDER_TO_MITSUBA_4 = R_MITSUBA_TO_BLENDER_4.T


def _resolve_rgb_path(
    data_dir: Path, frame: dict[str, Any], backend: str
) -> Path | None:
    """Find the LDR RGB PNG for ``backend`` within a transforms.json frame."""
    key = f"file_path_{backend}"
    rel = frame.get(key) or frame.get("file_path")
    if rel is None:
        return None
    p = data_dir / f"{rel}.png"
    return p if p.exists() else None


def _detect_backend(transforms: dict[str, Any]) -> str:
    """Pick a backend that actually has files written."""
    first = transforms["frames"][0]
    for b in ("blender", "mitsuba"):
        if f"file_path_{b}" in first:
            return b
    # Legacy single-backend layout (rgb/{stem}.png).
    return "blender"


def fuse(
    data_dir: Path,
    out_path: Path,
    backend: str | None = None,
    world_frame: str = "blender",
    voxel_length: float = 0.05,
    sdf_trunc: float = 0.20,
    depth_trunc: float = 8.0,
    frame_stride: int = 1,
    max_frames: int | None = None,
) -> Path:
    transforms = json.loads((data_dir / "transforms.json").read_text())
    intr = o3d.camera.PinholeCameraIntrinsic(
        width=int(transforms["w"]),
        height=int(transforms["h"]),
        fx=float(transforms["fl_x"]),
        fy=float(transforms["fl_y"]),
        cx=float(transforms["cx"]),
        cy=float(transforms["cy"]),
    )
    if backend is None:
        backend = _detect_backend(transforms)
        log.info("Auto-detected backend: %s", backend)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    frames = transforms["frames"][::frame_stride]
    if max_frames is not None:
        frames = frames[:max_frames]

    n_integrated = 0
    n_skipped = 0
    for i, frame in enumerate(frames):
        rgb_path = _resolve_rgb_path(data_dir, frame, backend)
        depth_path = data_dir / frame["depth_path"]
        if rgb_path is None or not depth_path.exists():
            log.warning("Frame %d: missing rgb or depth (%s, %s)",
                        i, rgb_path, depth_path)
            n_skipped += 1
            continue

        rgb = np.asarray(imageio.imread(rgb_path), dtype=np.uint8)
        if rgb.ndim == 2:
            rgb = np.stack([rgb] * 3, axis=-1)
        rgb = np.ascontiguousarray(rgb[..., :3])

        z_depth = np.asarray(
            imageio.imread(depth_path, format="EXR-FI"),  # type: ignore[arg-type]
            dtype=np.float32,
        )
        if z_depth.ndim == 3:
            z_depth = z_depth[..., 0]
        z_depth = np.where(
            np.isfinite(z_depth) & (z_depth > 0.0) & (z_depth < depth_trunc),
            z_depth,
            0.0,
        ).astype(np.float32)
        z_depth = np.ascontiguousarray(z_depth)

        # transforms.json stores c2w in OpenCV convention in the Blender Z-up
        # world frame. Optionally rotate to Mitsuba Y-up for users who want
        # the original XML coordinate system.
        c2w = np.asarray(frame["transform_matrix"], dtype=np.float64)
        if world_frame == "mitsuba":
            c2w = _R_BLENDER_TO_MITSUBA_4 @ c2w
        extrinsic = np.linalg.inv(c2w)

        color_img = o3d.geometry.Image(rgb)
        depth_img = o3d.geometry.Image(z_depth)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img, depth_img,
            depth_scale=1.0,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )
        volume.integrate(rgbd, intr, extrinsic)
        n_integrated += 1
        log.info("Integrated frame %d (%s)", i, rgb_path.name)

    log.info("Integrated %d frames, skipped %d. Extracting mesh…",
             n_integrated, n_skipped)
    if n_integrated == 0:
        raise RuntimeError("No frames were integrated; check --data path + backend.")

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(out_path), mesh)

    log.info(
        "Wrote %s (%d verts, %d triangles)",
        out_path, len(mesh.vertices), len(mesh.triangles),
    )
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(
        description="TSDF-fuse rendered RGB + depth into a colored mesh."
    )
    ap.add_argument("--data", type=Path, required=True,
                    help="Render output dir containing transforms.json + depth/ + rgb_<backend>/.")
    ap.add_argument("--output", type=Path, default=None,
                    help="Output PLY path (default: <data>/fused.ply).")
    ap.add_argument("--backend", choices=["blender", "mitsuba"], default=None,
                    help="Which RGB source to fuse. Default: auto-detect.")
    ap.add_argument("--world-frame", choices=["mitsuba", "blender"], default="blender",
                    help="Frame for the output mesh. 'blender' (default) "
                         "aligns with scene.ply and the saved poses; 'mitsuba' "
                         "rotates back to the original XML Y-up frame.")
    ap.add_argument("--voxel-length", type=float, default=0.05)
    ap.add_argument("--sdf-trunc", type=float, default=0.20)
    ap.add_argument("--depth-trunc", type=float, default=8.0,
                    help="Max depth in meters (pixels beyond this are dropped).")
    ap.add_argument("--frame-stride", type=int, default=1,
                    help="Use every Nth frame to speed up fusion (default 1 = all).")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Cap on the number of frames to fuse (after --frame-stride).")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    out = args.output if args.output is not None else (args.data / "fused.ply")
    fuse(
        data_dir=args.data,
        out_path=out,
        backend=args.backend,
        world_frame=args.world_frame,
        voxel_length=args.voxel_length,
        sdf_trunc=args.sdf_trunc,
        depth_trunc=args.depth_trunc,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
