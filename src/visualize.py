"""Viser-based viewer for a rendered multi-view dataset."""

from __future__ import annotations

import argparse
import colorsys
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import trimesh
import viser
import viser.transforms as vt
from numpy.typing import NDArray

# Mesh-handling code lives in src/mesh_utils.py so the renderer can use it
# without dragging viser in. Re-exported here for backward compatibility.
from src.mesh_utils import (  # noqa: F401  (re-exported names used by tests)
    ShapeBinding,
    _DEFAULT_MESH_COLOR,
    _parse_rgb_string,
    clip_top_percentile,
    find_obj_paths,
    load_scene_mesh,
    parse_scene_materials,
    parse_shape_bindings,
)
from src.viz_utils import depth_to_rgb, normal_to_rgb

log = logging.getLogger(__name__)


# ---- Dataset loading ---------------------------------------------------------


def load_transforms(data_dir: Path) -> dict[str, Any]:
    with (data_dir / "transforms.json").open() as f:
        out: dict[str, Any] = json.load(f)
    return out


def load_metadata(data_dir: Path) -> dict[str, Any]:
    with (data_dir / "metadata.json").open() as f:
        out: dict[str, Any] = json.load(f)
    return out


# ---- Coordinate-frame helpers -----------------------------------------------


def c2w_to_viser(c2w_opencv: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert a 4x4 OpenCV camera-to-world matrix to viser ``(wxyz, position)``.

    Viser camera frustums use the OpenCV convention (+Z forward, +X right,
    +Y down), the same as our ``transforms.json``. Pass the rotation through
    unchanged.
    """
    c2w = np.asarray(c2w_opencv, dtype=np.float64).reshape(4, 4)
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    wxyz = vt.SO3.from_matrix(R).wxyz
    return np.asarray(wxyz, dtype=np.float64), t


def _color_for(idx: int, total: int) -> tuple[int, int, int]:
    h = (idx / max(1, total)) * 0.85
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


# ---- Image loading for each modality ----------------------------------------


def _load_rgb(data_dir: Path, frame: dict[str, Any]) -> NDArray[np.uint8]:
    path = data_dir / f"{frame['file_path']}.png"
    img = np.asarray(imageio.imread(str(path)))
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    return img.astype(np.uint8)


def _try_load_pre_rendered_png(data_dir: Path, subdir: str, file_path: str) -> NDArray[np.uint8] | None:
    """If the render pipeline already saved a debug PNG for this frame, use it."""
    stem = Path(file_path).stem
    candidate = data_dir / subdir / f"{stem}.png"
    if not candidate.exists():
        return None
    img = np.asarray(imageio.imread(str(candidate)), dtype=np.uint8)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    return img


def _load_depth_viz(data_dir: Path, frame: dict[str, Any]) -> NDArray[np.uint8]:
    pre = _try_load_pre_rendered_png(data_dir, "depth_rgb", frame["depth_path"])
    if pre is not None:
        return pre
    d = np.asarray(imageio.imread(str(data_dir / frame["depth_path"])), dtype=np.float32)
    return depth_to_rgb(d)


def _load_normal_viz(data_dir: Path, frame: dict[str, Any]) -> NDArray[np.uint8]:
    pre = _try_load_pre_rendered_png(data_dir, "normal_rgb", frame["normal_path"])
    if pre is not None:
        return pre
    n = np.asarray(imageio.imread(str(data_dir / frame["normal_path"])), dtype=np.float32)
    if n.shape[-1] != 3:
        return np.zeros((*n.shape[:2], 3), dtype=np.uint8)
    return normal_to_rgb(n)


_LOADERS = {
    "rgb": _load_rgb,
    "depth": _load_depth_viz,
    "normal": _load_normal_viz,
}


# ---- Server + GUI ------------------------------------------------------------


def _load_mesh_for_viewer(
    data_dir: Path, scene_xml: Path | None
) -> trimesh.Trimesh | None:
    """Prefer the renderer-baked ``scene.ply`` (if any) over rebuilding from XML.

    The PLY already has tessellated + texture-baked vertex colors from the
    last render, so the viewer shows exactly what 3DGS init would see.
    Falls back to ``load_scene_mesh(scene_xml)`` when the PLY is missing.
    """
    ply_path = data_dir / "scene.ply"
    if ply_path.exists():
        loaded = trimesh.load(str(ply_path), process=False, force="mesh")
        if isinstance(loaded, trimesh.Trimesh):
            log.info("Loaded pre-baked mesh: %s", ply_path)
            return loaded
        log.warning("Could not load %s as a mesh; falling back to XML.", ply_path)
    if scene_xml is None or not scene_xml.exists():
        return None
    return load_scene_mesh(scene_xml)


def _add_mesh(
    server: viser.ViserServer,
    data_dir: Path,
    scene_xml: Path | None,
    up_axis: str,
    ceiling_cutoff_percentile: float,
) -> None:
    mesh = _load_mesh_for_viewer(data_dir, scene_xml)
    if mesh is None:
        log.warning("Skipping mesh add; no geometry found.")
        return
    if ceiling_cutoff_percentile < 100.0:
        mesh = clip_top_percentile(mesh, up_axis, ceiling_cutoff_percentile)
    log.info("Loaded mesh: %d verts, %d faces", len(mesh.vertices), len(mesh.faces))
    server.scene.add_mesh_trimesh(name="/scene_mesh", mesh=mesh)


def _add_frustums(
    server: viser.ViserServer,
    data_dir: Path,
    transforms: dict[str, Any],
    initial_modality: str,
    frustum_scale: float,
) -> list[viser.CameraFrustumHandle]:
    fov_x = float(transforms["camera_angle_x"])
    aspect = float(transforms["w"]) / float(transforms["h"])
    # viser expects vertical fov for add_camera_frustum.
    fov_y = 2.0 * float(np.arctan(np.tan(fov_x / 2.0) / aspect))
    frames = transforms["frames"]
    handles: list[viser.CameraFrustumHandle] = []
    loader = _LOADERS[initial_modality]
    for i, frame in enumerate(frames):
        c2w = np.asarray(frame["transform_matrix"], dtype=np.float64)
        wxyz, pos = c2w_to_viser(c2w)
        try:
            img = loader(data_dir, frame)
        except FileNotFoundError:
            img = None
        h = server.scene.add_camera_frustum(
            name=f"/cameras/{i:04d}",
            fov=fov_y,
            aspect=aspect,
            scale=frustum_scale,
            color=_color_for(i, len(frames)),
            image=img,
            wxyz=tuple(wxyz),
            position=tuple(pos),
        )
        handles.append(h)
    return handles


def _setup_gui(
    server: viser.ViserServer,
    handles: list[viser.CameraFrustumHandle],
    data_dir: Path,
    transforms: dict[str, Any],
    initial_scale: float,
) -> None:
    frames = transforms["frames"]
    n = len(frames)

    with server.gui.add_folder("Frustums"):
        visibility_modes = ("all", "first half", "second half", "every 4th")
        show_dropdown = server.gui.add_dropdown(
            "Show", visibility_modes, initial_value="all"
        )
        modality_dropdown = server.gui.add_dropdown(
            "Image on frustum",
            tuple(_LOADERS.keys()),
            initial_value="rgb",
        )
        scale_slider = server.gui.add_slider(
            "Frustum scale", min=0.02, max=1.0, step=0.01,
            initial_value=float(initial_scale),
        )

    def _apply_visibility(_event: Any = None) -> None:
        mode = show_dropdown.value
        for i, h in enumerate(handles):
            if mode == "all":
                h.visible = True
            elif mode == "first half":
                h.visible = i < n // 2
            elif mode == "second half":
                h.visible = i >= n // 2
            elif mode == "every 4th":
                h.visible = (i % 4) == 0

    def _apply_modality(_event: Any = None) -> None:
        loader = _LOADERS[modality_dropdown.value]
        for i, h in enumerate(handles):
            try:
                h.image = loader(data_dir, frames[i])
            except FileNotFoundError:
                continue

    def _apply_scale(_event: Any = None) -> None:
        s = float(scale_slider.value)
        for h in handles:
            h.scale = s

    show_dropdown.on_update(_apply_visibility)
    modality_dropdown.on_update(_apply_modality)
    scale_slider.on_update(_apply_scale)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Viser viewer for a rendered dataset.")
    p.add_argument("--data", type=Path, required=True,
                   help="Path to a rendered dataset directory.")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--frustum-scale", type=float, default=0.15,
                   help="Initial frustum scale (live-adjustable in the GUI).")
    p.add_argument("--no-mesh", action="store_true",
                   help="Skip loading the scene mesh.")
    p.add_argument("--ceiling-cutoff-percentile", type=float, default=95.0,
                   help="Drop mesh faces whose vertices sit above this height "
                        "percentile so the ceiling doesn't occlude top-down "
                        "views. Set to 100 to disable cropping. Default: 95.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s"
    )
    args = parse_args(argv)
    data_dir: Path = args.data
    transforms = load_transforms(data_dir)
    metadata = load_metadata(data_dir)

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.add_frame("/world", show_axes=True, axes_length=1.0)

    if not args.no_mesh:
        scene_xml_str = metadata.get("scene_xml")
        scene_xml = Path(scene_xml_str) if scene_xml_str else None
        up_axis = str(transforms.get("up_axis", "y"))
        _add_mesh(
            server, data_dir, scene_xml,
            up_axis=up_axis,
            ceiling_cutoff_percentile=args.ceiling_cutoff_percentile,
        )

    handles = _add_frustums(
        server, data_dir, transforms,
        initial_modality="rgb", frustum_scale=args.frustum_scale,
    )
    _setup_gui(server, handles, data_dir, transforms, initial_scale=args.frustum_scale)

    log.info("Viewer running. %d cameras. URL printed above. Ctrl-C to stop.",
             len(handles))
    try:
        while True:
            time.sleep(60.0)
    except KeyboardInterrupt:
        log.info("Stopping viewer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
