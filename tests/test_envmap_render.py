"""Tests for the per-pixel envmap renderer.

Unit-level tests (no Blender) live at the top of this file. The
``blender_slow``-marked test renders a real envmap mosaic on the synthetic
box scene.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.camera_sampling import CameraPose
from src.envmap_render import (
    EnvmapConfig,
    _backproject_pixel,
    _local_frame_from_normal,
    _world_matrix_for_envmap,
)
from src.io_utils import Intrinsics, compute_intrinsics
from src.pose_utils import pose_to_c2w


# ---- Unit tests (no Blender) ------------------------------------------------


def test_grid_shape_derived_from_image_dim() -> None:
    cfg = EnvmapConfig(patch_size=40, env_height=64, env_width=128)
    assert cfg.grid_shape(image_height=320, image_width=640) == (8, 16)
    assert cfg.grid_shape(image_height=480, image_width=640) == (12, 16)
    # Non-divisible dims floor, but produce at least 1.
    assert cfg.grid_shape(image_height=10, image_width=10) == (1, 1)


def test_mosaic_shape_is_grid_times_envmap_resolution() -> None:
    cfg = EnvmapConfig(patch_size=40, env_height=64, env_width=128)
    assert cfg.mosaic_shape(320, 640) == (8 * 64, 16 * 128)


def test_local_frame_from_normal_has_normal_as_y_column() -> None:
    n = np.array([0.0, 1.0, 0.0])
    R = _local_frame_from_normal(n)
    # Column 1 must be the normal direction.
    np.testing.assert_allclose(R[:, 1], n, atol=1e-9)
    # Frame must be orthonormal.
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-9)


def test_local_frame_handles_normal_parallel_to_world_up() -> None:
    """Edge case: normal == world_up means the natural 'right' vector is degenerate."""
    R = _local_frame_from_normal(np.array([0.0, 1.0, 0.0]))
    assert np.isfinite(R).all()
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-9)


def test_local_frame_for_floor_normal() -> None:
    """A floor (normal = world_up) and a wall (normal = world_x) both produce valid frames."""
    for n_world in (
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0),
    ):
        R = _local_frame_from_normal(n_world)
        np.testing.assert_allclose(R[:, 1], n_world, atol=1e-9)
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-6)


def test_world_matrix_offsets_along_normal() -> None:
    pt = np.array([1.0, 2.0, 3.0])
    n = np.array([0.0, 1.0, 0.0])
    m = _world_matrix_for_envmap(pt, n, offset=0.01)
    # Translation is point + offset * normal.
    np.testing.assert_allclose(m[:3, 3], np.array([1.0, 2.01, 3.0]), atol=1e-9)


def test_backproject_at_principal_point_recovers_camera_forward() -> None:
    """A pixel at (cx, cy) with depth t should land t meters in front of the camera."""
    intr = compute_intrinsics(fov_x_deg=60.0, width=640, height=480)
    pose = CameraPose(position=(0.0, 1.0, 0.0), yaw=0.0, pitch=0.0, up_axis="y")
    c2w = pose_to_c2w(pose).astype(np.float64)
    pt = _backproject_pixel(
        px=float(intr.cx), py=float(intr.cy),
        depth_t=2.0, intrinsics=intr, c2w=c2w,
    )
    # OpenCV convention with yaw=0/pitch=0: +Z forward of the camera maps to
    # world -Z (so the camera looks down -Z in world space).
    expected = np.array(pose.position) + 2.0 * pose.forward()
    np.testing.assert_allclose(pt, expected, atol=1e-6)


def test_backproject_off_axis_pixel_lies_at_correct_distance() -> None:
    intr = compute_intrinsics(fov_x_deg=60.0, width=640, height=480)
    pose = CameraPose(position=(0.5, 1.0, -0.5), yaw=0.3, pitch=-0.1, up_axis="y")
    c2w = pose_to_c2w(pose).astype(np.float64)
    pt = _backproject_pixel(
        px=100.0, py=200.0, depth_t=3.5, intrinsics=intr, c2w=c2w,
    )
    # By the ray-t convention, the world distance from the camera origin
    # equals depth_t for a back-projected pixel.
    dist = float(np.linalg.norm(pt - np.array(pose.position)))
    assert dist == pytest.approx(3.5, abs=1e-6)


# ---- Slow integration test --------------------------------------------------


@pytest.mark.blender_slow
def test_envmap_mosaic_on_synthetic_box(tmp_path: Path) -> None:
    """End-to-end: render a real envmap mosaic on the synthetic box scene."""
    from src.envmap_render import render_envmaps_for_frame
    from src.render_blender import render_aov_blender
    from src.scene_blender import (
        ensure_mitsuba_addon,
        init_blender,
        load_scene_blender,
    )

    scene_xml = tmp_path / "scene.xml"
    scene_xml.write_text(
        """<?xml version="1.0"?>
<scene version="3.0.0">
  <integrator type="path"><integer name="max_depth" value="4"/></integrator>
  <bsdf type="twosided" id="WallsBSDF">
    <bsdf type="diffuse">
      <rgb name="reflectance" value="0.7, 0.7, 0.7"/>
    </bsdf>
  </bsdf>
  <bsdf type="diffuse" id="LampBSDF">
    <rgb name="reflectance" value="0.9, 0.9, 0.9"/>
  </bsdf>
  <shape type="cube" id="walls">
    <transform name="to_world"><scale x="2.0" y="1.2" z="2.0"/></transform>
    <boolean name="flip_normals" value="true"/>
    <ref id="WallsBSDF"/>
  </shape>
  <shape type="sphere" id="lamp">
    <transform name="to_world"><translate x="0" y="0.5" z="0"/><scale value="0.1"/></transform>
    <ref id="LampBSDF"/>
    <emitter type="area"><rgb name="radiance" value="30, 30, 30"/></emitter>
  </shape>
</scene>
"""
    )

    init_blender(ensure_addon=True, device="CPU", denoiser="NONE")
    load_scene_blender(scene_xml)

    # A camera near the centre of the box looking forward.
    intr = compute_intrinsics(fov_x_deg=60.0, width=80, height=60)
    pose = CameraPose(position=(0.0, 0.6, 0.5), yaw=0.0, pitch=0.0, up_axis="y")

    aov = render_aov_blender(
        pose=pose, intrinsics=intr, spp=4, max_depth=3, seed=0,
    )
    cfg = EnvmapConfig(patch_size=20, env_height=16, env_width=32, spp=4, max_depth=2)
    mosaic = render_envmaps_for_frame(
        pose=pose, intrinsics=intr,
        depth=aov.depth, normal=aov.normal,
        cfg=cfg, seed=0,
    )
    env_row, env_col = cfg.grid_shape(intr.height, intr.width)
    assert mosaic.shape == (env_row * cfg.env_height, env_col * cfg.env_width, 3)
    finite = mosaic[np.isfinite(mosaic)]
    assert finite.size > 0
    # The synthetic box is lit by a bright lamp — some pixels should be > 0.
    assert float(np.percentile(finite, 90)) > 0.0
