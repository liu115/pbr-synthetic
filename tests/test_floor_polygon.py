"""Tests for ``src.scene_utils.extract_floor_polygon``."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest

from src.scene_utils import extract_floor_polygon


def _write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write a minimal OBJ file with the given vertices and triangle faces."""
    lines = ["# unit test mesh"]
    for v in vertices:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    for f in faces:
        # OBJ is 1-indexed.
        lines.append(f"f {int(f[0]) + 1} {int(f[1]) + 1} {int(f[2]) + 1}")
    path.write_text("\n".join(lines) + "\n")


def _make_box_obj(path: Path, half_extents: np.ndarray) -> None:
    """Axis-aligned box centred at origin."""
    hx, hy, hz = half_extents
    verts = np.array(
        [
            [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
            [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
        ]
    )
    faces = np.array(
        [
            [0, 1, 2], [0, 2, 3],  # back
            [4, 6, 5], [4, 7, 6],  # front
            [0, 4, 5], [0, 5, 1],
            [2, 6, 7], [2, 7, 3],
            [0, 3, 7], [0, 7, 4],
            [1, 5, 6], [1, 6, 2],
        ]
    )
    _write_obj(path, verts, faces)


def test_extract_uses_named_walls_when_present(tmp_path: Path) -> None:
    """Bitterli-style scene with shape ``id`` containing 'wall' wins over MBR."""
    walls_obj = tmp_path / "walls.obj"
    # 2x4 footprint in the X-Z plane (Y is up). The "wall" shape draws an
    # explicit rectangular footprint.
    _make_box_obj(walls_obj, np.array([1.0, 1.5, 2.0]))

    # An additional non-wall object — sits inside, has a smaller footprint.
    chair_obj = tmp_path / "chair.obj"
    _make_box_obj(chair_obj, np.array([0.3, 0.5, 0.3]))

    scene_xml = tmp_path / "scene.xml"
    scene_xml.write_text(
        dedent(
            f"""
            <scene version="3.0.0">
              <shape type="obj" id="Walls">
                <string name="filename" value="{walls_obj.name}" />
              </shape>
              <shape type="obj" id="Chair">
                <string name="filename" value="{chair_obj.name}" />
              </shape>
            </scene>
            """
        ).strip()
    )

    poly = extract_floor_polygon(scene_xml, up_axis="y")
    # Convex hull of the walls projection should span [-1, 1] in X and [-2, 2] in Z.
    assert poly.shape[1] == 2
    assert poly[:, 0].min() == pytest.approx(-1.0, abs=1e-6)
    assert poly[:, 0].max() == pytest.approx(1.0, abs=1e-6)
    assert poly[:, 1].min() == pytest.approx(-2.0, abs=1e-6)
    assert poly[:, 1].max() == pytest.approx(2.0, abs=1e-6)


def test_extract_falls_back_to_mbr_without_named_walls(tmp_path: Path) -> None:
    """No wall/floor IDs → MBR of all geometry's floor projection."""
    obj1 = tmp_path / "thing1.obj"
    _make_box_obj(obj1, np.array([1.0, 1.0, 1.0]))
    obj2 = tmp_path / "thing2.obj"
    _make_box_obj(obj2, np.array([0.5, 0.5, 0.5]))

    scene_xml = tmp_path / "scene.xml"
    scene_xml.write_text(
        dedent(
            f"""
            <scene version="3.0.0">
              <shape type="obj" id="Thing1">
                <string name="filename" value="{obj1.name}" />
              </shape>
              <shape type="obj" id="Thing2">
                <string name="filename" value="{obj2.name}" />
              </shape>
            </scene>
            """
        ).strip()
    )

    poly = extract_floor_polygon(scene_xml, up_axis="y")
    # MBR returns exactly 4 corners; for an axis-aligned union these match the
    # outermost (X, Z) box of the larger thing.
    assert poly.shape == (4, 2)
    xs = poly[:, 0]
    zs = poly[:, 1]
    assert xs.min() == pytest.approx(-1.0, abs=1e-6)
    assert xs.max() == pytest.approx(1.0, abs=1e-6)
    assert zs.min() == pytest.approx(-1.0, abs=1e-6)
    assert zs.max() == pytest.approx(1.0, abs=1e-6)


def test_extract_respects_up_axis_z(tmp_path: Path) -> None:
    """When up=z, floor polygon is in (X, Y)."""
    walls_obj = tmp_path / "walls.obj"
    _make_box_obj(walls_obj, np.array([1.0, 2.0, 1.5]))

    scene_xml = tmp_path / "scene.xml"
    scene_xml.write_text(
        dedent(
            f"""
            <scene version="3.0.0">
              <shape type="obj" id="Floor">
                <string name="filename" value="{walls_obj.name}" />
              </shape>
            </scene>
            """
        ).strip()
    )

    poly = extract_floor_polygon(scene_xml, up_axis="z")
    # With up=z, the polygon is over (X, Y) → spans (-1, 1) × (-2, 2).
    assert poly[:, 0].min() == pytest.approx(-1.0, abs=1e-6)
    assert poly[:, 0].max() == pytest.approx(1.0, abs=1e-6)
    assert poly[:, 1].min() == pytest.approx(-2.0, abs=1e-6)
    assert poly[:, 1].max() == pytest.approx(2.0, abs=1e-6)


def test_extract_on_real_bedroom_scene(tmp_path: Path) -> None:
    """Sanity check against the real bedroom scene (skipped if absent)."""
    bedroom = Path("/cluster_HDD/umoja/yliu/pbr-test/raw/bedroom/scene_v3.xml")
    if not bedroom.exists():
        pytest.skip("bedroom scene not available in this environment")
    poly = extract_floor_polygon(bedroom, up_axis="y")
    assert poly.shape[1] == 2
    assert poly.shape[0] >= 3
    # Heuristic: a realistic bedroom is at least 2m on each side.
    extents = poly.max(axis=0) - poly.min(axis=0)
    assert extents[0] > 2.0
    assert extents[1] > 2.0
