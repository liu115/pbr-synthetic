"""Smoke tests for the Blender scene backend (opt-in via ``-m blender_slow``).

These tests do real bpy + mitsuba-blender add-on work. Run with::

    conda run -n mitsuba pytest -m blender_slow tests/test_scene_blender.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.blender_slow


@pytest.fixture(scope="module")
def synthetic_box_xml(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Tiny Mitsuba XML scene the mitsuba-blender addon can ingest."""
    p = tmp_path_factory.mktemp("blender_scene") / "scene.xml"
    p.write_text(
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
    return p


def test_init_and_load(synthetic_box_xml: Path) -> None:
    """init_blender + load_scene_blender + bbox + inside-room should all work."""
    from src.scene_blender import (
        derive_scene_info_blender,
        init_blender,
        inside_room_test_blender,
        load_scene_blender,
    )

    init_blender(ensure_addon=True, device="CPU", denoiser="NONE")
    obj_mapping, mat_mapping = load_scene_blender(synthetic_box_xml)
    assert obj_mapping, "load_scene_blender returned an empty object mapping"
    # Every material picked up by Cycles must have a unique pass_index.
    pass_indices = [mat.pass_index for mat in mat_mapping.values()]
    assert len(set(pass_indices)) == len(pass_indices)
    assert all(idx >= 1 for idx in pass_indices)

    info = derive_scene_info_blender(
        height_range=(0.3, 1.0),
        placement_margin=0.2,
        height_margin=0.1,
    )
    # Bbox should bound the 2x1.2x2 box (roughly).
    extent = np.asarray(info.bbox_max) - np.asarray(info.bbox_min)
    assert float(extent.min()) > 0.5
    assert float(extent.max()) < 10.0

    # The scene center is inside the room.
    center = 0.5 * (np.asarray(info.bbox_min) + np.asarray(info.bbox_max))
    ok = inside_room_test_blender(center[None, :], up_axis=info.up_axis)
    assert ok.shape == (1,)
    assert bool(ok[0])
