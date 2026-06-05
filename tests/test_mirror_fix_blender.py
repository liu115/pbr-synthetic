"""End-to-end test of the mirror (delta conductor) Roughness fix (opt-in: -m blender_slow).

Proves that the flag-gated ``mirror_roughness`` patch turns a Mitsuba
``<bsdf type="conductor"/>`` delta-mirror into a *sharp* Cycles reflector instead of
the half-rough metal the un-patched addon produces (it never writes the Glossy
``Roughness`` socket, so it inherits Cycles' default ~0.5).

Two independent checks:
  (a) static node-graph — baseline (``mirror_roughness=None``) leaves Roughness > 0;
      patched (``mirror_roughness=0.0``) sets it to exactly 0.0, unlinked, while
      ``roughconductor`` keeps its real roughness.
  (b) render — the patched mirror reflects the bright target sharply, so its pixel
      region has markedly higher spatial contrast than the rough baseline.

Run::

    ~/miniconda3/envs/pbr-capture-blender-test/bin/python -m pytest -m blender_slow \
        tests/test_mirror_fix_blender.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.blender_slow


@pytest.fixture(scope="module")
def mirror_xml(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A box with a flat conductor mirror on the back wall facing a bright target."""
    p = tmp_path_factory.mktemp("mirror_scene") / "mirror.xml"
    p.write_text(
        """<?xml version="1.0"?>
<scene version="3.0.0">
  <integrator type="path"><integer name="max_depth" value="6"/></integrator>
  <bsdf type="twosided" id="WallsBSDF">
    <bsdf type="diffuse"><rgb name="reflectance" value="0.6, 0.6, 0.6"/></bsdf>
  </bsdf>
  <bsdf type="conductor" id="MirrorBSDF">
    <string name="material" value="none"/>
  </bsdf>
  <bsdf type="diffuse" id="LampBSDF"><rgb name="reflectance" value="0.9, 0.9, 0.9"/></bsdf>

  <shape type="cube" id="walls">
    <transform name="to_world"><scale x="2.0" y="1.5" z="2.0"/></transform>
    <boolean name="flip_normals" value="true"/>
    <ref id="WallsBSDF"/>
  </shape>
  <!-- mirror plane standing in front of the back wall -->
  <shape type="rectangle" id="mirror">
    <transform name="to_world">
      <scale x="1.2" y="1.2" z="1.0"/>
      <rotate x="1" angle="0"/>
      <translate x="0" y="0" z="1.85"/>
    </transform>
    <ref id="MirrorBSDF"/>
  </shape>
  <!-- bright emissive target the mirror reflects -->
  <shape type="sphere" id="lamp">
    <transform name="to_world"><translate x="-1.2" y="0.4" z="-1.2"/><scale value="0.18"/></transform>
    <ref id="LampBSDF"/>
    <emitter type="area"><rgb name="radiance" value="60, 60, 60"/></emitter>
  </shape>
</scene>
"""
    )
    return p


def _glossy_roughness(bpy) -> list[tuple[str, float, bool]]:
    out = []
    for m in bpy.data.materials:
        if not getattr(m, "use_nodes", False) or m.node_tree is None:
            continue
        for n in m.node_tree.nodes:
            if n.bl_idname == "ShaderNodeBsdfGlossy":
                r = n.inputs["Roughness"]
                out.append((m.name, float(r.default_value), bool(r.is_linked)))
    return out


def test_mirror_static_nodegraph(mirror_xml: Path) -> None:
    from src.scene_blender import (
        ensure_mitsuba_addon,
        init_blender,
        load_scene_blender,
        _import_bpy,
    )

    # --- baseline: addon installed WITHOUT the mirror patch ---
    ensure_mitsuba_addon(force_reinstall=True, mirror_roughness=None,
                         roughplastic_coat_weight=0.0)
    init_blender(ensure_addon=False, device="CPU", denoiser="NONE")
    load_scene_blender(mirror_xml)
    bpy = _import_bpy()
    baseline = _glossy_roughness(bpy)
    assert baseline, "no conductor Glossy node found in the imported scene"
    assert all(r > 0.0 for _, r, _ in baseline), \
        f"baseline conductor should inherit a non-zero Roughness, got {baseline}"
    # Informational: the inherited default is expected to be ~0.5 (doc-unverifiable).
    if not all(abs(r - 0.5) < 1e-6 for _, r, _ in baseline):
        print(f"NOTE: baseline Glossy roughness != 0.5 exactly: {baseline}")

    # --- patched: mirror_roughness=0.0 ---
    ensure_mitsuba_addon(force_reinstall=True, mirror_roughness=0.0,
                         roughplastic_coat_weight=0.0)
    load_scene_blender(mirror_xml)
    patched = _glossy_roughness(bpy)
    assert patched
    for name, r, linked in patched:
        assert r == 0.0 and not linked, \
            f"{name}: expected Roughness 0.0 unlinked after patch, got r={r} linked={linked}"


def test_mirror_render_is_sharper_when_patched(mirror_xml: Path) -> None:
    """The sharp (patched) mirror reflects the bright lamp as a localized hot spot,
    giving its pixel region much higher spatial contrast than the rough baseline."""
    from src.io_utils import compute_intrinsics
    from src.camera_sampling import CameraPose
    from src.scene_blender import (
        ensure_mitsuba_addon,
        init_blender,
        load_scene_blender,
    )
    from src.render_blender import render_beauty_blender, render_aov_blender

    intr = compute_intrinsics(60.0, 160, 120)
    # Camera near the front wall looking at the mirror on the back wall (+z).
    pose = CameraPose(position=(0.0, 0.0, -1.6), yaw=0.0, pitch=0.0, up_axis="z")

    def _render(mirror_roughness):
        ensure_mitsuba_addon(force_reinstall=True, mirror_roughness=mirror_roughness,
                             roughplastic_coat_weight=0.0)
        init_blender(ensure_addon=False, device="CPU", denoiser="NONE")
        load_scene_blender(mirror_xml)
        hdr = render_beauty_blender(pose, intr, spp=128, max_depth=6, seed=0)
        aov = render_aov_blender(pose, intr, spp=8, max_depth=6, seed=0)
        return hdr, aov.shape_index

    hdr_rough, idx_rough = _render(None)
    hdr_sharp, idx_sharp = _render(0.0)

    # Locate the mirror pixels via the object-index pass (the mirror is its own shape).
    def _mirror_contrast(hdr, shape_index) -> float:
        lum = hdr @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
        # The mirror is the shape whose pixels show the largest luminance spread
        # (it reflects the bright lamp); pick the index region with max variance.
        best = 0.0
        for sidx in np.unique(shape_index):
            if sidx <= 0:
                continue
            mask = shape_index == sidx
            if mask.sum() < 20:
                continue
            best = max(best, float(np.var(lum[mask])))
        return best

    var_rough = _mirror_contrast(hdr_rough, idx_rough)
    var_sharp = _mirror_contrast(hdr_sharp, idx_sharp)
    assert var_sharp > 1.5 * max(var_rough, 1e-8), (
        f"patched mirror should be sharper (higher reflected-highlight variance): "
        f"sharp={var_sharp:.4g} vs rough={var_rough:.4g}"
    )
