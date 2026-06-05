"""End-to-end test of the phantom-world neutralization (opt-in: -m blender_slow).

The mitsuba-blender importer always leaves a World: a real one for an
``<emitter type="envmap"|"constant"/>``, or a grey fallback (~0.05088 @ strength 1.0)
when the XML has none. Mitsuba treats "no env emitter" as zero radiance, so that
fallback injects phantom ambient. ``load_scene_blender(..., wipe_world=True)`` (the
default) neutralizes it; ``wipe_world=False`` leaves it for the A/B baseline.

Uses an *open* scene (floating spheres, no enclosure) so escaping rays actually see the
world — a closed box would hide the effect entirely.

  T1  open scene: default-world Cycles mean > zeroed-world mean, and zeroed is closer to
      the Mitsuba reference (which has no env).
  T2  emitters-off isolation: with all lamps disabled the default world still lights the
      scene (>0); zeroing it gives ~black. Pins the brightness delta on the world alone.
  T3  envmap regression guard: on a scene WITH a constant/envmap emitter, wipe_world is a
      no-op — the real env world is preserved and the scene stays lit.

Run::

    ~/miniconda3/envs/pbr-capture-blender-test/bin/python -m pytest -m blender_slow \
        tests/test_world_background_dimming.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.blender_slow

_OPEN_SCENE = """<?xml version="1.0"?>
<scene version="3.0.0">
  <integrator type="path"><integer name="max_depth" value="4"/></integrator>
  <bsdf type="diffuse" id="BallBSDF"><rgb name="reflectance" value="0.7, 0.7, 0.7"/></bsdf>
  <bsdf type="diffuse" id="LampBSDF"><rgb name="reflectance" value="0.9, 0.9, 0.9"/></bsdf>
  <shape type="sphere" id="ball">
    <transform name="to_world"><scale value="0.6"/></transform>
    <ref id="BallBSDF"/>
  </shape>
  <shape type="sphere" id="lamp">
    <transform name="to_world"><translate x="1.0" y="1.0" z="-1.0"/><scale value="0.2"/></transform>
    <ref id="LampBSDF"/>
    <emitter type="area"><rgb name="radiance" value="50, 50, 50"/></emitter>
  </shape>
</scene>
"""

_ENV_SCENE = """<?xml version="1.0"?>
<scene version="3.0.0">
  <integrator type="path"><integer name="max_depth" value="4"/></integrator>
  <bsdf type="diffuse" id="BallBSDF"><rgb name="reflectance" value="0.7, 0.7, 0.7"/></bsdf>
  <shape type="sphere" id="ball">
    <transform name="to_world"><scale value="0.6"/></transform>
    <ref id="BallBSDF"/>
  </shape>
  <emitter type="constant"><rgb name="radiance" value="2.0, 2.0, 2.0"/></emitter>
</scene>
"""


@pytest.fixture(scope="module")
def open_xml(tmp_path_factory: pytest.TempPathFactory) -> Path:
    p = tmp_path_factory.mktemp("world_open") / "open.xml"
    p.write_text(_OPEN_SCENE)
    return p


@pytest.fixture(scope="module")
def env_xml(tmp_path_factory: pytest.TempPathFactory) -> Path:
    p = tmp_path_factory.mktemp("world_env") / "env.xml"
    p.write_text(_ENV_SCENE)
    return p


def _pose():
    from src.camera_sampling import CameraPose
    return CameraPose(position=(0.0, 0.0, -3.0), yaw=0.0, pitch=0.0, up_axis="z")


def _intr():
    from src.io_utils import compute_intrinsics
    return compute_intrinsics(60.0, 120, 90)


def _world_strength(bpy) -> float:
    w = bpy.context.scene.world
    if w is None or not getattr(w, "use_nodes", False) or w.node_tree is None:
        return 0.0
    bg = next((n for n in w.node_tree.nodes if n.bl_idname == "ShaderNodeBackground"), None)
    return float(bg.inputs["Strength"].default_value) if bg else 0.0


def _disable_all_lights(bpy) -> None:
    """Zero every emissive material and remove every light so only the world remains."""
    for obj in list(bpy.data.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)
    for mat in bpy.data.materials:
        if not getattr(mat, "use_nodes", False) or mat.node_tree is None:
            continue
        for n in mat.node_tree.nodes:
            if n.bl_idname == "ShaderNodeEmission":
                n.inputs["Strength"].default_value = 0.0


def test_world_dimming_and_mitsuba_alignment(open_xml: Path) -> None:
    from src.scene_blender import init_blender, load_scene_blender
    from src.render_blender import render_beauty_blender
    from src.scene_utils import init_mitsuba, load_scene
    from src.render import render_beauty

    pose, intr = _pose(), _intr()
    init_blender(ensure_addon=True, device="CPU", denoiser="NONE",
                 mirror_roughness=0.0, roughplastic_coat_weight=0.0)

    # default world left in place (phantom ambient)
    load_scene_blender(open_xml, wipe_world=False)
    m_default = render_beauty_blender(pose, intr, spp=128, max_depth=4, seed=0).mean((0, 1))

    # world neutralized (production default)
    load_scene_blender(open_xml, wipe_world=True)
    m_zeroed = render_beauty_blender(pose, intr, spp=128, max_depth=4, seed=0).mean((0, 1))

    # Mitsuba reference (no env emitter -> zero-radiance background)
    init_mitsuba(prefer=None)  # keep the scalar_rgb variant the addon set
    scene = load_scene(open_xml)
    m_mi = render_beauty(scene, pose, intr, spp=256, max_depth=4, seed=0,
                         pose_frame="blender").mean((0, 1))

    assert np.all(m_zeroed < m_default), \
        f"zeroing the world should dim the render: zeroed={m_zeroed} default={m_default}"
    assert np.all(np.abs(m_zeroed - m_mi) < np.abs(m_default - m_mi)), (
        f"zeroed world should be closer to Mitsuba: "
        f"zeroed={m_zeroed} default={m_default} mitsuba={m_mi}"
    )


def test_world_ambient_isolation(open_xml: Path) -> None:
    from src.scene_blender import init_blender, load_scene_blender, _import_bpy
    from src.render_blender import render_beauty_blender

    pose, intr = _pose(), _intr()
    init_blender(ensure_addon=True, device="CPU", denoiser="NONE",
                 mirror_roughness=0.0, roughplastic_coat_weight=0.0)
    bpy = _import_bpy()

    # all lamps off, default grey world present -> the world is the only light
    load_scene_blender(open_xml, wipe_world=False)
    _disable_all_lights(bpy)
    m_amb = render_beauty_blender(pose, intr, spp=128, max_depth=4, seed=0).mean()

    # all lamps off, world zeroed -> nothing emits -> ~black
    load_scene_blender(open_xml, wipe_world=True)
    _disable_all_lights(bpy)
    m_dark = render_beauty_blender(pose, intr, spp=128, max_depth=4, seed=0).mean()

    assert m_amb > 5e-3, f"default world should light the scene on its own, got {m_amb}"
    assert m_dark < 1e-4, f"zeroed world + no emitters should be ~black, got {m_dark}"


def test_envmap_world_is_preserved(env_xml: Path) -> None:
    from src.scene_blender import init_blender, load_scene_blender, _import_bpy
    from src.render_blender import render_beauty_blender

    pose, intr = _pose(), _intr()
    init_blender(ensure_addon=True, device="CPU", denoiser="NONE",
                 mirror_roughness=0.0, roughplastic_coat_weight=0.0)
    bpy = _import_bpy()

    # wipe_world=True must NOT touch a real (constant-emitter) env world.
    load_scene_blender(env_xml, wipe_world=True)
    strength_fixed = _world_strength(bpy)
    m_fixed = render_beauty_blender(pose, intr, spp=128, max_depth=4, seed=0).mean()

    load_scene_blender(env_xml, wipe_world=False)
    strength_raw = _world_strength(bpy)
    m_raw = render_beauty_blender(pose, intr, spp=128, max_depth=4, seed=0).mean()

    assert strength_fixed > 0.0, "env world strength was wrongly zeroed by the fix"
    assert m_fixed > 0.05, f"env scene should be lit by its constant emitter, got {m_fixed}"
    # The guard makes the fix a no-op for env scenes -> renders match.
    assert abs(m_fixed - m_raw) < 1e-3 and abs(strength_fixed - strength_raw) < 1e-6, (
        f"wipe_world should be a no-op on env scenes: "
        f"fixed(mean={m_fixed}, s={strength_fixed}) vs raw(mean={m_raw}, s={strength_raw})"
    )
