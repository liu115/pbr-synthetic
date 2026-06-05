"""Unit-level tests for the addon-patching / world helpers that don't need bpy.

The expensive end-to-end tests (real addon install + render) live in
``test_scene_blender.py`` / ``test_mirror_fix_blender.py`` /
``test_world_background_dimming.py`` behind the ``blender_slow`` marker. The
checks here are pure-Python (file I/O + regex) and run in the fast suite.
"""

from __future__ import annotations

from pathlib import Path

from src.scene_blender import _patch_mirror_roughness, _xml_has_env_emitter

# A minimal stand-in for the addon's materials.py: the three smooth writers carry
# the literal ``= 'BECKMANN'`` distribution (what the SHARP->BECKMANN bpy-4 patch
# leaves behind), while the rough writer sets distribution via a function call and
# already writes its own roughness. The mirror patch must touch the former only.
_FAKE_MATERIALS = """\
def write_mi_conductor_bsdf(mi_context, mi_mat, bl_mat_wrap, out_socket_id):
    bl_glossy = bl_mat_wrap.ensure_node_type([out_socket_id], 'ShaderNodeBsdfGlossy', 'BSDF')
    bl_glossy.distribution = 'BECKMANN'  # patched: bpy 4.0 dropped 'SHARP'
    write_mi_rgb_value(mi_context, reflectance, bl_glossy_wrap, 'Color')
    return True

def write_mi_roughconductor_bsdf(mi_context, mi_mat, bl_mat_wrap, out_socket_id):
    bl_glossy = bl_mat_wrap.ensure_node_type([out_socket_id], 'ShaderNodeBsdfGlossy', 'BSDF')
    bl_glossy.distribution = mi_microfacet_to_bl_microfacet(mi_context, mi_mat.get('distribution', 'beckmann'))
    write_mi_roughness_property(mi_context, mi_mat, 'alpha', bl_glossy_wrap, 'Roughness', 0.1)
    return True

def write_mi_dielectric_bsdf(mi_context, mi_mat, bl_mat_wrap, out_socket_id):
    bl_glass = bl_mat_wrap.ensure_node_type([out_socket_id], 'ShaderNodeBsdfGlass', 'BSDF')
    bl_glass.distribution = 'BECKMANN'  # patched: bpy 4.0 dropped 'SHARP'
    return True

def write_mi_thindielectric_bsdf(mi_context, mi_mat, bl_mat_wrap, out_socket_id):
    bl_glass = bl_mat_wrap.ensure_node_type([out_socket_id], 'ShaderNodeBsdfGlass', 'BSDF')
    bl_glass.distribution = 'BECKMANN'  # patched: bpy 4.0 dropped 'SHARP'
    bl_glass.inputs['IOR'].default_value = 1.0
    return True
"""


def _write(tmp_path: Path) -> Path:
    p = tmp_path / "materials.py"
    p.write_text(_FAKE_MATERIALS)
    return p


def test_mirror_patch_injects_the_three_smooth_writers(tmp_path: Path) -> None:
    p = _write(tmp_path)
    _patch_mirror_roughness(p, 0.0)
    out = p.read_text()
    injected = [ln for ln in out.splitlines() if "sharp delta reflector" in ln]
    assert len(injected) == 3
    # Each injection sets Roughness=0.0 on the node created just above it.
    assert "bl_glossy.inputs['Roughness'].default_value = 0.0" in out  # conductor
    assert out.count("bl_glass.inputs['Roughness'].default_value = 0.0") == 2  # both glass


def test_mirror_patch_leaves_roughconductor_alone(tmp_path: Path) -> None:
    p = _write(tmp_path)
    _patch_mirror_roughness(p, 0.0)
    out = p.read_text()
    # roughconductor keeps its real roughness call and gets no injected line.
    rc = out.split("def write_mi_roughconductor_bsdf")[1].split("def ")[0]
    assert "sharp delta reflector" not in rc
    assert "write_mi_roughness_property(mi_context, mi_mat, 'alpha'" in rc


def test_mirror_patch_is_idempotent(tmp_path: Path) -> None:
    p = _write(tmp_path)
    _patch_mirror_roughness(p, 0.0)
    once = p.read_text()
    _patch_mirror_roughness(p, 0.0)
    twice = p.read_text()
    assert once == twice


def test_mirror_patch_value_is_configurable(tmp_path: Path) -> None:
    p = _write(tmp_path)
    _patch_mirror_roughness(p, 0.25)
    out = p.read_text()
    assert "bl_glossy.inputs['Roughness'].default_value = 0.25" in out


def test_xml_env_emitter_detection(tmp_path: Path) -> None:
    def has(xml: str) -> bool:
        f = tmp_path / "s.xml"
        f.write_text(xml)
        return _xml_has_env_emitter(f)

    assert has('<scene><emitter type="envmap"><string name="filename" value="x.exr"/></emitter></scene>')
    assert has('<scene><emitter type="constant"><rgb name="radiance" value="1,1,1"/></emitter></scene>')
    assert has('<scene><emitter   type = "envmap" /></scene>')  # whitespace-tolerant
    assert not has('<scene><shape><emitter type="area"/></shape></scene>')
    assert not has('<scene><bsdf type="conductor"/></scene>')


def test_xml_env_emitter_missing_file_is_false(tmp_path: Path) -> None:
    assert _xml_has_env_emitter(tmp_path / "does-not-exist.xml") is False
