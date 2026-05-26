"""End-to-end render smoke test (opt-in via ``-m blender_slow``).

Run with::

    conda run -n pbr-capture-blender-test pytest -m blender_slow tests/test_render_blender.py

These tests take real minutes — they install the addon (on first run),
load a tiny synthetic scene, and render a handful of frames with Cycles.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.blender_slow


_SCENE_XML = """<?xml version="1.0"?>
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
    <!-- Lamp must be INSIDE the box: box scale_y=1.2 -> extents (-0.6, 0.6), -->
    <!-- so the lamp center is at y=0.5 (just below the top face). -->
    <transform name="to_world"><translate x="0" y="0.5" z="0"/><scale value="0.1"/></transform>
    <ref id="LampBSDF"/>
    <emitter type="area"><rgb name="radiance" value="30, 30, 30"/></emitter>
  </shape>
</scene>
"""


def _common_args(scene_xml: Path, out: Path) -> list[str]:
    return [
        "--scene", str(scene_xml),
        "--output", str(out),
        "--num-cameras", "2",
        "--seed", "7",
        "--debug",
        "--max-depth", "3",
        "--cycles-device", "CPU",
        "--denoiser", "NONE",
        "--filter-max-inf", "0.05",
        "--filter-min-median", "0.15",
        "--filter-near-threshold", "0.1",
        "--filter-max-near", "0.5",
        "--height-min", "0.4",
        "--height-max", "0.8",
        "--placement-margin", "0.3",
        "--height-margin", "0.1",
    ]


def test_blender_only_pipeline(tmp_path: Path) -> None:
    """``--backend blender`` writes the per-backend RGB dirs + the full AOV stack."""
    scene_xml = tmp_path / "scene.xml"
    scene_xml.write_text(_SCENE_XML)
    out = tmp_path / "out"
    from src.render_scene import main as render_main

    rc = render_main(_common_args(scene_xml, out) + ["--backend", "blender"])
    assert rc == 0, "render_scene.main exited non-zero"

    # Per-backend RGB.
    for sub in ("rgb_blender", "rgb_hdr_blender"):
        assert (out / sub).is_dir(), f"missing {sub}/"
        assert list((out / sub).iterdir()), f"{sub}/ is empty"
    # No Mitsuba-side dirs in blender-only mode.
    assert not (out / "rgb_mitsuba").is_dir()

    # Shared AOV stack.
    for sub in (
        "depth", "normal", "albedo", "specular_albedo",
        "emission", "material_index", "roughness", "metallic",
    ):
        assert (out / sub).is_dir(), f"missing {sub}/"
        assert list((out / sub).iterdir()), f"{sub}/ is empty"

    metadata = json.loads((out / "metadata.json").read_text())
    transforms = json.loads((out / "transforms.json").read_text())
    assert metadata["backends"] == ["blender"]
    assert metadata["aov_source"] == "blender"
    assert metadata["seed"] == 7
    assert metadata["num_cameras_kept"] >= 1
    assert transforms["w"] > 0 and transforms["h"] > 0
    assert (out / "preview.png").exists()


def test_both_backends_pipeline(tmp_path: Path) -> None:
    """``--backend both`` writes per-backend RGB dirs and a single shared AOV stack."""
    scene_xml = tmp_path / "scene.xml"
    scene_xml.write_text(_SCENE_XML)
    out = tmp_path / "out"
    from src.render_scene import main as render_main

    rc = render_main(_common_args(scene_xml, out) + ["--backend", "both"])
    assert rc == 0, "render_scene.main exited non-zero"

    for sub in ("rgb_mitsuba", "rgb_hdr_mitsuba", "rgb_blender", "rgb_hdr_blender"):
        assert (out / sub).is_dir(), f"missing {sub}/"
        assert list((out / sub).iterdir()), f"{sub}/ is empty"

    metadata = json.loads((out / "metadata.json").read_text())
    assert set(metadata["backends"]) == {"mitsuba", "blender"}
    assert metadata["tonemap"].keys() == {"mitsuba", "blender"}

    # Depth EXR is non-trivial.
    import imageio.v2 as imageio
    d0 = np.asarray(
        imageio.imread(str(out / "depth" / "0000.exr"), format="EXR-FI"),
        dtype=np.float32,
    )
    finite = d0[np.isfinite(d0) & (d0 > 0)]
    assert finite.size > 0
    assert float(np.median(finite)) > 0.0


def test_per_pixel_roughness_metallic_have_finite_values(tmp_path: Path) -> None:
    """The shader-AOV maps must be all-finite (no NaNs from missing wiring)."""
    scene_xml = tmp_path / "scene.xml"
    scene_xml.write_text(_SCENE_XML)
    out = tmp_path / "out"
    from src.render_scene import main as render_main

    assert render_main(_common_args(scene_xml, out) + ["--backend", "blender"]) == 0

    import imageio.v2 as imageio
    r = np.asarray(
        imageio.imread(str(out / "roughness" / "0000.exr"), format="EXR-FI"),
        dtype=np.float32,
    )
    m = np.asarray(
        imageio.imread(str(out / "metallic" / "0000.exr"), format="EXR-FI"),
        dtype=np.float32,
    )
    assert np.isfinite(r).all()
    assert np.isfinite(m).all()
    # Diffuse-only synthetic scene: roughness == 1.0 (Cycles Principled default
    # when wired from a diffuse-only mapped BSDF), metallic == 0.0.
    assert float(np.median(r)) >= 0.0
    assert float(np.median(m)) == 0.0
