"""End-to-end render smoke test (opt-in via ``-m mitsuba_slow``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.mitsuba_slow


def test_full_pipeline_on_synthetic_box(
    tmp_path: Path, mitsuba_variant: str
) -> None:
    """Build a tiny box scene, run render_scene.main on it, assert outputs."""
    import mitsuba as mi

    scene_xml = tmp_path / "scene.xml"
    scene_xml.write_text(
        """<?xml version="1.0"?>
<scene version="3.0.0">
  <integrator type="path"><integer name="max_depth" value="4"/></integrator>
  <shape type="cube" id="walls">
    <transform name="to_world">
      <scale x="2.0" y="1.2" z="2.0"/>
    </transform>
    <boolean name="flip_normals" value="true"/>
    <bsdf type="twosided">
      <bsdf type="diffuse">
        <rgb name="reflectance" value="0.7, 0.7, 0.7"/>
      </bsdf>
    </bsdf>
  </shape>
  <shape type="sphere" id="lamp">
    <transform name="to_world">
      <translate x="0" y="1.0" z="0"/>
      <scale value="0.15"/>
    </transform>
    <emitter type="area"><rgb name="radiance" value="30, 30, 30"/></emitter>
  </shape>
</scene>
"""
    )

    out = tmp_path / "out"
    from src.render_scene import main as render_main

    rc = render_main(
        [
            "--scene", str(scene_xml),
            "--output", str(out),
            "--num-cameras", "4",
            "--seed", "7",
            "--debug",
            "--mitsuba-variant", mitsuba_variant,
            "--max-depth", "3",
            "--filter-max-inf", "0.05",
            "--filter-min-median", "0.15",
            "--filter-near-threshold", "0.1",
            "--filter-max-near", "0.5",
            "--height-min", "0.4",
            "--height-max", "0.8",
            "--placement-margin", "0.3",
            "--height-margin", "0.1",
        ]
    )
    assert rc == 0, "render_scene.main exited non-zero"

    # Output structure
    for sub in ("rgb", "rgb_hdr", "depth", "normal", "albedo", "roughness", "metallic"):
        assert (out / sub).is_dir(), f"missing {sub}/ dir"
        # At least one file present
        files = list((out / sub).iterdir())
        assert files, f"{sub}/ is empty"

    # JSONs
    import json
    transforms = json.loads((out / "transforms.json").read_text())
    metadata = json.loads((out / "metadata.json").read_text())
    assert transforms["w"] > 0 and transforms["h"] > 0
    assert len(transforms["frames"]) >= 1
    assert metadata["seed"] == 7
    assert metadata["num_cameras_kept"] >= 1
    # Preview present
    assert (out / "preview.png").exists()

    # Depth EXR is non-trivial
    import imageio.v2 as imageio
    d0 = np.asarray(imageio.imread(str(out / "depth" / "0000.exr")), dtype=np.float32)
    assert d0.ndim == 2
    finite = d0[np.isfinite(d0) & (d0 > 0)]
    assert finite.size > 0
    assert float(np.median(finite)) > 0.0
