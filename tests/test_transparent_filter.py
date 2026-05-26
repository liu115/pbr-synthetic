"""Unit tests for ``src.transparent_filter``."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from textwrap import dedent

import pytest

from src.transparent_filter import (
    FilterReport,
    _is_transparent_bsdf,
    filter_transparent_scene,
)


def _write(tmp_path: Path, xml: str) -> Path:
    p = tmp_path / "scene.xml"
    p.write_text(dedent(xml).strip())
    return p


def _bsdf(xml: str) -> ET.Element:
    return ET.fromstring(dedent(xml).strip())


def test_leaf_dielectric_is_transparent() -> None:
    assert _is_transparent_bsdf(_bsdf('<bsdf type="dielectric" id="g" />'))


def test_leaf_thindielectric_is_transparent() -> None:
    assert _is_transparent_bsdf(_bsdf('<bsdf type="thindielectric" id="g" />'))


def test_leaf_roughdielectric_is_transparent() -> None:
    assert _is_transparent_bsdf(_bsdf('<bsdf type="roughdielectric" id="g" />'))


def test_leaf_diffuse_is_opaque() -> None:
    assert not _is_transparent_bsdf(_bsdf('<bsdf type="diffuse" id="d" />'))


def test_twosided_wrapper_around_dielectric_is_transparent() -> None:
    assert _is_transparent_bsdf(
        _bsdf(
            """
            <bsdf type="twosided" id="g">
                <bsdf type="dielectric" />
            </bsdf>
            """
        )
    )


def test_twosided_wrapper_around_diffuse_is_opaque() -> None:
    assert not _is_transparent_bsdf(
        _bsdf(
            """
            <bsdf type="twosided" id="d">
                <bsdf type="diffuse" />
            </bsdf>
            """
        )
    )


def test_bumpmap_wrapper_recurses() -> None:
    assert _is_transparent_bsdf(
        _bsdf(
            """
            <bsdf type="bumpmap" id="g">
                <bsdf type="dielectric" />
            </bsdf>
            """
        )
    )


def test_no_transparent_shapes_keeps_every_shape(tmp_path: Path) -> None:
    src = _write(
        tmp_path,
        """
        <scene version="3.0.0">
            <bsdf type="twosided" id="WallBSDF">
                <bsdf type="diffuse" />
            </bsdf>
            <shape type="obj" id="wall">
                <string name="filename" value="wall.obj" />
                <ref id="WallBSDF" />
            </shape>
        </scene>
        """,
    )
    work_dir = tmp_path / "_work"
    out, report = filter_transparent_scene(src, work_dir)

    assert out == work_dir / "scene_filtered.xml"
    assert report == FilterReport(kept_shape_count=1)
    tree = ET.parse(out)
    shapes = tree.getroot().findall("shape")
    assert [s.get("id") for s in shapes] == ["wall"]


def test_relative_filename_paths_are_absolutized(tmp_path: Path) -> None:
    """The filtered XML must carry absolute paths so it loads from any cwd."""
    src = _write(
        tmp_path,
        """
        <scene version="3.0.0">
            <bsdf type="twosided" id="WallBSDF">
                <bsdf type="diffuse">
                    <texture type="bitmap">
                        <string name="filename" value="textures/wall.jpg" />
                    </texture>
                </bsdf>
            </bsdf>
            <shape type="obj" id="wall">
                <string name="filename" value="models/wall.obj" />
                <ref id="WallBSDF" />
            </shape>
        </scene>
        """,
    )
    work_dir = tmp_path / "_work"
    out, _ = filter_transparent_scene(src, work_dir)

    tree = ET.parse(out)
    filenames = [
        elem.get("value")
        for elem in tree.getroot().iter("string")
        if elem.get("name") == "filename"
    ]
    # Both filenames must be absolute and rooted at the original scene's dir.
    expected_obj = str((tmp_path / "models" / "wall.obj").resolve())
    expected_tex = str((tmp_path / "textures" / "wall.jpg").resolve())
    assert expected_obj in filenames
    assert expected_tex in filenames


def test_drops_shapes_referencing_dielectric(tmp_path: Path) -> None:
    src = _write(
        tmp_path,
        """
        <scene version="3.0.0">
            <bsdf type="twosided" id="WallBSDF">
                <bsdf type="diffuse" />
            </bsdf>
            <bsdf type="dielectric" id="GlassBSDF">
                <float name="int_ior" value="1.5" />
            </bsdf>
            <shape type="obj" id="wall">
                <string name="filename" value="wall.obj" />
                <ref id="WallBSDF" />
            </shape>
            <shape type="obj" id="glass">
                <string name="filename" value="glass.obj" />
                <ref id="GlassBSDF" />
            </shape>
        </scene>
        """,
    )
    out, report = filter_transparent_scene(src, tmp_path / "_work")

    tree = ET.parse(out)
    shapes = tree.getroot().findall("shape")
    assert [s.get("id") for s in shapes] == ["wall"]
    assert report.dropped_bsdf_ids == ("GlassBSDF",)
    assert report.dropped_shape_ids == ("glass",)
    assert report.kept_shape_count == 1


def test_drops_multiple_shapes_sharing_one_transparent_bsdf(tmp_path: Path) -> None:
    src = _write(
        tmp_path,
        """
        <scene version="3.0.0">
            <bsdf type="thindielectric" id="GlassBSDF" />
            <shape type="obj" id="g1"><ref id="GlassBSDF" /></shape>
            <shape type="obj" id="g2"><ref id="GlassBSDF" /></shape>
            <shape type="obj" id="g3"><ref id="GlassBSDF" /></shape>
        </scene>
        """,
    )
    _, report = filter_transparent_scene(src, tmp_path / "_work")
    assert sorted(report.dropped_shape_ids) == ["g1", "g2", "g3"]
    assert report.kept_shape_count == 0


def test_drops_shape_with_inline_transparent_bsdf(tmp_path: Path) -> None:
    src = _write(
        tmp_path,
        """
        <scene version="3.0.0">
            <shape type="obj" id="inline_glass">
                <bsdf type="dielectric">
                    <float name="int_ior" value="1.5" />
                </bsdf>
            </shape>
            <shape type="obj" id="inline_diffuse">
                <bsdf type="diffuse" />
            </shape>
        </scene>
        """,
    )
    _, report = filter_transparent_scene(src, tmp_path / "_work")
    assert report.dropped_shape_ids == ("inline_glass",)
    assert report.kept_shape_count == 1


def test_kitchen_scene_drops_known_glass_bsdfs(tmp_path: Path) -> None:
    """Smoke test against the real kitchen scene XML (skipped if not present)."""
    real = Path("/cluster_HDD/umoja/yliu/pbr-test/raw/kitchen/scene_v3.xml")
    if not real.exists():
        pytest.skip("kitchen scene XML not available in this environment")

    _, report = filter_transparent_scene(real, tmp_path / "_work")
    # The kitchen scene contains these dielectric/thindielectric BSDFs.
    expected_subset = {
        "WineGlassesBSDF",
        "RadioGlassBSDF",
        "CookerGlassBSDF",
        "MicrowaveGlassBSDF",
        "MicrowaveDigitalBSDF",
    }
    assert expected_subset.issubset(set(report.dropped_bsdf_ids))
    assert report.kept_shape_count > 0
    assert len(report.dropped_shape_ids) > 0
