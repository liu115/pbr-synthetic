"""Unit tests for the XML parsing helpers in src/visualize.py."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

import trimesh

from src.visualize import (
    _DEFAULT_MESH_COLOR,
    _parse_rgb_string,
    clip_top_percentile,
    parse_scene_materials,
    parse_shape_bindings,
)


def _tall_dense_mesh(up_axis: str = "y") -> trimesh.Trimesh:
    """An icosphere stretched along ``up_axis``. Enough vertices to exercise
    percentile cropping cleanly (unlike an 8-vertex box)."""
    m = trimesh.creation.icosphere(subdivisions=3)
    scale = np.eye(4)
    idx = {"x": 0, "y": 1, "z": 2}[up_axis]
    scale[idx, idx] = 2.0  # stretch along the up axis
    m.apply_transform(scale)
    return m


def _write_scene_xml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "scene.xml"
    p.write_text(
        f'<?xml version="1.0"?>\n<scene version="3.0.0">\n{body}\n</scene>\n'
    )
    return p


def test_parse_rgb_string_three_values() -> None:
    assert _parse_rgb_string("0.5, 0.25, 0.125") == (0.5, 0.25, 0.125)


def test_parse_rgb_string_single_value() -> None:
    assert _parse_rgb_string("0.4") == (0.4, 0.4, 0.4)


def test_parse_rgb_string_garbage_returns_none() -> None:
    assert _parse_rgb_string("not numbers") is None


def test_parse_scene_materials_direct_rgb(tmp_path: Path) -> None:
    xml = _write_scene_xml(
        tmp_path,
        """
        <bsdf type="diffuse" id="A">
            <rgb name="reflectance" value="0.2, 0.4, 0.6"/>
        </bsdf>
        """,
    )
    mats = parse_scene_materials(xml)
    assert mats["A"] == (0.2, 0.4, 0.6)


def test_parse_scene_materials_nested_twosided(tmp_path: Path) -> None:
    xml = _write_scene_xml(
        tmp_path,
        """
        <bsdf type="twosided" id="B">
            <bsdf type="diffuse">
                <rgb name="reflectance" value="0.8, 0.7, 0.6"/>
            </bsdf>
        </bsdf>
        """,
    )
    mats = parse_scene_materials(xml)
    assert mats["B"] == (0.8, 0.7, 0.6)


def test_parse_scene_materials_textured_uses_mean(tmp_path: Path) -> None:
    # Pre-create a 16x16 RGB texture with known mean.
    tex_path = tmp_path / "tex.png"
    arr = np.zeros((16, 16, 3), dtype=np.uint8)
    arr[..., 0] = 100  # red
    arr[..., 1] = 50   # green
    arr[..., 2] = 200  # blue
    imageio.imwrite(str(tex_path), arr)
    xml = _write_scene_xml(
        tmp_path,
        """
        <bsdf type="twosided" id="C">
            <bsdf type="diffuse">
                <texture name="reflectance" type="bitmap">
                    <string name="filename" value="tex.png"/>
                </texture>
            </bsdf>
        </bsdf>
        """,
    )
    mats = parse_scene_materials(xml)
    r, g, b = mats["C"]
    assert r == pytest.approx(100 / 255, abs=0.02)
    assert g == pytest.approx(50 / 255, abs=0.02)
    assert b == pytest.approx(200 / 255, abs=0.02)


def test_parse_scene_materials_falls_back_to_gray(tmp_path: Path) -> None:
    xml = _write_scene_xml(
        tmp_path,
        """
        <bsdf type="conductor" id="D">
            <string name="material" value="Au"/>
        </bsdf>
        """,
    )
    mats = parse_scene_materials(xml)
    assert mats["D"] == _DEFAULT_MESH_COLOR


def test_parse_shape_bindings_links_ref_and_filename(tmp_path: Path) -> None:
    xml = _write_scene_xml(
        tmp_path,
        """
        <bsdf type="diffuse" id="MyBSDF">
            <rgb name="reflectance" value="0.5, 0.5, 0.5"/>
        </bsdf>
        <shape type="obj" id="MyShape">
            <string name="filename" value="models/a.obj"/>
            <ref id="MyBSDF"/>
        </shape>
        """,
    )
    bindings = parse_shape_bindings(xml)
    assert len(bindings) == 1
    b = bindings[0]
    assert b.obj_path == (tmp_path / "models" / "a.obj").resolve()
    assert b.bsdf_id == "MyBSDF"
    assert b.transform is None


def test_parse_shape_bindings_inline_no_ref(tmp_path: Path) -> None:
    xml = _write_scene_xml(
        tmp_path,
        """
        <shape type="obj">
            <string name="filename" value="m.obj"/>
        </shape>
        """,
    )
    bindings = parse_shape_bindings(xml)
    assert len(bindings) == 1
    assert bindings[0].bsdf_id is None


def test_parse_shape_bindings_non_identity_transform(tmp_path: Path) -> None:
    matrix = (
        "1 0 0 5  0 1 0 0  0 0 1 0  0 0 0 1"
    )
    xml = _write_scene_xml(
        tmp_path,
        f"""
        <shape type="obj">
            <string name="filename" value="m.obj"/>
            <transform name="to_world">
                <matrix value="{matrix}"/>
            </transform>
        </shape>
        """,
    )
    b = parse_shape_bindings(xml)[0]
    assert b.transform is not None
    assert b.transform.shape == (4, 4)
    assert b.transform[0, 3] == pytest.approx(5.0)


def test_parse_shape_bindings_identity_transform_normalized_to_none(
    tmp_path: Path,
) -> None:
    matrix = "1 0 0 0  0 1 0 0  0 0 1 0  0 0 0 1"
    xml = _write_scene_xml(
        tmp_path,
        f"""
        <shape type="obj">
            <string name="filename" value="m.obj"/>
            <transform name="to_world">
                <matrix value="{matrix}"/>
            </transform>
        </shape>
        """,
    )
    b = parse_shape_bindings(xml)[0]
    assert b.transform is None


def test_clip_top_percentile_disabled_returns_input() -> None:
    mesh = _tall_dense_mesh("y")
    out = clip_top_percentile(mesh, up_axis="y", percentile=100.0)
    assert out is mesh


def test_clip_top_percentile_drops_top_vertices_along_up_axis() -> None:
    mesh = _tall_dense_mesh("y")  # icosphere stretched 2x along Y
    full_max = float(np.asarray(mesh.vertices)[:, 1].max())
    cropped = clip_top_percentile(mesh, up_axis="y", percentile=80.0)
    verts = np.asarray(cropped.vertices, dtype=np.float64)
    assert verts.size > 0
    # Top of the cropped mesh must be strictly below the original top.
    assert verts[:, 1].max() < full_max
    # Bottom should be untouched.
    assert verts[:, 1].min() == pytest.approx(-full_max, abs=1e-6)


def test_clip_top_percentile_respects_z_up() -> None:
    mesh = _tall_dense_mesh("z")
    full_max = float(np.asarray(mesh.vertices)[:, 2].max())
    cropped = clip_top_percentile(mesh, up_axis="z", percentile=50.0)
    verts = np.asarray(cropped.vertices, dtype=np.float64)
    assert verts[:, 2].max() < full_max


def test_export_colored_mesh_ply_roundtrip(tmp_path: Path) -> None:
    """Flat-RGB BSDF -> PLY with vertex colors that round-trip."""
    from src.mesh_utils import export_colored_mesh_ply

    obj_path = tmp_path / "models"
    obj_path.mkdir()
    (obj_path / "tri.obj").write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
    )
    xml = _write_scene_xml(
        tmp_path,
        """
        <bsdf type="diffuse" id="MyBSDF">
            <rgb name="reflectance" value="0.1, 0.5, 0.9"/>
        </bsdf>
        <shape type="obj">
            <string name="filename" value="models/tri.obj"/>
            <ref id="MyBSDF"/>
        </shape>
        """,
    )
    ply = tmp_path / "scene.ply"
    out = export_colored_mesh_ply(xml, ply)
    assert out is not None
    assert out.path == ply.resolve()
    assert out.total_verts == 3
    assert out.total_faces == 1
    assert ply.exists() and ply.stat().st_size > 0

    loaded = trimesh.load(str(ply), process=False, force="mesh")
    assert isinstance(loaded, trimesh.Trimesh)
    assert len(loaded.vertices) == 3
    vc = np.asarray(loaded.visual.vertex_colors)
    # Reflectance (0.1, 0.5, 0.9) -> uint8 (25/26, 128, 230).
    assert vc.shape == (3, 4)
    expected = np.array(
        [
            round(0.1 * 255),
            round(0.5 * 255),
            round(0.9 * 255),
        ],
        dtype=np.int32,
    )
    diff = np.abs(vc[:, :3].astype(np.int32) - expected[None, :]).max()
    assert diff <= 1, f"vertex colors differ by {diff}: got {vc[:, :3]}"


def test_export_colored_mesh_ply_empty_scene(tmp_path: Path) -> None:
    """An XML with no OBJ shapes returns None and doesn't create a file."""
    from src.mesh_utils import export_colored_mesh_ply

    xml = _write_scene_xml(tmp_path, "")
    ply = tmp_path / "scene.ply"
    out = export_colored_mesh_ply(xml, ply)
    assert out is None
    assert not ply.exists()


# ---------------------------------------------------------------------------
# New tests: diffuse_reflectance + tessellate-then-bake
# ---------------------------------------------------------------------------


def _write_obj_with_uv(
    path: Path, verts: list[tuple[float, float, float]],
    uvs: list[tuple[float, float]],
    faces: list[tuple[int, int, int]],
) -> None:
    """Write a Wavefront OBJ with explicit ``vt`` lines and ``f a/a b/b c/c``."""
    lines: list[str] = []
    for v in verts:
        lines.append(f"v {v[0]} {v[1]} {v[2]}")
    for uv in uvs:
        lines.append(f"vt {uv[0]} {uv[1]}")
    for f in faces:
        a, b, c = f
        lines.append(f"f {a}/{a} {b}/{b} {c}/{c}")
    path.write_text("\n".join(lines) + "\n")


def test_resolve_bsdf_color_diffuse_reflectance_rgb(tmp_path: Path) -> None:
    """A roughplastic BSDF with <rgb name="diffuse_reflectance"> now resolves."""
    xml = _write_scene_xml(
        tmp_path,
        """
        <bsdf type="twosided" id="Wood">
            <bsdf type="roughplastic">
                <rgb name="diffuse_reflectance" value="0.2, 0.6, 0.4"/>
            </bsdf>
        </bsdf>
        """,
    )
    mats = parse_scene_materials(xml)
    assert mats["Wood"] == pytest.approx((0.2, 0.6, 0.4), abs=1e-6)


def test_resolve_bsdf_color_diffuse_reflectance_texture(tmp_path: Path) -> None:
    """A roughplastic BSDF with <texture name="diffuse_reflectance"> uses the texture mean."""
    tex_path = tmp_path / "wood.png"
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    arr[..., 0] = 160
    arr[..., 1] = 96
    arr[..., 2] = 32
    imageio.imwrite(str(tex_path), arr)
    xml = _write_scene_xml(
        tmp_path,
        """
        <bsdf type="twosided" id="WoodFloor">
            <bsdf type="roughplastic">
                <texture name="diffuse_reflectance" type="bitmap">
                    <string name="filename" value="wood.png"/>
                </texture>
            </bsdf>
        </bsdf>
        """,
    )
    mats = parse_scene_materials(xml)
    r, g, b = mats["WoodFloor"]
    assert r == pytest.approx(160 / 255, abs=0.02)
    assert g == pytest.approx(96 / 255, abs=0.02)
    assert b == pytest.approx(32 / 255, abs=0.02)


def test_parse_scene_material_textures_returns_texture_path(tmp_path: Path) -> None:
    """parse_scene_material_textures returns the texture path for textured BSDFs."""
    from src.mesh_utils import parse_scene_material_textures

    tex_path = tmp_path / "t.png"
    imageio.imwrite(str(tex_path), np.zeros((4, 4, 3), dtype=np.uint8))
    xml = _write_scene_xml(
        tmp_path,
        """
        <bsdf type="diffuse" id="HasTex">
            <texture name="reflectance" type="bitmap">
                <string name="filename" value="t.png"/>
            </texture>
        </bsdf>
        <bsdf type="diffuse" id="NoTex">
            <rgb name="reflectance" value="0.5, 0.5, 0.5"/>
        </bsdf>
        """,
    )
    out = parse_scene_material_textures(xml)
    assert out["HasTex"].resolve() == tex_path.resolve()
    assert "NoTex" not in out


def test_bary_grid_vertices_count() -> None:
    from src.mesh_utils import _bary_grid_vertices

    for n in (1, 2, 3, 5, 10):
        v = _bary_grid_vertices(n)
        assert v.shape == ((n + 1) * (n + 2) // 2, 3)
        # Each row sums to 1.
        assert np.allclose(v.sum(axis=1), 1.0)
        # All weights in [0, 1].
        assert (v >= 0).all() and (v <= 1).all()


def test_bary_grid_faces_count() -> None:
    from src.mesh_utils import _bary_grid_faces

    for n in (1, 2, 3, 4):
        f = _bary_grid_faces(n)
        assert f.shape == (n * n, 3)


def test_tessellate_face_grid_subdivides_large_triangle() -> None:
    from src.mesh_utils import _tessellate_face_grid

    # A 1m x 1m right triangle; longest edge = sqrt(2) ~= 1.414.
    corners = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64,
    )
    uvs = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    pos, out_uvs, faces, n = _tessellate_face_grid(corners, uvs, target_spacing=0.10)
    # max_edge / spacing = 1.414 / 0.10 -> ceil = 15.
    assert n == 15
    assert pos.shape == ((n + 1) * (n + 2) // 2, 3)
    assert out_uvs.shape == ((n + 1) * (n + 2) // 2, 2)
    assert faces.shape == (n * n, 3)


def test_tessellate_face_grid_passes_through_small_triangle() -> None:
    from src.mesh_utils import _tessellate_face_grid

    corners = np.array(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]], dtype=np.float64,
    )
    uvs = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    pos, out_uvs, faces, n = _tessellate_face_grid(corners, uvs, target_spacing=0.10)
    assert n == 1
    assert pos.shape == (3, 3)
    assert faces.shape == (1, 3)


def test_bilinear_sample_corners_and_midpoints() -> None:
    """Sample at known UVs on a 2x2 checkerboard and check the right pixel wins."""
    from src.mesh_utils import _bilinear_sample

    # 2x2 image, rows go top->bottom:
    # row 0: [black, white]   <-- image row 0 (top, v=1)
    # row 1: [white, black]   <-- image row 1 (bottom, v=0)
    img = np.array(
        [
            [[0, 0, 0], [255, 255, 255]],
            [[255, 255, 255], [0, 0, 0]],
        ],
        dtype=np.uint8,
    )
    # Use UVs at the actual pixel centers: top-left of image is v=1, u=0.
    uv = np.array(
        [
            [0.0, 1.0],  # top-left  -> black
            [1.0, 1.0],  # top-right -> white
            [0.0, 0.0],  # bot-left  -> white
            [1.0, 0.0],  # bot-right -> black
        ],
        dtype=np.float64,
    )
    out = _bilinear_sample(img, uv)
    assert out[0, 0] == 0 and out[1, 0] == 255
    assert out[2, 0] == 255 and out[3, 0] == 0


def test_export_colored_mesh_ply_tessellated_bakes_texture(tmp_path: Path) -> None:
    """A single triangle with a half-red / half-blue texture, tessellated, has
    spatially-varying colors after the bake."""
    from src.mesh_utils import export_colored_mesh_ply

    # 32x32 texture: top half red, bottom half blue (in image coords; with the
    # v-flip the bottom of the OBJ UV space (v=0) maps to the red region).
    tex_arr = np.zeros((32, 32, 3), dtype=np.uint8)
    tex_arr[:16, :] = (255, 0, 0)   # top of image = OBJ v=1 region
    tex_arr[16:, :] = (0, 0, 255)   # bottom of image = OBJ v=0 region
    tex_file = tmp_path / "halves.png"
    imageio.imwrite(str(tex_file), tex_arr)

    obj_dir = tmp_path / "models"
    obj_dir.mkdir()
    # 1m-wide right triangle; UVs span the unit square.
    _write_obj_with_uv(
        obj_dir / "tri.obj",
        verts=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        uvs=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        faces=[(1, 2, 3)],
    )
    xml = _write_scene_xml(
        tmp_path,
        """
        <bsdf type="diffuse" id="Halves">
            <texture name="reflectance" type="bitmap">
                <string name="filename" value="halves.png"/>
            </texture>
        </bsdf>
        <shape type="obj">
            <string name="filename" value="models/tri.obj"/>
            <ref id="Halves"/>
        </shape>
        """,
    )
    ply = tmp_path / "scene.ply"
    result = export_colored_mesh_ply(xml, ply, tessellate_spacing=0.10)
    assert result is not None
    # Source had 3 verts; tessellation must have added many more.
    assert result.total_verts > 50
    # The summary records this shape as tessellated.
    modes = {s["mode"] for s in result.per_shape}
    assert modes == {"tessellated"}

    loaded = trimesh.load(str(ply), process=False, force="mesh")
    assert isinstance(loaded, trimesh.Trimesh)
    verts = np.asarray(loaded.vertices)
    vc = np.asarray(loaded.visual.vertex_colors)
    # Vertices in the "low v" half (UV v < 0.5) sampled blue in image; v-flip
    # means image row near bottom (= blue).
    # The actual UV at each vertex is bary-interp of (0,0), (1,0), (0,1) by
    # (i/N, j/N, k/N); UV v = i/N (since only uv2's v is 1). So vertices with
    # large y (close to v=1 corner) sample red, small y sample blue.
    big_y = verts[:, 1] > 0.7
    small_y = verts[:, 1] < 0.3
    # In the red region: R should dominate B (and vice versa). Some interpolation
    # near the half-line is unavoidable so we only assert the extreme regions.
    assert (vc[big_y, 0] > vc[big_y, 2]).all(), "verts near v=1 should sample red"
    assert (vc[small_y, 2] > vc[small_y, 0]).all(), "verts near v=0 should sample blue"


def test_export_colored_mesh_ply_mixed_tessellated_and_flat(tmp_path: Path) -> None:
    """A textured shape gets tessellated; a flat-rgb shape does not."""
    from src.mesh_utils import export_colored_mesh_ply

    tex_file = tmp_path / "tex.png"
    imageio.imwrite(str(tex_file), np.full((8, 8, 3), 128, dtype=np.uint8))

    obj_dir = tmp_path / "models"
    obj_dir.mkdir()
    _write_obj_with_uv(
        obj_dir / "textured.obj",
        verts=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        uvs=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        faces=[(1, 2, 3)],
    )
    (obj_dir / "flat.obj").write_text(
        "v 0 0 0\nv 0.01 0 0\nv 0 0.01 0\nf 1 2 3\n"
    )
    xml = _write_scene_xml(
        tmp_path,
        """
        <bsdf type="diffuse" id="TexBSDF">
            <texture name="reflectance" type="bitmap">
                <string name="filename" value="tex.png"/>
            </texture>
        </bsdf>
        <bsdf type="diffuse" id="FlatBSDF">
            <rgb name="reflectance" value="0.5, 0.5, 0.5"/>
        </bsdf>
        <shape type="obj">
            <string name="filename" value="models/textured.obj"/>
            <ref id="TexBSDF"/>
        </shape>
        <shape type="obj">
            <string name="filename" value="models/flat.obj"/>
            <ref id="FlatBSDF"/>
        </shape>
        """,
    )
    ply = tmp_path / "scene.ply"
    result = export_colored_mesh_ply(xml, ply, tessellate_spacing=0.10)
    assert result is not None
    modes = {s["shape"]: s["mode"] for s in result.per_shape}
    assert modes["textured.obj"] == "tessellated"
    assert modes["flat.obj"] == "flat"
    # Textured shape grew, flat shape stayed at 3 verts.
    by_shape = {s["shape"]: s for s in result.per_shape}
    assert by_shape["textured.obj"]["out_verts"] > 3
    assert by_shape["flat.obj"]["out_verts"] == 3


def test_export_colored_mesh_ply_tessellation_disabled(tmp_path: Path) -> None:
    """tessellate_spacing=None preserves the legacy single-color behavior."""
    from src.mesh_utils import export_colored_mesh_ply

    tex_file = tmp_path / "t.png"
    imageio.imwrite(str(tex_file), np.full((4, 4, 3), 200, dtype=np.uint8))
    obj_dir = tmp_path / "models"
    obj_dir.mkdir()
    _write_obj_with_uv(
        obj_dir / "big.obj",
        verts=[(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0)],
        uvs=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        faces=[(1, 2, 3)],
    )
    xml = _write_scene_xml(
        tmp_path,
        """
        <bsdf type="diffuse" id="TexBSDF">
            <texture name="reflectance" type="bitmap">
                <string name="filename" value="t.png"/>
            </texture>
        </bsdf>
        <shape type="obj">
            <string name="filename" value="models/big.obj"/>
            <ref id="TexBSDF"/>
        </shape>
        """,
    )
    ply = tmp_path / "scene.ply"
    result = export_colored_mesh_ply(xml, ply, tessellate_spacing=None)
    assert result is not None
    assert result.total_verts == 3  # no subdivision
    modes = {s["mode"] for s in result.per_shape}
    assert modes == {"flat"}


def test_clip_top_percentile_preserves_vertex_colors() -> None:
    mesh = _tall_dense_mesh("y")
    n = int(len(mesh.vertices))
    colors = np.tile(np.array([200, 100, 50, 255], dtype=np.uint8), (n, 1))
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=colors)
    cropped = clip_top_percentile(mesh, up_axis="y", percentile=50.0)
    vc = np.asarray(cropped.visual.vertex_colors)
    assert vc.shape[0] == len(cropped.vertices)
    # All surviving colors should still be (200, 100, 50, 255).
    assert (vc[:, :3] == [200, 100, 50]).all()
