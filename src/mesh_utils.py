"""Mitsuba scene XML -> colored trimesh + PLY export.

Pure-data utilities: no viser, no Mitsuba. Used by both the renderer
(``render_scene.py``) and the visualizer (``visualize.py``).

Coloring modes per shape:

- **Flat** — one color per shape (the BSDF's ``<rgb>`` value, or the mean of
  its bitmap reflectance). Suitable for surfaces whose BSDF has no spatial
  detail.
- **Tessellate-then-bake** — when ``tessellate_spacing`` is set and the BSDF
  binds a bitmap texture, each face is adaptively subdivided so its world-
  space edges are <= the spacing, and each new vertex's color is the
  bilinear sample of the texture at the interpolated UV. Produces a PLY
  whose vertex colors actually follow the texture, suitable for 3DGS init.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import trimesh
from numpy.typing import NDArray

log = logging.getLogger(__name__)

_DEFAULT_MESH_COLOR: tuple[float, float, float] = (0.7, 0.7, 0.7)
# Reflectance child names we treat as the diffuse color. ``roughplastic`` BSDFs
# expose ``diffuse_reflectance``; plain ``diffuse`` BSDFs use ``reflectance``.
_REFLECTANCE_NAMES: frozenset[str] = frozenset({"reflectance", "diffuse_reflectance"})


def _parse_rgb_string(value: str) -> tuple[float, float, float] | None:
    """Parse a Mitsuba ``rgb`` value like ``"0.5, 0.6, 0.7"``. Returns None on error."""
    try:
        parts = [float(x.strip()) for x in value.replace(",", " ").split()]
    except ValueError:
        return None
    if len(parts) == 1:
        return (parts[0], parts[0], parts[0])
    if len(parts) == 3:
        return (parts[0], parts[1], parts[2])
    return None


def _texture_mean_color(
    tex_path: Path, cache: dict[Path, tuple[float, float, float]]
) -> tuple[float, float, float]:
    """Return the mean RGB (in [0, 1]) of a bitmap texture, cached by path."""
    if tex_path in cache:
        return cache[tex_path]
    if not tex_path.exists():
        log.warning("Texture not found: %s; using default gray", tex_path)
        cache[tex_path] = _DEFAULT_MESH_COLOR
        return _DEFAULT_MESH_COLOR
    try:
        img = np.asarray(imageio.imread(str(tex_path)))
    except Exception as e:
        log.warning("Failed to read texture %s: %s; using default gray", tex_path, e)
        cache[tex_path] = _DEFAULT_MESH_COLOR
        return _DEFAULT_MESH_COLOR
    sample = img[::8, ::8]
    if sample.ndim == 2:
        sample = np.repeat(sample[..., None], 3, axis=-1)
    sample = sample[..., :3].astype(np.float64)
    mean_rgb = sample.mean(axis=(0, 1)) / 255.0
    color = (float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2]))
    cache[tex_path] = color
    return color


def _resolve_bsdf_color(
    bsdf_elem: ET.Element,
    xml_dir: Path,
    cache: dict[Path, tuple[float, float, float]],
) -> tuple[float, float, float] | None:
    """Look inside a <bsdf> element for the first diffuse-reflectance color.

    Recurses through nested ``<bsdf>`` wrappers (``twosided``, ``roughplastic``,
    etc.). Returns the first ``<rgb>`` value whose ``name`` matches
    ``reflectance`` or ``diffuse_reflectance``, or the mean color of the first
    matching bitmap texture. ``None`` if neither is present.
    """
    for child in bsdf_elem.iter("rgb"):
        if child.get("name") in _REFLECTANCE_NAMES:
            value = child.get("value")
            if value is not None:
                parsed = _parse_rgb_string(value)
                if parsed is not None:
                    return parsed
    for tex in bsdf_elem.iter("texture"):
        if tex.get("name") not in _REFLECTANCE_NAMES or tex.get("type") != "bitmap":
            continue
        for s in tex:
            if s.tag == "string" and s.get("name") == "filename":
                filename = s.get("value")
                if filename is None:
                    continue
                return _texture_mean_color((xml_dir / filename).resolve(), cache)
    return None


def _resolve_bsdf_texture_path(
    bsdf_elem: ET.Element, xml_dir: Path
) -> Path | None:
    """Return the absolute path to the BSDF's bitmap reflectance texture, or None.

    Same name-matching rules as ``_resolve_bsdf_color``: accepts both
    ``reflectance`` and ``diffuse_reflectance``. Bump maps and other non-color
    textures are ignored.
    """
    for tex in bsdf_elem.iter("texture"):
        if tex.get("name") not in _REFLECTANCE_NAMES or tex.get("type") != "bitmap":
            continue
        for s in tex:
            if s.tag == "string" and s.get("name") == "filename":
                filename = s.get("value")
                if filename is not None:
                    return (xml_dir / filename).resolve()
    return None


def parse_scene_materials(
    xml_path: Path,
) -> dict[str, tuple[float, float, float]]:
    """Return ``{bsdf_id: (r, g, b)}`` for every top-level ``<bsdf id="...">``.

    Color is the first ``<rgb name="reflectance"|"diffuse_reflectance">`` found
    anywhere inside the BSDF, or the mean RGB of the first matching bitmap
    texture. Falls back to a neutral light gray when neither is present.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    out: dict[str, tuple[float, float, float]] = {}
    tex_cache: dict[Path, tuple[float, float, float]] = {}
    for elem in root:
        if elem.tag != "bsdf":
            continue
        bsdf_id = elem.get("id")
        if bsdf_id is None:
            continue
        color = _resolve_bsdf_color(elem, xml_path.parent, tex_cache)
        out[bsdf_id] = color if color is not None else _DEFAULT_MESH_COLOR
    return out


def parse_scene_material_textures(xml_path: Path) -> dict[str, Path]:
    """Return ``{bsdf_id: absolute_texture_path}`` for BSDFs with a bitmap reflectance.

    BSDFs whose reflectance is a plain ``<rgb>`` (or absent entirely) do not
    appear in the returned dict.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    out: dict[str, Path] = {}
    for elem in root:
        if elem.tag != "bsdf":
            continue
        bsdf_id = elem.get("id")
        if bsdf_id is None:
            continue
        tex_path = _resolve_bsdf_texture_path(elem, xml_path.parent)
        if tex_path is not None:
            out[bsdf_id] = tex_path
    return out


@dataclass(slots=True, frozen=True)
class ShapeBinding:
    obj_path: Path
    bsdf_id: str | None
    transform: NDArray[np.float64] | None  # 4x4, or None if identity
    shape_id: str | None = None            # value of <shape id="...">, if any


def _parse_transform(elem: ET.Element) -> NDArray[np.float64] | None:
    """Parse a Mitsuba ``<transform>`` element. Only ``<matrix>`` is supported."""
    for child in elem:
        if child.tag == "matrix":
            value = child.get("value")
            if value is None:
                return None
            try:
                vals = [float(x) for x in value.replace(",", " ").split()]
            except ValueError:
                return None
            if len(vals) != 16:
                return None
            mat = np.asarray(vals, dtype=np.float64).reshape(4, 4)
            if np.allclose(mat, np.eye(4), atol=1e-6):
                return None
            return mat
    return None


def parse_shape_bindings(xml_path: Path) -> list[ShapeBinding]:
    """Walk every ``<shape type="obj">`` and return its OBJ path + BSDF ref + transform."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bindings: list[ShapeBinding] = []
    xml_dir = xml_path.parent
    for shape in root.iter("shape"):
        if shape.get("type") != "obj":
            continue
        filename: str | None = None
        bsdf_id: str | None = None
        transform: NDArray[np.float64] | None = None
        for child in shape:
            if child.tag == "string" and child.get("name") == "filename":
                filename = child.get("value")
            elif child.tag == "ref":
                ref_id = child.get("id")
                if ref_id is not None:
                    bsdf_id = ref_id
            elif child.tag == "transform":
                transform = _parse_transform(child)
        if filename is None:
            continue
        bindings.append(
            ShapeBinding(
                obj_path=(xml_dir / filename).resolve(),
                bsdf_id=bsdf_id,
                transform=transform,
                shape_id=shape.get("id"),
            )
        )
    return bindings


def find_obj_paths(scene_xml: Path) -> list[Path]:
    """Return absolute paths to every ``<shape type="obj">`` entry."""
    return [b.obj_path for b in parse_shape_bindings(scene_xml)]


def clip_top_percentile(
    mesh: trimesh.Trimesh, up_axis: str, percentile: float
) -> trimesh.Trimesh:
    """Drop faces whose vertices sit above the ``percentile``-th height value.

    Used to crop the ceiling out of an indoor scene so frustum positions are
    visible from above. ``percentile >= 100`` returns the input mesh unchanged.
    Vertex colors are preserved.
    """
    if percentile >= 100.0:
        return mesh
    up_idx = {"x": 0, "y": 1, "z": 2}[up_axis]
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if verts.size == 0:
        return mesh
    cutoff = float(np.percentile(verts[:, up_idx], percentile))
    faces = np.asarray(mesh.faces, dtype=np.int64)
    keep_vert = verts[:, up_idx] <= cutoff
    keep_face = keep_vert[faces].all(axis=1)
    cropped = mesh.copy()
    cropped.update_faces(keep_face)
    cropped.remove_unreferenced_vertices()
    log.info(
        "Ceiling crop: kept %d/%d faces (cutoff %.3fm along '%s')",
        int(keep_face.sum()),
        int(keep_face.size),
        cutoff,
        up_axis,
    )
    return cropped


# ---- Texture image cache + bilinear sampling --------------------------------


def _load_texture_rgb(
    path: Path, cache: dict[Path, NDArray[np.uint8] | None]
) -> NDArray[np.uint8] | None:
    """Load a bitmap as an HxWx3 uint8 array. Cached by path. None on failure."""
    if path in cache:
        return cache[path]
    if not path.exists():
        log.warning("Texture not found: %s", path)
        cache[path] = None
        return None
    try:
        img = np.asarray(imageio.imread(str(path)))
    except Exception as e:
        log.warning("Failed to read texture %s: %s", path, e)
        cache[path] = None
        return None
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    img = np.ascontiguousarray(img[..., :3])
    cache[path] = img
    return img


def _bilinear_sample(
    img: NDArray[np.uint8], uv: NDArray[np.float64]
) -> NDArray[np.uint8]:
    """Vectorized bilinear sample at N (u, v) coords. Returns Nx3 uint8.

    UVs follow the OBJ convention (origin at bottom-left). The v axis is
    flipped to match image rows that go top-to-bottom. Values inside [0, 1]
    are sampled as-is; values outside are wrapped via ``u - floor(u)`` so
    tiled textures repeat without collapsing the seam at exact ``u = 1``.
    """
    h, w, _ = img.shape
    if uv.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    raw_u = uv[:, 0]
    raw_v = uv[:, 1]
    u = np.where(
        (raw_u >= 0.0) & (raw_u <= 1.0), raw_u, raw_u - np.floor(raw_u)
    )
    v = np.where(
        (raw_v >= 0.0) & (raw_v <= 1.0), raw_v, raw_v - np.floor(raw_v)
    )
    y = (1.0 - v) * (h - 1)
    x = u * (w - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    arr = img.astype(np.float64)
    c00 = arr[y0, x0]
    c01 = arr[y0, x1]
    c10 = arr[y1, x0]
    c11 = arr[y1, x1]
    top = c00 * (1.0 - fx) + c01 * fx
    bot = c10 * (1.0 - fx) + c11 * fx
    out = top * (1.0 - fy) + bot * fy
    return np.asarray(np.clip(out, 0, 255), dtype=np.uint8)


# ---- Tessellate + bake ------------------------------------------------------


def _bary_grid_vertices(n: int) -> NDArray[np.float64]:
    """Return the K x 3 barycentric weights for an order-N triangle subdivision.

    Vertex order: row-major in ``i`` (the v0 weight). Row ``i`` contains
    ``N + 1 - i`` vertices with ``j`` ranging over ``0..N - i``; the v2 weight
    is ``k = N - i - j``. K = (N + 1) * (N + 2) / 2.
    """
    rows: list[tuple[float, float, float]] = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            rows.append((i / n, j / n, k / n))
    return np.asarray(rows, dtype=np.float64)


def _bary_grid_faces(n: int) -> NDArray[np.int64]:
    """Return the M x 3 face index list for the barycentric grid of order N.

    Produces ``N**2`` small triangles (``N*(N+1)/2`` upward, ``(N-1)*N/2``
    downward), preserving the original face's CCW winding by ordering vertices
    consistently with the (v0, v1, v2) layout used in ``_bary_grid_vertices``.
    """
    def vid(i: int, j: int) -> int:
        # Row i starts at offset i*(N+1) - i*(i-1)/2.
        return i * (n + 1) - (i * (i - 1)) // 2 + j

    faces: list[tuple[int, int, int]] = []
    for i in range(n):
        for j in range(n - i):
            a = vid(i, j)
            b = vid(i, j + 1)
            c = vid(i + 1, j)
            # Upward: from corner near "high k" toward "high j" then "high i".
            # Match the original (v0, v1, v2) winding (v0=high i, v1=high j,
            # v2=high k): a (near v2) -> c (toward v0) -> b (toward v1).
            faces.append((a, c, b))
            if j + 1 < n - i:
                d = vid(i + 1, j + 1)
                # Downward: b -> c -> d in the same winding.
                faces.append((b, c, d))
    if not faces:
        return np.zeros((0, 3), dtype=np.int64)
    return np.asarray(faces, dtype=np.int64)


def _tessellate_face_grid(
    corners: NDArray[np.float64],
    corner_uvs: NDArray[np.float64],
    target_spacing: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64], int]:
    """Subdivide a single triangle into a barycentric grid.

    Args:
        corners: 3x3 array of (v0, v1, v2) world positions.
        corner_uvs: 3x2 array of (uv0, uv1, uv2).
        target_spacing: max allowed edge length (world units).

    Returns:
        positions: Kx3, uvs: Kx2, faces: Mx3, n (the chosen subdivision order).
    """
    edges = np.array(
        [
            np.linalg.norm(corners[1] - corners[0]),
            np.linalg.norm(corners[2] - corners[1]),
            np.linalg.norm(corners[0] - corners[2]),
        ]
    )
    n = max(1, int(np.ceil(float(edges.max()) / target_spacing)))
    bary = _bary_grid_vertices(n)
    pos = bary @ corners
    uvs = bary @ corner_uvs
    faces = _bary_grid_faces(n)
    return pos, uvs, faces, n


def _tessellate_shape(
    positions: NDArray[np.float64],
    faces: NDArray[np.int64],
    target_spacing: float,
    *,
    uvs: NDArray[np.float64] | None = None,
    texture: NDArray[np.uint8] | None = None,
    flat_color: NDArray[np.uint8] | None = None,
) -> trimesh.Trimesh:
    """Adaptive subdivision + per-vertex color assignment.

    Per face, choose ``n = max(1, ceil(max_edge / target_spacing))`` and
    subdivide into a barycentric grid. Faces are processed in batches grouped
    by ``n`` so the inner loop runs at most ``max(n)`` times.

    Color source:
      - If ``texture`` and ``uvs`` are both provided, each new vertex's color
        is a bilinear sample of ``texture`` at its interpolated UV.
      - Otherwise ``flat_color`` (uint8 shape ``(3,)``) is broadcast to every
        new vertex. Used for flat-RGB BSDFs and for textured BSDFs whose OBJ
        has no usable UV data.
    """
    use_texture = texture is not None and uvs is not None
    if not use_texture and flat_color is None:
        raise ValueError(
            "_tessellate_shape requires either (texture + uvs) or flat_color"
        )
    if faces.size == 0:
        return trimesh.Trimesh(
            vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64),
            process=False,
        )

    corners = positions[faces]      # F x 3 x 3
    edges = np.stack(
        [
            np.linalg.norm(corners[:, 1] - corners[:, 0], axis=-1),
            np.linalg.norm(corners[:, 2] - corners[:, 1], axis=-1),
            np.linalg.norm(corners[:, 0] - corners[:, 2], axis=-1),
        ],
        axis=-1,
    )
    max_edge = edges.max(axis=-1)
    n_per_face = np.maximum(1, np.ceil(max_edge / target_spacing).astype(np.int64))

    corner_uvs: NDArray[np.float64] | None = None  # F x 3 x 2
    if use_texture:
        assert uvs is not None  # guaranteed by use_texture
        corner_uvs = uvs[faces]

    out_pos: list[NDArray[np.float64]] = []
    out_col: list[NDArray[np.uint8]] = []
    out_faces: list[NDArray[np.int64]] = []
    vertex_offset = 0

    unique_ns = sorted({int(x) for x in n_per_face.tolist()})
    for n in unique_ns:
        mask = n_per_face == n
        f_n = int(mask.sum())
        if f_n == 0:
            continue
        bary = _bary_grid_vertices(n)
        grid_faces = _bary_grid_faces(n)
        k = bary.shape[0]
        group_corners = corners[mask]                            # f_n x 3 x 3
        pos = np.einsum("kb,fbc->fkc", bary, group_corners)      # f_n x k x 3
        pos_flat = pos.reshape(-1, 3)

        if use_texture:
            assert corner_uvs is not None and texture is not None
            group_uvs = corner_uvs[mask]                         # f_n x 3 x 2
            uv = np.einsum("kb,fbc->fkc", bary, group_uvs)       # f_n x k x 2
            uv_flat = uv.reshape(-1, 2)
            col_flat = _bilinear_sample(texture, uv_flat)        # (f_n * k) x 3
        else:
            assert flat_color is not None
            col_flat = np.broadcast_to(
                flat_color, (pos_flat.shape[0], 3),
            ).copy()

        repeated = np.tile(grid_faces[None, :, :], (f_n, 1, 1))
        offsets = (np.arange(f_n) * k)[:, None, None]
        repeated = repeated + offsets
        faces_local = repeated.reshape(-1, 3)

        out_pos.append(pos_flat)
        out_col.append(col_flat)
        out_faces.append(faces_local + vertex_offset)
        vertex_offset += pos_flat.shape[0]

    new_positions = np.concatenate(out_pos, axis=0)
    new_colors = np.concatenate(out_col, axis=0)
    new_faces = np.concatenate(out_faces, axis=0)
    rgba = np.concatenate(
        [new_colors, np.full((new_colors.shape[0], 1), 255, dtype=np.uint8)],
        axis=-1,
    )
    mesh = trimesh.Trimesh(vertices=new_positions, faces=new_faces, process=False)
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=rgba)
    return mesh


def _rgb_to_uint8(color: tuple[float, float, float]) -> NDArray[np.uint8]:
    """Convert linear [0, 1] RGB to uint8 (3,)."""
    return np.array(
        [int(round(np.clip(c, 0.0, 1.0) * 255)) for c in color], dtype=np.uint8,
    )


# ---- Top-level loaders ------------------------------------------------------


@dataclass(slots=True)
class MeshExportResult:
    """Result of :func:`export_colored_mesh_ply`: where + what was written."""
    path: Path
    total_verts: int
    total_faces: int
    per_shape: list[dict[str, Any]] = field(default_factory=list)


def _decimate_if_dense(
    mesh: trimesh.Trimesh,
    label: str,
    threshold: int,
) -> trimesh.Trimesh:
    """Best-effort quadric decimation when mesh vert count exceeds threshold."""
    if len(mesh.vertices) <= threshold:
        return mesh
    target_faces = max(1, threshold * 2)
    try:
        decimated = mesh.simplify_quadric_decimation(face_count=target_faces)
    except Exception as e:
        log.warning(
            "Decimation failed for %s (%d verts): %s; keeping original",
            label, len(mesh.vertices), e,
        )
        return mesh
    if not isinstance(decimated, trimesh.Trimesh):
        log.warning("Decimation of %s returned non-mesh; keeping original", label)
        return mesh
    log.info(
        "Decimated %s: %d -> %d verts, %d -> %d faces",
        label, len(mesh.vertices), len(decimated.vertices),
        len(mesh.faces), len(decimated.faces),
    )
    return decimated


def _apply_flat_color(
    mesh: trimesh.Trimesh, color_rgb: tuple[float, float, float]
) -> trimesh.Trimesh:
    """Paint every vertex of ``mesh`` with a single RGBA color."""
    rgba = np.array(
        [
            int(round(np.clip(color_rgb[0], 0.0, 1.0) * 255)),
            int(round(np.clip(color_rgb[1], 0.0, 1.0) * 255)),
            int(round(np.clip(color_rgb[2], 0.0, 1.0) * 255)),
            255,
        ],
        dtype=np.uint8,
    )
    n_verts = int(len(mesh.vertices))
    vertex_colors = np.broadcast_to(rgba, (n_verts, 4)).copy()
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh, vertex_colors=vertex_colors,
    )
    return mesh


def load_scene_mesh(
    scene_xml: Path,
    *,
    tessellate_spacing: float | None = None,
    simplify_dense_meshes: bool = False,
    simplify_vertex_threshold: int = 100_000,
) -> trimesh.Trimesh | None:
    """Load + color + concatenate every OBJ shape referenced from the scene XML.

    Args:
        scene_xml: path to ``scene_v3.xml`` (or similar Mitsuba 3 scene file).
        tessellate_spacing: when set, every shape's faces are adaptively
            subdivided so world-space edges don't exceed this spacing (meters).
            Shapes whose BSDF binds a bitmap texture (and whose OBJ has valid
            UVs) get per-vertex colors via bilinear texture sampling; all
            other shapes get the BSDF's solid color broadcast to every new
            vertex. This produces a roughly uniform face size across the
            whole scene (useful for 3DGS init). ``None`` (default) disables
            tessellation entirely and keeps the original geometry with a
            single color per shape.
        simplify_dense_meshes: when True, decimate any individual mesh with
            more than ``simplify_vertex_threshold`` vertices via
            ``trimesh.simplify_quadric_decimation``.
        simplify_vertex_threshold: vertex count above which decimation kicks
            in (when ``simplify_dense_meshes`` is True).

    The per-shape coloring choice is recorded on
    ``mesh.metadata["mesh_export_summary"]`` as a list of dicts.
    """
    bindings = parse_shape_bindings(scene_xml)
    if not bindings:
        log.warning("No .obj shapes found in %s", scene_xml)
        return None
    materials = parse_scene_materials(scene_xml)
    textures = parse_scene_material_textures(scene_xml)
    tex_img_cache: dict[Path, NDArray[np.uint8] | None] = {}

    parts: list[trimesh.Trimesh] = []
    summary: list[dict[str, Any]] = []
    do_tessellate = (
        tessellate_spacing is not None and tessellate_spacing > 0.0
    )
    spacing = float(tessellate_spacing) if tessellate_spacing is not None else 0.0

    for b in bindings:
        if not b.obj_path.exists():
            log.warning("OBJ not found: %s", b.obj_path)
            continue
        loaded = trimesh.load(b.obj_path, process=False, force="mesh")
        if not isinstance(loaded, trimesh.Trimesh):
            continue
        mesh: trimesh.Trimesh = loaded
        if b.transform is not None:
            mesh.apply_transform(b.transform)
        if simplify_dense_meshes:
            mesh = _decimate_if_dense(
                mesh, b.obj_path.name, simplify_vertex_threshold,
            )

        tex_path = textures.get(b.bsdf_id) if b.bsdf_id is not None else None
        tex_img: NDArray[np.uint8] | None = None
        if do_tessellate and tex_path is not None:
            tex_img = _load_texture_rgb(tex_path, tex_img_cache)

        uv_raw = getattr(mesh.visual, "uv", None)
        uv_arr = np.asarray(uv_raw, dtype=np.float64) if uv_raw is not None else None
        has_valid_uv = (
            uv_arr is not None
            and uv_arr.ndim == 2
            and uv_arr.shape == (len(mesh.vertices), 2)
        )

        color_rgb = (
            materials.get(b.bsdf_id, _DEFAULT_MESH_COLOR)
            if b.bsdf_id is not None
            else _DEFAULT_MESH_COLOR
        )

        if do_tessellate:
            positions = np.asarray(mesh.vertices, dtype=np.float64)
            faces = np.asarray(mesh.faces, dtype=np.int64)
            use_texture = tex_img is not None and has_valid_uv

            # Skip tessellation entirely when every face is already fine enough.
            # This avoids exploding already-dense meshes (e.g. the bedroom
            # carpet with ~3mm edges) into 3x as many duplicated corner verts.
            corners_for_check = positions[faces]
            edges_for_check = np.stack(
                [
                    np.linalg.norm(
                        corners_for_check[:, 1] - corners_for_check[:, 0], axis=-1,
                    ),
                    np.linalg.norm(
                        corners_for_check[:, 2] - corners_for_check[:, 1], axis=-1,
                    ),
                    np.linalg.norm(
                        corners_for_check[:, 0] - corners_for_check[:, 2], axis=-1,
                    ),
                ],
                axis=-1,
            )
            needs_subdivision = bool((edges_for_check > spacing).any())

            if not needs_subdivision:
                if use_texture:
                    assert uv_arr is not None and tex_img is not None
                    colors_rgb = _bilinear_sample(tex_img, uv_arr)
                    rgba = np.concatenate(
                        [
                            colors_rgb,
                            np.full((colors_rgb.shape[0], 1), 255, dtype=np.uint8),
                        ],
                        axis=-1,
                    )
                    mesh.visual = trimesh.visual.ColorVisuals(
                        mesh=mesh, vertex_colors=rgba,
                    )
                    parts.append(mesh)
                    mode = "fine_texture"
                else:
                    flat = _apply_flat_color(mesh, color_rgb)
                    parts.append(flat)
                    mode = "fine_flat"
                summary.append({
                    "shape": b.obj_path.name,
                    "bsdf": b.bsdf_id,
                    "mode": mode,
                    "texture": tex_path.name if tex_path is not None else None,
                    "in_verts": int(positions.shape[0]),
                    "in_faces": int(faces.shape[0]),
                    "out_verts": int(positions.shape[0]),
                    "out_faces": int(faces.shape[0]),
                })
                continue

            if use_texture:
                assert uv_arr is not None  # for mypy
                tessellated = _tessellate_shape(
                    positions, faces, spacing,
                    uvs=uv_arr, texture=tex_img,
                )
                mode = "tessellated_texture"
            else:
                tessellated = _tessellate_shape(
                    positions, faces, spacing,
                    flat_color=_rgb_to_uint8(color_rgb),
                )
                mode = "tessellated_flat"
            # Tessellation emits one corner-vertex copy per face; adjacent
            # faces share their edge verts at the same world position with
            # identical colors. Dedup within this shape only -- merging
            # across shapes would risk collapsing genuine seams.
            tessellated.merge_vertices()
            parts.append(tessellated)
            summary.append({
                "shape": b.obj_path.name,
                "bsdf": b.bsdf_id,
                "mode": mode,
                "texture": tex_path.name if tex_path is not None else None,
                "in_verts": int(positions.shape[0]),
                "in_faces": int(faces.shape[0]),
                "out_verts": int(len(tessellated.vertices)),
                "out_faces": int(len(tessellated.faces)),
            })
            continue

        flat = _apply_flat_color(mesh, color_rgb)
        parts.append(flat)
        summary.append({
            "shape": b.obj_path.name,
            "bsdf": b.bsdf_id,
            "mode": "flat",
            "texture": tex_path.name if tex_path is not None else None,
            "in_verts": int(len(flat.vertices)),
            "in_faces": int(len(flat.faces)),
            "out_verts": int(len(flat.vertices)),
            "out_faces": int(len(flat.faces)),
        })

    if not parts:
        return None
    concatenated: trimesh.Trimesh = trimesh.util.concatenate(parts)
    # Drop degenerate triangles: zero-area or repeated-index faces. These can
    # come from sliver source triangles whose corners collapsed after
    # ``merge_vertices``, or from malformed OBJ input. MeshLab counts them as
    # errors and 3DGS bootstrappers may divide-by-zero on them.
    n_faces_before = int(len(concatenated.faces))
    concatenated.update_faces(concatenated.nondegenerate_faces())
    n_degenerate = n_faces_before - int(len(concatenated.faces))
    if n_degenerate > 0:
        log.info("Removed %d degenerate faces", n_degenerate)
    concatenated.metadata = dict(concatenated.metadata or {})
    concatenated.metadata["mesh_export_summary"] = summary
    concatenated.metadata["mesh_export_degenerate_removed"] = n_degenerate
    return concatenated


def export_colored_mesh_ply(
    scene_xml: Path,
    ply_path: Path,
    *,
    tessellate_spacing: float | None = None,
    simplify_dense_meshes: bool = False,
    simplify_vertex_threshold: int = 100_000,
) -> MeshExportResult | None:
    """Load the colored scene mesh and write it as a PLY (with vertex colors).

    Returns a :class:`MeshExportResult` carrying the absolute PLY path, total
    vertex/face counts, and the per-shape coloring summary, or ``None`` if the
    scene has no OBJ geometry to export.

    See :func:`load_scene_mesh` for the meaning of the tessellation and
    decimation arguments.
    """
    mesh = load_scene_mesh(
        scene_xml,
        tessellate_spacing=tessellate_spacing,
        simplify_dense_meshes=simplify_dense_meshes,
        simplify_vertex_threshold=simplify_vertex_threshold,
    )
    if mesh is None:
        return None
    ply_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(ply_path), file_type="ply")
    summary_obj = mesh.metadata.get("mesh_export_summary", []) if mesh.metadata else []
    summary: list[dict[str, Any]] = (
        list(summary_obj) if isinstance(summary_obj, list) else []
    )
    log.info(
        "Exported colored mesh: %s (%d verts, %d faces)",
        ply_path, len(mesh.vertices), len(mesh.faces),
    )
    return MeshExportResult(
        path=ply_path.resolve(),
        total_verts=int(len(mesh.vertices)),
        total_faces=int(len(mesh.faces)),
        per_shape=summary,
    )
