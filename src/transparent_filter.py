"""Strip transparent shapes from a Mitsuba scene XML.

Detects glass-like BSDFs (``dielectric``, ``thindielectric``, ``roughdielectric``,
``null``) including those wrapped in ``twosided`` / ``bumpmap`` / ``mask``
containers, then drops every ``<shape>`` that references one of them. Writes the
filtered scene to ``work_dir/scene_filtered.xml`` and returns a report listing
what was removed.

Both the Mitsuba and Blender renderers, plus the PLY exporter, load the
filtered XML so transparent geometry never reaches any backend.
"""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

# BSDF types that we treat as transparent. ``mask`` is a wrapper that can hide
# arbitrary BSDFs behind an alpha cutout — we only flag it as transparent if it
# wraps one of the genuinely-transparent inner BSDFs.
_TRANSPARENT_LEAF_TYPES: frozenset[str] = frozenset(
    {"dielectric", "thindielectric", "roughdielectric", "null"}
)
# Types whose <bsdf> children should be recursed into when classifying.
_WRAPPER_TYPES: frozenset[str] = frozenset({"twosided", "bumpmap", "mask", "normalmap"})


@dataclass(frozen=True)
class FilterReport:
    """What ``filter_transparent_scene`` actually removed."""

    dropped_bsdf_ids: tuple[str, ...] = field(default_factory=tuple)
    dropped_shape_ids: tuple[str, ...] = field(default_factory=tuple)
    kept_shape_count: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "dropped_bsdf_ids": list(self.dropped_bsdf_ids),
            "dropped_shape_ids": list(self.dropped_shape_ids),
            "kept_shape_count": self.kept_shape_count,
        }


def _is_transparent_bsdf(bsdf_elem: ET.Element) -> bool:
    """Recurse into wrapper BSDFs and return True if any leaf is transparent."""
    bsdf_type = bsdf_elem.get("type", "")
    if bsdf_type in _TRANSPARENT_LEAF_TYPES:
        return True
    if bsdf_type in _WRAPPER_TYPES:
        for child in bsdf_elem.findall("bsdf"):
            if _is_transparent_bsdf(child):
                return True
    return False


def _collect_transparent_bsdf_ids(root: ET.Element) -> set[str]:
    """All top-level ``<bsdf id=...>`` whose leaf BSDF is transparent."""
    ids: set[str] = set()
    for bsdf in root.findall("bsdf"):
        bsdf_id = bsdf.get("id")
        if bsdf_id is None:
            continue
        if _is_transparent_bsdf(bsdf):
            ids.add(bsdf_id)
    return ids


def _shape_references_bsdf(shape: ET.Element, bsdf_ids: set[str]) -> bool:
    """True if the shape's ``<ref id=...>`` (or nested inline BSDF) matches."""
    for ref in shape.findall("ref"):
        ref_id = ref.get("id")
        if ref_id is not None and ref_id in bsdf_ids:
            return True
    # Inline BSDF inside the shape (no top-level id): recurse into it.
    for inline in shape.findall("bsdf"):
        if _is_transparent_bsdf(inline):
            return True
    return False


def _absolutize_filename_strings(root: ET.Element, source_dir: Path) -> None:
    """Rewrite every ``<string name="filename" value="...">`` to an absolute path.

    Mitsuba resolves these paths relative to the XML file's directory, so a
    filtered XML living outside the original scene directory would fail to
    find textures and OBJs. Walk every such element and make the value
    absolute (relative to ``source_dir``) so the filtered XML is portable.
    """
    for elem in root.iter("string"):
        if elem.get("name") != "filename":
            continue
        value = elem.get("value")
        if value is None:
            continue
        p = Path(value)
        if p.is_absolute():
            continue
        elem.set("value", str((source_dir / p).resolve()))


def filter_transparent_scene(
    scene_xml: Path, work_dir: Path
) -> tuple[Path, FilterReport]:
    """Write a copy of ``scene_xml`` with transparent shapes stripped.

    The output path is ``work_dir / scene_filtered.xml`` (the work_dir is
    created if necessary). Relative texture/OBJ filenames are rewritten to
    absolute paths against ``scene_xml.parent`` so the filtered XML can be
    loaded from any directory. If no transparent shapes are found, the file
    is still rewritten (rather than byte-copied) so the absolute-path
    rewrite still applies — callers can always use the returned path.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = work_dir / "scene_filtered.xml"
    source_dir = scene_xml.parent.resolve()

    tree = ET.parse(scene_xml)
    root = tree.getroot()

    transparent_ids = _collect_transparent_bsdf_ids(root)

    dropped_shape_ids: list[str] = []
    kept_count = 0
    for child in list(root):
        if child.tag != "shape":
            continue
        if _shape_references_bsdf(child, transparent_ids):
            shape_id = child.get("id") or child.get("filename") or "<unknown>"
            dropped_shape_ids.append(shape_id)
            root.remove(child)
        else:
            kept_count += 1

    _absolutize_filename_strings(root, source_dir)

    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path, FilterReport(
        dropped_bsdf_ids=tuple(sorted(transparent_ids)),
        dropped_shape_ids=tuple(dropped_shape_ids),
        kept_shape_count=kept_count,
    )
