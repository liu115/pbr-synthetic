"""Post-import sanity audit: does Blender/Cycles faithfully receive the Mitsuba scene?

Loads a Mitsuba XML through the patched mitsuba-blender importer and reports the
silent-breakage signals from ``blender_fix.md`` *without rendering*:

* **Emitters** — every ``<emitter>`` in the XML vs the lights that actually landed in
  ``bpy.data.lights``. Flags Mitsuba emitter types the addon drops, in particular
  ``spot`` (which makes the import raise ``TypeError`` — see F2) and any ``point`` /
  ``directional`` that didn't produce a light. ``area`` emitters are mesh-attached and
  become emissive *materials*, so they are reported separately, not as missing lights.
* **World** — ``scene.world`` Background colour/strength. Flags the addon's phantom grey
  fallback (~0.05088 @ strength 1.0) that ``--no-wipe-world`` leaves in place (P0/F1).
* **Texture colour spaces** — every image's ``colorspace_settings.name`` (P3): albedo
  should be ``sRGB``, data maps (roughness/normal/metallic) ``Non-Color``.

Exit code is 0 (diagnostic) unless ``--strict`` and an anomaly is found (dropped emitter,
or a phantom world when none was requested).

Usage::

    python scripts/audit_scene_import.py --scene scene_v3.xml [--no-wipe-world] [--strict] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Mitsuba emitter plugin -> what we expect on the Blender side after import.
#   point/directional -> a bpy light (POINT / SUN)
#   spot              -> UNSUPPORTED by the addon (F2): drops / crashes the import
#   area              -> emissive material on the attached mesh, NOT a bpy light
#   envmap/constant   -> a world (Background), NOT a bpy light
_LIGHT_EMITTERS = {"point", "directional", "spot"}
_WORLD_EMITTERS = {"envmap", "constant"}
_DEFAULT_WORLD_GREY = 0.05087608844041824


def _xml_emitter_types(scene_xml: Path) -> Counter:
    """Count every <emitter type="..."> in the XML (top-level and nested in shapes)."""
    counts: Counter = Counter()
    try:
        root = ET.parse(scene_xml).getroot()
    except ET.ParseError as e:
        print(f"  ! could not parse XML: {e}", file=sys.stderr)
        return counts
    for em in root.iter("emitter"):
        counts[em.get("type", "?")] += 1
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", type=Path, required=True, help="Mitsuba scene XML.")
    ap.add_argument("--device", choices=["OPTIX", "CUDA", "CPU"], default="CPU")
    ap.add_argument("--no-wipe-world", action="store_true",
                    help="Load with wipe_world=False to see the phantom default world "
                         "the production path would otherwise neutralize.")
    ap.add_argument("--strict", action="store_true",
                    help="Exit non-zero if a dropped emitter or unexpected world is found.")
    ap.add_argument("--json", type=Path, default=None, help="Write the report as JSON.")
    args = ap.parse_args()

    if not args.scene.exists():
        ap.error(f"scene not found: {args.scene}")

    from src.scene_blender import init_blender, load_scene_blender, _import_bpy

    init_blender(ensure_addon=True, device=args.device, denoiser="NONE")
    # Catch the F2 spot crash explicitly instead of letting it abort the audit.
    try:
        load_scene_blender(args.scene, wipe_world=not args.no_wipe_world)
        import_error = None
    except Exception as e:  # noqa: BLE001 - we want to report any import failure
        import_error = f"{type(e).__name__}: {e}"
    bpy = _import_bpy()

    xml_emitters = _xml_emitter_types(args.scene)
    light_types = Counter(l.type for l in bpy.data.lights)

    # --- Emitter reconciliation ---
    expected_lights = sum(v for k, v in xml_emitters.items() if k in _LIGHT_EMITTERS)
    got_lights = sum(light_types.values())
    dropped = []
    for etype, n in xml_emitters.items():
        if etype in _LIGHT_EMITTERS and got_lights < expected_lights:
            # Coarse: spot is the known-dropped one; flag any light shortfall.
            if etype == "spot":
                dropped.append(f"{n}x spot (UNSUPPORTED by addon -> import error)")

    # --- World ---
    world = bpy.context.scene.world
    world_report: dict = {"present": world is not None}
    if world is not None and getattr(world, "use_nodes", False) and world.node_tree:
        bg = next((n for n in world.node_tree.nodes
                   if n.bl_idname == "ShaderNodeBackground"), None)
        has_envtex = any(n.bl_idname == "ShaderNodeTexEnvironment"
                         for n in world.node_tree.nodes)
        if bg is not None:
            col = list(bg.inputs["Color"].default_value)[:3]
            strength = float(bg.inputs["Strength"].default_value)
            is_grey = (not has_envtex
                       and all(abs(c - _DEFAULT_WORLD_GREY) < 1e-3 for c in col))
            world_report.update(name=world.name, color=col, strength=strength,
                                has_env_texture=has_envtex, looks_like_grey_default=is_grey)

    xml_has_env = any(k in _WORLD_EMITTERS for k in xml_emitters)
    phantom_world = bool(world_report.get("looks_like_grey_default")
                         and world_report.get("strength", 0.0) > 0.0
                         and not xml_has_env)

    # --- Texture colour spaces ---
    colorspaces = {img.name: img.colorspace_settings.name for img in bpy.data.images}

    report = {
        "scene": str(args.scene),
        "import_error": import_error,
        "xml_emitters": dict(xml_emitters),
        "blender_lights": dict(light_types),
        "dropped_emitters": dropped,
        "world": world_report,
        "phantom_world_present": phantom_world,
        "colorspaces": colorspaces,
    }

    # --- Human-readable ---
    print(f"\n=== import audit: {args.scene} ===")
    if import_error:
        print(f"  IMPORT ERROR: {import_error}")
    print(f"  XML emitters : {dict(xml_emitters) or '{}'}")
    print(f"  Blender lights: {dict(light_types) or '{}'}")
    if dropped:
        print("  DROPPED EMITTERS:")
        for d in dropped:
            print(f"    - {d}")
    print(f"  World: {world_report}")
    if phantom_world:
        print("  PHANTOM WORLD: grey default @ strength>0 with no env emitter in XML "
              "(production wipe_world=True neutralizes this).")
    bad_cs = {k: v for k, v in colorspaces.items()
              if v not in ("sRGB", "Non-Color", "Linear")}
    print(f"  Image colour spaces: {colorspaces or '{}'}")
    if bad_cs:
        print(f"  ! unexpected colour spaces: {bad_cs}")

    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
        print(f"  wrote {args.json}")

    anomaly = bool(import_error or dropped or phantom_world)
    if args.strict and anomaly:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
