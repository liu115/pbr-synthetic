"""Blender backend for scene loading, bbox + raycasting.

Mirrors the public surface of ``src/scene_utils.py`` so the CLI orchestrator
can stay nearly identical. Functions live in this module to keep ``bpy``
imports lazy: ``mypy`` and code that only needs other backends never pay
the bpy import cost.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from src.scene_utils import SceneInfo, UpAxis, _AXIS_INDEX, scene_info_from_bbox

if TYPE_CHECKING:  # pragma: no cover - typing-only
    import bpy  # noqa: F401

log = logging.getLogger(__name__)


# Public-ish module-level constants ------------------------------------------

ADDON_MODULE_NAMES: tuple[str, ...] = ("mitsuba_blender", "mitsuba-blender")
"""Possible Blender add-on module identifiers for the Mitsuba importer."""

ADDON_RELEASE_API = (
    "https://api.github.com/repos/mitsuba-renderer/mitsuba-blender/releases/latest"
)
ADDON_CACHE_DIR = Path(
    os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
) / "pbr-capture-synthetic"
ADDON_CACHE_PATH = ADDON_CACHE_DIR / "mitsuba-blender.zip"


# Lazy bpy import -------------------------------------------------------------


def _import_bpy() -> Any:
    try:
        import bpy  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "bpy is not installed. Install Blender as a Python module:\n"
            "    pip install bpy\n"
            f"Original error: {e}"
        ) from e
    return bpy


# Add-on install --------------------------------------------------------------


def _resolve_addon_zip_url() -> str:
    """Query the GitHub API for the latest mitsuba-blender release asset URL."""
    log.info("Querying %s for the latest mitsuba-blender release", ADDON_RELEASE_API)
    req = urllib.request.Request(
        ADDON_RELEASE_API, headers={"Accept": "application/vnd.github+json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    assets = data.get("assets", []) or []
    zip_assets = [a for a in assets if str(a.get("name", "")).endswith(".zip")]
    if not zip_assets:
        raise RuntimeError(
            f"No .zip asset on the latest mitsuba-blender release ({data.get('tag_name')})"
        )
    return str(zip_assets[0]["browser_download_url"])


def _download_addon_zip(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    url = _resolve_addon_zip_url()
    log.info("Downloading mitsuba-blender addon: %s -> %s", url, target)
    with urllib.request.urlopen(url, timeout=120) as resp, target.open("wb") as f:
        f.write(resp.read())
    return target


def _zip_top_dir(zip_path: Path) -> str:
    """Return the single top-level directory inside the addon zip."""
    with zipfile.ZipFile(zip_path) as z:
        roots = sorted({n.split("/", 1)[0] for n in z.namelist() if "/" in n})
    if not roots:
        raise RuntimeError(f"Empty mitsuba-blender zip: {zip_path}")
    for r in roots:
        if "mitsuba" in r.lower():
            return r
    return roots[0]


def _addon_is_enabled(bpy: Any) -> str | None:
    """Return the enabled mitsuba-blender module name, or None if absent."""
    enabled = set(bpy.context.preferences.addons.keys())
    for name in ADDON_MODULE_NAMES:
        if name in enabled:
            return name
    return None


def _addons_dir(bpy: Any) -> Path:
    """Resolve Blender's user-addons directory in a version-portable way."""
    path = bpy.utils.user_resource("SCRIPTS", path="addons", create=True)
    return Path(path)


_XML_TO_PROPS_POLYFILL = (
    "    # patched: mitsuba 3.6+ dropped xml_to_props; fall back to the new parser API.\n"
    "    # The downstream MitsubaSceneProperties keys by prop.id(), so we synthesize\n"
    "    # IDs for objects without one (Scene, Integrator, BSDF etc. typically have empty\n"
    "    # id() in the new parser) to avoid OrderedDict key collisions that drop nodes.\n"
    "    try:\n"
    "        from mitsuba import xml_to_props\n"
    "        raw_props = xml_to_props(filepath)\n"
    "    except ImportError:\n"
    "        import mitsuba as _mi\n"
    "        from mitsuba.parser import ParserConfig as _PC, parse_file as _pf\n"
    "        _state = _pf(_PC(_mi.variant()), str(filepath))\n"
    "        raw_props = []\n"
    "        for _i, _n in enumerate(_state.nodes):\n"
    "            _cls = str(_n.type).rsplit('.', 1)[-1]\n"
    "            if not _n.props.id():\n"
    "                _n.props.set_id(f'__auto_{_i:04d}_{_cls.lower()}')\n"
    "            raw_props.append((_cls, _n.props))\n"
)


def _patch_addon_for_compat(
    addon_dir: Path, *, roughplastic_coat_weight: float = 0.8
) -> None:
    """In-place patches to make mitsuba-blender v0.4.0 load against mitsuba 3.6+.

    Three changes (the upstream addon was written against mitsuba 3.5):

    1. **Version pin** in ``__init__.py``. ``register()`` calls
       ``check_pip_dependencies`` which compares the installed mitsuba version
       against the literal ``DEPS_MITSUBA_VERSION = '3.5.0'``. With a newer
       mitsuba, ``has_valid_dependencies_version`` becomes False, ``io.register``
       is skipped, and the importer operator is never created. Rewrite the
       constant to match the actually-installed mitsuba version.

    2. **``ThreadEnvironment`` compat shim** in ``__init__.py``. ``init_mitsuba``
       does ``from mitsuba import ThreadEnvironment`` — symbol removed in
       mitsuba 3.6+. The addon only catches ``ModuleNotFoundError`` (not
       ``ImportError``), so this aborts the whole register chain. ``thread_env``
       is only used by the Mitsuba *render engine* integration
       (``engine/final.py``); the importer — the only thing we need — doesn't
       touch it. Wrap that line in its own ``try`` so the importer still
       registers.

    3. **``xml_to_props`` polyfill** in ``io/importer/__init__.py``. mitsuba 3.6+
       replaced ``xml_to_props`` with the new ``mitsuba.parser`` module. The
       importer's main entry point does ``from mitsuba import xml_to_props``;
       fall back to ``mitsuba.parser.parse_file`` and reconstruct the
       ``(class_name, Properties)`` list the rest of the addon expects.

    All edits are no-ops on already-patched files (idempotent).
    """
    try:
        import mitsuba  # type: ignore[import-not-found]
    except ImportError:
        log.warning(
            "mitsuba not importable; skipping addon compat patches. "
            "The addon will likely fail to register."
        )
        return
    # mitsuba <3.6 lazy-loads its variant module; reading any attribute
    # (including __version__) before a variant is set raises ImportError.
    try:
        mi_version = mitsuba.__version__
    except ImportError:
        mitsuba.set_variant("scalar_rgb")
        mi_version = mitsuba.__version__

    init_py = addon_dir / "__init__.py"
    importer_init_py = addon_dir / "io" / "importer" / "__init__.py"

    # ----- Patches 1 + 2: top-level __init__.py
    src = init_py.read_text()
    new_src = src
    new_src, n_ver = re.subn(
        r"DEPS_MITSUBA_VERSION\s*=\s*['\"][^'\"]+['\"]",
        f"DEPS_MITSUBA_VERSION = '{mi_version}'",
        new_src,
        count=1,
    )
    thread_env_pattern = (
        r"        from mitsuba import ThreadEnvironment\n"
        r"        bpy\.types\.Scene\.thread_env = ThreadEnvironment\(\)\n"
    )
    thread_env_replacement = (
        "        try:\n"
        "            from mitsuba import ThreadEnvironment\n"
        "            bpy.types.Scene.thread_env = ThreadEnvironment()\n"
        "        except ImportError:\n"
        "            pass  # patched: mitsuba 3.6+ dropped ThreadEnvironment; "
        "only used by the engine, not the importer.\n"
    )
    new_src, n_thr = re.subn(thread_env_pattern, thread_env_replacement, new_src, count=1)
    if new_src != src:
        init_py.write_text(new_src)
    if n_ver:
        log.info("Patched DEPS_MITSUBA_VERSION -> %s", mi_version)
    if n_thr:
        log.info("Patched ThreadEnvironment import to be optional")

    # ----- Patch 3: io/importer/__init__.py — xml_to_props polyfill
    if importer_init_py.exists():
        src2 = importer_init_py.read_text()
        xml_pattern = (
            r"    from mitsuba import xml_to_props\n"
            r"    raw_props = xml_to_props\(filepath\)\n"
        )
        new_src2, n_xml = re.subn(
            xml_pattern, _XML_TO_PROPS_POLYFILL, src2, count=1
        )
        if n_xml and new_src2 != src2:
            importer_init_py.write_text(new_src2)
            log.info("Patched xml_to_props polyfill into %s", importer_init_py)

    # ----- Patches 4-6: bpy 4.x compat (per upstream issue #113 + our own debugging).
    # Detect the running bpy version; the 4.x patches break things on bpy 3.x.
    bpy = _import_bpy()
    bpy_major = int(bpy.app.version[0])
    if bpy_major >= 4:
        _patch_addon_for_bpy4(addon_dir)

    # ----- Patch: roughplastic Coat Weight (works on both 3.x and 4.x).
    # The upstream addon hardcodes Coat Weight = 0.8 in write_mi_roughplastic_bsdf,
    # which makes roughplastic surfaces noticeably brighter than Mitsuba renders.
    # This patch lets callers tune the value (0.0 == Mitsuba-flavoured: no coat).
    _patch_roughplastic_coat(
        addon_dir / "io" / "importer" / "materials.py",
        roughplastic_coat_weight,
    )

    if n_ver == 0 and n_thr == 0:
        log.info("Top-level __init__.py compat patches already applied.")


def _patch_roughplastic_coat(materials_py: Path, coat_weight: float) -> None:
    """Override the addon's hardcoded `Coat Weight = 0.8` for roughplastic/plastic.

    Runs after the bpy 4.x rename patch (if applicable), so on bpy 4.x the line
    reads ``inputs['Coat Weight']`` and on bpy 3.x it still reads
    ``inputs['Clearcoat']``. We rewrite both forms so the patch is bpy-agnostic.

    Idempotent: replaces any literal ``= 0.8`` or ``= 0.0`` on the Coat line
    with the requested value.
    """
    if not materials_py.exists():
        return
    src = materials_py.read_text()
    target = f"{coat_weight:.4g}"
    pattern = re.compile(
        r"(bl_principled\.inputs\['(Coat Weight|Clearcoat)'\]\.default_value\s*=\s*)"
        r"[0-9]+\.?[0-9]*"
    )
    new_src, n = pattern.subn(rf"\g<1>{target}", src)
    if new_src != src:
        materials_py.write_text(new_src)
        log.info("Patched %d Coat Weight site(s) -> %s in materials.py", n, target)


def _patch_addon_for_bpy4(addon_dir: Path) -> None:
    """Extra patches needed when running under bpy >= 4.0.

    The mitsuba-blender v0.4.0 release claims Blender 4.0 support, but the
    coverage is partial. Three call sites still use pre-4.0 APIs and crash
    on Blender 4.x; all are reported in upstream issue
    https://github.com/mitsuba-renderer/mitsuba-blender/issues/113.

    1. **Principled BSDF socket names** (``materials.py``). Blender 4.0
       renamed the sockets:
       ``Clearcoat``→``Coat Weight``, ``Clearcoat Roughness``→``Coat Roughness``,
       ``Specular``→``Specular IOR Level``, ``Specular Tint``→``Specular Tint``
       (kept), ``Sheen Tint``→``Sheen Tint`` (kept). ``write_mi_principled_bsdf``
       already does this conditionally; ``write_mi_roughplastic_bsdf`` and
       ``write_mi_plastic_bsdf`` still hardcode the pre-4.0 names. Rewrite
       those hardcoded literals to the 4.0+ names. (We can't easily inject a
       version conditional via regex, but the addon is already pinned to
       Mitsuba 3.5; the bpy 4.0 names are what we need.)
    2. **``Mesh.calc_normals()`` removed in 4.0** (``shapes.py``). Wrap the
       call in a ``hasattr`` check so it no-ops on 4.x. Auto-smooth normals
       still get computed because the importer also calls
       ``bl_mesh.use_auto_smooth = True``.
    3. **``bmesh.ops.create_uvsphere(diameter=...)``** (``shapes.py``).
       Renamed to ``radius=`` in 4.x. Replace the keyword in the call.

    All edits are idempotent (no-op on already-patched files).
    """
    materials_py = addon_dir / "io" / "importer" / "materials.py"
    shapes_py = addon_dir / "io" / "importer" / "shapes.py"
    renderer_py = addon_dir / "io" / "importer" / "renderer.py"

    if materials_py.exists():
        src = materials_py.read_text()
        new_src = src
        # Order matters: 'Clearcoat Roughness' BEFORE 'Clearcoat' so the
        # latter pattern doesn't accidentally also rewrite the former.
        replacements = [
            ("'Clearcoat Roughness'", "'Coat Roughness'"),
            ("'Clearcoat'", "'Coat Weight'"),
            ("'Specular'", "'Specular IOR Level'"),
        ]
        n_mat = 0
        for old, new in replacements:
            count = new_src.count(old)
            new_src = new_src.replace(old, new)
            n_mat += count
        # In bpy 4.0+ the 'Specular Tint' socket changed from float to color
        # (4-component RGBA). Rewrite hardcoded float assignments to RGBA
        # tuples so they don't raise "sequence expected at dimension 1".
        spec_tint_old = "bl_principled.inputs['Specular Tint'].default_value = 1.0"
        spec_tint_new = "bl_principled.inputs['Specular Tint'].default_value = (1.0, 1.0, 1.0, 1.0)"
        if spec_tint_old in new_src:
            new_src = new_src.replace(spec_tint_old, spec_tint_new)
            n_mat += 1
        # Cycles 4.0 dropped the 'SHARP' microfacet distribution from the
        # Glass/Glossy BSDFs (you now express that as roughness=0). Closest
        # remaining option is 'BECKMANN'.
        sharp_old = ".distribution = 'SHARP'"
        sharp_new = ".distribution = 'BECKMANN'  # patched: bpy 4.0 dropped 'SHARP'"
        sharp_n = new_src.count(sharp_old)
        if sharp_n:
            new_src = new_src.replace(sharp_old, sharp_new)
            n_mat += sharp_n
        if new_src != src:
            materials_py.write_text(new_src)
            log.info("Patched %d pre-4.0 BSDF socket sites in materials.py", n_mat)

    if shapes_py.exists():
        src = shapes_py.read_text()
        new_src = src
        # `Mesh.calc_normals()` removed in Blender 4.0. The hasattr() guard
        # lets us keep the call for older Blender while no-op'ing on 4.x.
        new_src = new_src.replace(
            "        bl_mesh.calc_normals()\n",
            "        if hasattr(bl_mesh, 'calc_normals'):\n"
            "            bl_mesh.calc_normals()  # patched: no-op on bpy 4.x\n",
        )
        # `bmesh.ops.create_uvsphere` kwarg renamed in 4.0.
        new_src = new_src.replace("diameter=radius,", "radius=radius,")

        # Replace the addon's bundled bl_import_obj.load() call (the bundled
        # importer uses `Mesh.create_normals_split` removed in bpy 4.0) with
        # a call to Blender's built-in C++ OBJ importer at bpy.ops.wm.obj_import
        # (available since Blender 3.4, much faster than the legacy Python one).
        obj_load_old = "    bl_meshes = bl_import_obj.load(abs_path)\n"
        obj_load_new = (
            "    # patched: swap the addon's bundled OBJ importer for the\n"
            "    # built-in bpy.ops.wm.obj_import (Blender 3.4+, bpy 4.x safe).\n"
            "    _existing = set(bpy.data.objects)\n"
            "    bpy.ops.wm.obj_import(filepath=str(abs_path), forward_axis='Y', up_axis='Z')\n"
            "    _new_objs = [o for o in bpy.data.objects if o not in _existing]\n"
            "    bl_meshes = [o.data for o in _new_objs if o.type == 'MESH']\n"
            "    for _o in _new_objs:\n"
            "        bpy.data.objects.remove(_o, do_unlink=True)\n"
        )
        if obj_load_old in new_src:
            new_src = new_src.replace(obj_load_old, obj_load_new)

        if new_src != src:
            shapes_py.write_text(new_src)
            log.info("Patched bpy 4.x shapes.py incompatibilities")

    if renderer_py.exists():
        src = renderer_py.read_text()
        # 'SOBOL_BURLEY' was renamed to 'TABULATED_SOBOL' in Cycles 4.2. The
        # addon picks 'SOBOL_BURLEY' for any bpy >= 3.5, but that string is
        # gone on 4.2+ (allowed: 'AUTOMATIC', 'TABULATED_SOBOL', 'BLUE_NOISE').
        new_src = src.replace("'SOBOL_BURLEY'", "'TABULATED_SOBOL'")
        if new_src != src:
            renderer_py.write_text(new_src)
            log.info("Patched bpy 4.x sampling_pattern enum")


def _install_addon_from_zip(
    bpy: Any, zip_path: Path, *, roughplastic_coat_weight: float = 0.8
) -> str:
    """Extract the addon manually, normalize the folder name to a Python identifier.

    ``bpy.ops.preferences.addon_install`` keeps the zip's top-level folder name
    verbatim. The mitsuba-blender release ships with ``mitsuba-blender/`` (hyphen),
    which Python cannot import as a module, so Blender silently registers zero
    addons. We extract ourselves and rename to ``mitsuba_blender``.
    """
    addons_dir = _addons_dir(bpy)
    top = _zip_top_dir(zip_path)
    target_module = top.replace("-", "_")
    target_dir = addons_dir / target_module

    # Remove both the broken hyphen-named install (if any) and the previous
    # underscore install so a re-run gives a clean slate.
    for stale in (addons_dir / top, target_dir):
        if stale.exists():
            log.info("Removing stale addon dir %s", stale)
            shutil.rmtree(stale)

    log.info("Extracting %s -> %s", zip_path, target_dir)
    target_dir.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(zip_path) as z:
        prefix = top + "/"
        for member in z.namelist():
            if not member.startswith(prefix) or member.endswith("/"):
                continue
            rel = member[len(prefix):]
            out_path = target_dir / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as src, out_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)

    _patch_addon_for_compat(
        target_dir, roughplastic_coat_weight=roughplastic_coat_weight
    )
    return target_module


def ensure_mitsuba_addon(
    *, force_reinstall: bool = False, roughplastic_coat_weight: float = 0.8
) -> str:
    """Install + enable the mitsuba-blender addon. Returns the module name.

    Args:
        force_reinstall: re-download + re-patch even if the addon is already enabled.
        roughplastic_coat_weight: value written into the addon's hardcoded
            ``Coat Weight`` slot for roughplastic/plastic BSDFs. The upstream
            default is 0.8 which makes Cycles renders much glossier than the
            equivalent Mitsuba render; set 0.0 to drop the coat layer and get
            closer to Mitsuba's roughplastic appearance.
    """
    bpy = _import_bpy()
    if not force_reinstall:
        existing = _addon_is_enabled(bpy)
        if existing is not None:
            log.info("mitsuba-blender addon already enabled (%s)", existing)
            return existing

    if not ADDON_CACHE_PATH.exists() or force_reinstall:
        _download_addon_zip(ADDON_CACHE_PATH)

    module_name = _install_addon_from_zip(
        bpy, ADDON_CACHE_PATH,
        roughplastic_coat_weight=roughplastic_coat_weight,
    )
    # Tell Blender to rescan the scripts dir so the freshly-installed module
    # becomes visible to addon_enable.
    if hasattr(bpy.utils, "refresh_script_paths"):
        bpy.utils.refresh_script_paths()
    log.info("Enabling addon module=%s", module_name)
    bpy.ops.preferences.addon_enable(module=module_name)
    return module_name


# Cycles config ---------------------------------------------------------------


def _configure_cycles(
    bpy: Any, *, device: str = "OPTIX", denoiser: str = "OPTIX"
) -> None:
    """Set engine to Cycles, pick the requested device, enable denoising."""
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.feature_set = "SUPPORTED"

    prefs = bpy.context.preferences.addons.get("cycles")
    if device == "CPU":
        scene.cycles.device = "CPU"
    else:
        scene.cycles.device = "GPU"
        if prefs is not None:
            cprefs = prefs.preferences
            # Try the requested compute device first; fall back to CUDA/OptiX/None.
            for backend in (device, "OPTIX", "CUDA", "HIP", "ONEAPI"):
                try:
                    cprefs.compute_device_type = backend
                    break
                except (TypeError, ValueError):
                    continue
            cprefs.get_devices()
            for d in cprefs.devices:
                # Enable all non-CPU devices of the chosen backend.
                d.use = d.type != "CPU"

    # Linear scene-referred so our existing tonemap.compute_scene_exposure works.
    scene.view_settings.view_transform = "Raw"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.display_settings.display_device = "sRGB"

    vl = scene.view_layers[0]
    if denoiser == "NONE":
        vl.cycles.use_denoising = False
    else:
        vl.cycles.use_denoising = True
        try:
            scene.cycles.denoiser = denoiser  # 'OPTIX' or 'OPENIMAGEDENOISE'
        except (TypeError, ValueError):
            scene.cycles.denoiser = "OPENIMAGEDENOISE"


def init_blender(
    *,
    ensure_addon: bool = True,
    device: str = "OPTIX",
    denoiser: str = "OPTIX",
    roughplastic_coat_weight: float = 0.8,
) -> str:
    """Initialize Blender + Cycles for our pipeline. Returns the bpy version string."""
    bpy = _import_bpy()
    if ensure_addon:
        ensure_mitsuba_addon(roughplastic_coat_weight=roughplastic_coat_weight)
    _configure_cycles(bpy, device=device, denoiser=denoiser)
    ver = ".".join(str(x) for x in bpy.app.version)
    log.info("Blender %s ready (engine=Cycles, device=%s, denoiser=%s)",
             ver, device, denoiser)
    return ver


# Scene load ------------------------------------------------------------------


def _wipe_default_scene(bpy: Any) -> None:
    """Empty the current scene without resetting preferences/enabled addons.

    ``bpy.ops.wm.read_factory_settings(use_empty=True)`` would also disable
    every addon we just enabled (including mitsuba-blender), so we delete
    objects + orphan data manually instead.
    """
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    # Orphan data left behind after object removal: meshes, materials,
    # images, cameras, lights. Free them so subsequent imports don't pile up.
    for coll in (
        bpy.data.meshes, bpy.data.materials, bpy.data.images,
        bpy.data.cameras, bpy.data.lights, bpy.data.curves,
        bpy.data.textures,
    ):
        for item in list(coll):
            try:
                coll.remove(item)
            except (RuntimeError, ReferenceError):
                continue


def load_scene_blender(
    scene_xml: Path,
) -> tuple[dict[int, Any], dict[int, Any]]:
    """Clear the current Blender scene and import a Mitsuba XML.

    Returns two mappings:

    * ``pass_index_to_object``: per-object index used by the ``IndexOB`` pass.
      Indices start at 1; 0 is the "no hit" sentinel.
    * ``pass_index_to_material``: per-material index used by the ``IndexMA``
      pass. Same indexing convention. Each Blender material gets a unique
      ``pass_index`` so the per-pixel material segmentation is meaningful
      (without this, every pixel in the ``IndexMA`` pass would be 0).
    """
    bpy = _import_bpy()
    _wipe_default_scene(bpy)

    if not scene_xml.exists():
        raise FileNotFoundError(scene_xml)

    if not hasattr(bpy.ops.import_scene, "mitsuba"):
        raise RuntimeError(
            "bpy.ops.import_scene.mitsuba is unavailable. Make sure the "
            "mitsuba-blender add-on is installed and enabled "
            "(call ensure_mitsuba_addon() first)."
        )

    log.info("Importing Mitsuba scene: %s", scene_xml)
    bpy.ops.import_scene.mitsuba(filepath=str(scene_xml))

    # The mitsuba-blender importer's `init_mitsuba_renderer` sets
    # scene.render.engine = 'MITSUBA' (the addon's own render engine, which
    # we don't want — we render with Cycles). Force Cycles back.
    bpy.context.scene.render.engine = "CYCLES"

    obj_mapping: dict[int, Any] = {}
    next_idx = 1
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        obj.pass_index = next_idx
        obj_mapping[next_idx] = obj
        next_idx += 1

    mat_mapping: dict[int, Any] = {}
    next_mat_idx = 1
    for mat in bpy.data.materials:
        if mat is None:
            continue
        mat.pass_index = next_mat_idx
        mat_mapping[next_mat_idx] = mat
        next_mat_idx += 1

    log.info(
        "Loaded %d mesh objects and %d materials",
        len(obj_mapping), len(mat_mapping),
    )
    return obj_mapping, mat_mapping


# Bbox + scene info -----------------------------------------------------------


def compute_bbox_blender() -> tuple[
    tuple[float, float, float], tuple[float, float, float]
]:
    """Axis-aligned union bbox of every mesh in the current Blender scene."""
    bpy = _import_bpy()
    lo = np.full(3, np.inf, dtype=np.float64)
    hi = np.full(3, -np.inf, dtype=np.float64)
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        mw = np.asarray(obj.matrix_world, dtype=np.float64)
        corners = np.asarray(
            [list(c) for c in obj.bound_box], dtype=np.float64
        )  # (8, 3)
        homog = np.concatenate(
            [corners, np.ones((corners.shape[0], 1), dtype=np.float64)], axis=1
        )  # (8, 4)
        world = (mw @ homog.T).T[:, :3]
        lo = np.minimum(lo, world.min(axis=0))
        hi = np.maximum(hi, world.max(axis=0))
    if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        raise RuntimeError("No mesh objects in the Blender scene; bbox undefined.")
    return (float(lo[0]), float(lo[1]), float(lo[2])), (float(hi[0]), float(hi[1]), float(hi[2]))


def derive_scene_info_blender(
    height_range: tuple[float, float] = (0.8, 1.8),
    placement_margin: float = 0.3,
    height_margin: float = 0.3,
    up_axis_override: UpAxis | None = None,
) -> SceneInfo:
    bb_min, bb_max = compute_bbox_blender()
    return scene_info_from_bbox(
        bb_min=bb_min,
        bb_max=bb_max,
        height_range=height_range,
        placement_margin=placement_margin,
        height_margin=height_margin,
        up_axis_override=up_axis_override,
    )


# Inside-room raycast ---------------------------------------------------------


def inside_room_test_blender(
    positions: NDArray[np.float64],
    up_axis: UpAxis,
    max_room_extent: float = 15.0,
    n_horizontal: int = 8,
) -> NDArray[np.bool_]:
    """Same semantics as ``scene_utils.inside_room_test`` using Blender raycasts.

    Casts rays in ±up plus ``n_horizontal`` evenly-spaced horizontal directions
    from each candidate position. A point is "inside" iff every probe ray hits
    a surface within ``max_room_extent``.
    """
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must be (N, 3); got {positions.shape}")
    n = positions.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.bool_)

    bpy = _import_bpy()
    depsgraph = bpy.context.evaluated_depsgraph_get()

    up_idx = _AXIS_INDEX[up_axis]
    horiz_axes = [i for i in range(3) if i != up_idx]

    up_vec = np.zeros(3, dtype=np.float64)
    up_vec[up_idx] = 1.0
    down_vec = -up_vec

    angles = np.linspace(0.0, 2.0 * np.pi, n_horizontal, endpoint=False)
    horiz_dirs = np.zeros((n_horizontal, 3), dtype=np.float64)
    horiz_dirs[:, horiz_axes[0]] = np.cos(angles)
    horiz_dirs[:, horiz_axes[1]] = np.sin(angles)

    all_dirs = np.vstack([up_vec[None, :], down_vec[None, :], horiz_dirs])

    out = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        origin = (float(positions[i, 0]), float(positions[i, 1]), float(positions[i, 2]))
        ok = True
        for d in all_dirs:
            direction = (float(d[0]), float(d[1]), float(d[2]))
            hit, location, _normal, _index, _obj, _matrix = bpy.context.scene.ray_cast(
                depsgraph, origin, direction, distance=max_room_extent
            )
            if not hit:
                ok = False
                break
            dist = float(np.linalg.norm(np.asarray(location) - np.asarray(origin)))
            if dist > max_room_extent:
                ok = False
                break
        out[i] = ok
    return out
