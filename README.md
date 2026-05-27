# pbr-capture-synthetic

Mitsuba 3 multi-view synthetic indoor capture pipeline for PBR / inverse-rendering
benchmarks. Randomly places `N` cameras inside an indoor scene, filters them with
raycast + low-resolution depth checks, then renders a NeRF-style dataset (RGB
LDR + HDR, depth, normal, albedo, roughness, metallic) plus a viser-based
viewer.

## Setup

```bash
# All deps live in the existing `mitsuba` conda env.
conda activate mitsuba
# (one-time) extras required for type-checking + tests:
pip install pytest mypy pyyaml
# (optional) for --simplify-dense-meshes (quadric decimation):
pip install fast-simplification
```

## Rendering a scene

```bash
conda run -n mitsuba python src/render_scene.py \
    --scene /cluster_HDD/umoja/yliu/pbr-test/raw/bedroom/scene_v3.xml \
    --output /cluster_HDD/umoja/yliu/pbr-test/rendered/bedroom \
    --num-cameras 200 \
    --seed 42
```

For fast iteration:

```bash
conda run -n mitsuba python src/render_scene.py \
    --scene .../scene_v3.xml --output /tmp/out --num-cameras 8 --debug
```

Common knobs (see `--help` for the full list):

- `--debug` — 160×120, lower spp, ~minutes instead of hours.
- `--up-axis {x,y,z}` — override the auto-detected up axis.
- `--height-min` / `--height-max` — camera height range in meters (default 0.8–1.8).
- `--placement-margin` — meters to shrink the placement box on each horizontal side.
- `--filter-*` — thresholds for the depth-validity filter. If sampling rejects
  too many candidates, log output shows per-reason counts so you can tune.
- `--scene-config path.yaml` — YAML overrides; also auto-detected if a
  `scene_config.yaml` sits next to the scene XML.
- `--tessellate-spacing F` — when exporting `scene.ply`, adaptively subdivide
  textured faces so world-space edges don't exceed `F` meters (default `0.10`),
  then bilinear-bake each vertex's color from the BSDF texture. Set to `0`
  (or use `--no-tessellate`) to fall back to one mean color per textured
  shape (smaller PLY, but textures get averaged away).
- `--simplify-dense-meshes` — opt-in quadric decimation for any individual
  mesh with more than `--simplify-vertex-threshold` (default 100000)
  vertices. Useful when one shape (e.g. an over-subdivided carpet) dominates
  the PLY file size. Requires `pip install fast-simplification`; if absent,
  the flag logs a warning and keeps the mesh untouched.

### `scene_config.yaml` (optional)

```yaml
fov: 65
up_axis: y
placement_margin: 0.5
height_min: 1.0
height_max: 1.7
pitch_min_deg: -15
pitch_max_deg: 15
max_depth: 12
```

### Resume

If a previous run was interrupted, rerun the same command with the same `--seed`.
Frames whose `rgb_hdr/{idx:04d}.exr` already exists are skipped; the tonemap +
JSON-writing pass is always re-run from the available HDRs.

### Regenerate just the PLY

Use `--only-ply` to rewrite `<output>/scene.ply` from the current scene XML
without touching anything else (no camera sampling, no Mitsuba renders).
Takes seconds; convenient when you've changed `--tessellate-spacing`,
turned `--simplify-dense-meshes` on/off, or pulled a new commit with a
mesh-export fix:

```bash
conda run -n mitsuba python -m src.render_scene \
    --scene .../scene_v3.xml \
    --output /path/to/existing/output \
    --only-ply
```

## Output schema

```
<output>/
  rgb/          0000.png   ...   LDR uint8 PNG
  rgb_hdr/      0000.exr   ...   HDR linear radiance (float32 EXR)
  depth/        0000.exr   ...   ray-distance from camera (float32, meters)
  normal/       0000.exr   ...   world-space surface normal (float32, 3 ch)
  albedo/       0000.exr   ...   diffuse base color (float32, 3 ch)
  roughness/    0000.exr   ...   per-pixel roughness (float32, NaN = no hit)
  metallic/     0000.exr   ...   per-pixel metallic  (float32, NaN = no hit)
  preview.png                    4×4 contact sheet (debug-quality)
  transforms.json                NeRF-style camera poses + intrinsics
  metadata.json                  Render config, seeds, exposure, stats
```

`transforms.json` uses OpenCV camera convention (`+X right, +Y down, +Z
forward`) in a Z-up world. The header carries `coordinate_convention:
"opencv"`, `world_up_axis: "z"`, and `depth_format: "z_depth_meters"`. Depth
EXRs store perspective z-depth (camera-space `+Z`, in meters), so they feed
straight into Open3D / COLMAP / NeRF / 3DGS without conversion. `scene.ply`
is exported in the same Z-up world frame as the saved poses, so it overlays
the fused TSDF mesh and frustums directly.

Per-pixel `roughness` / `metallic` come from a per-shape best-effort lookup
(see `metadata.aov_caveats`). For the bedroom + kitchen scenes (mostly diffuse
BSDFs) they evaluate to `roughness=1.0`, `metallic=0.0`.

`scene.ply` is a colored mesh suitable for 3DGS initialization. By default,
every shape's faces are adaptively subdivided so world-space edges are
≤ `--tessellate-spacing` (default 10 cm). Textured shapes get per-vertex
colors via bilinear texture sampling; flat-`<rgb>` shapes (walls, ceiling,
etc.) get the BSDF's solid color broadcast to every new vertex. Shapes whose
faces are already finer than the spacing (e.g. an over-subdivided carpet)
skip subdivision and color the original vertices in place — no wasteful
duplication. `metadata.mesh_export.per_shape[*].mode` records one of
`tessellated_texture`, `tessellated_flat`, `fine_texture`, `fine_flat`, or
`flat` (when `--no-tessellate`).

## Blender backend (optional)

A parallel Cycles-based entry point exists at `src/render_scene_blender.py`.
It reads the same Mitsuba scene XML (via the official `mitsuba-blender`
add-on), produces the same output schema (`transforms.json`, `metadata.json`,
`rgb/`, `depth/`, `normal/`, `albedo/`, `roughness/`, `metallic/`,
`scene.ply`), and is interchangeable with the Mitsuba pipeline downstream.
`metadata.renderer` is `"blender"` for these runs (vs. `"mitsuba"` from the
Mitsuba script). Useful when you want Cycles + OptiX denoising for clean
images at low spp.

Setup — **a separate conda env** is required because the `mitsuba-blender`
add-on v0.4.0 was written for **Mitsuba 3.5** + **Blender 3.x** (pre-4.0):

- Mitsuba 3.6+ removed `xml_to_props` and `ThreadEnvironment` and changed
  the scene-parser data model.
- Blender 4.0 rewrote the Principled BSDF and renamed input sockets
  (`'Clearcoat'` → `'Coat Weight'`, `'Specular'` → `'Specular IOR Level'`,
  etc.); the addon hardcodes pre-4.0 names in several BSDF code paths.
- Blender 4.0 also removed `Mesh.calc_normals()` and `Mesh.create_normals_split()`,
  renamed `bmesh.ops.create_uvsphere(diameter=…)` to `radius=…`, dropped the
  `'SHARP'` Cycles distribution, and renamed the `'SOBOL_BURLEY'` sampling
  pattern to `'TABULATED_SOBOL'`.

The main `mitsuba` env stays at Mitsuba 3.8 for the Mitsuba pipeline; the
Blender pipeline runs in a sister env pinned to Mitsuba 3.5 + bpy 4.2. An
install script automates the env creation and applies all the addon
compatibility patches at install time:

```bash
# One-shot: creates a fresh `pbr-capture-blender` env, installs deps,
# downloads + patches the mitsuba-blender add-on. Idempotent.
./install_blender_env.sh                       # default env name
./install_blender_env.sh my-blender-env        # custom env name
```

What it installs:
- Python 3.11 (bpy 4.2 LTS's required version)
- bpy 4.2 LTS (Blender as a Python module)
- mitsuba 3.5.2 (the addon's expected Mitsuba version)
- numpy, imageio[freeimage], trimesh, pyyaml, pytest, mypy, viser
- FreeImage binary (one-time download for EXR I/O)
- mitsuba-blender add-on v0.4.0 from GitHub, extracted into Blender's
  user-addons dir and patched in-place

The addon-install step applies these idempotent compatibility patches:

1. Renames the hyphenated `mitsuba-blender/` folder to `mitsuba_blender/`
   so Python's import system can load it as a module.
2. Rewrites `DEPS_MITSUBA_VERSION` to match the installed Mitsuba version.
3. Wraps the `ThreadEnvironment` import in a try/except (Mitsuba 3.6+).
4. Polyfills `xml_to_props` via `mitsuba.parser.parse_file` (Mitsuba 3.6+).
5. Rewrites pre-4.0 Principled BSDF socket names (`Clearcoat`, `Specular`,
   etc.) to their 4.0+ equivalents, plus float→color `Specular Tint`
   fix-up and `'SHARP'` → `'BECKMANN'` for the Glass/Glossy BSDFs.
6. Guards `Mesh.calc_normals()` with `hasattr` (removed in bpy 4.0).
7. Renames `bmesh.ops.create_uvsphere(diameter=…)` → `radius=…`.
8. Replaces the addon's bundled OBJ importer (`bl_import_obj.load`) with
   Blender's built-in `bpy.ops.wm.obj_import` (3.4+; uses `create_normals_split`
   which was removed in 4.0).
9. Maps the `'SOBOL_BURLEY'` sampling pattern to `'TABULATED_SOBOL'`
   (renamed in Cycles 4.2).

Run:

```bash
conda activate pbr-capture-blender
python -m src.render_scene_blender \
    --scene /cluster_HDD/umoja/yliu/pbr-test/raw/bedroom/scene_v3.xml \
    --output /cluster_HDD/umoja/yliu/pbr-test/rendered_blender/bedroom \
    --num-cameras 200 \
    --seed 42
```

Extra Blender-only flags:

- `--cycles-device {OPTIX,CUDA,CPU}` — Cycles compute device. Default `OPTIX`.
- `--denoiser {OPTIX,OPENIMAGEDENOISE,NONE}` — denoiser for beauty renders.
  Default `OPTIX`.
- `--install-addon-only` — short-circuit that installs and enables the
  `mitsuba-blender` add-on from the latest GitHub release, then exits.

Caveats:

- BSDF mapping is approximate (`roughplastic` → Cycles Principled BSDF), so
  beauty pixels won't match the Mitsuba pipeline byte-for-byte. Same seed →
  same poses (camera sampling is renderer-free numpy), but rendered radiance
  values differ.
- `roughness` / `metallic` AOVs come from the object-index pass + a per-object
  LUT read off each material's Principled BSDF input — same caveats as the
  Mitsuba side (no spatially-varying lookup).

## Visualizer

```bash
conda run -n mitsuba python src/visualize.py --data /path/to/rendered/bedroom --port 8080
```

Opens a viser server at `http://localhost:8080`. Shows the scene mesh (colored
per shape from the BSDF reflectance values in the scene XML), every camera
frustum (colored by index, image on the far plane), and a sidebar GUI for
filtering visible cameras, switching between RGB / depth / normal display on
the frustums, and a live frustum-scale slider.

Useful flags:

- `--ceiling-cutoff-percentile P` — drop mesh faces whose vertices sit above
  the `P`-th height percentile so the ceiling doesn't occlude top-down views.
  Default `95.0` (drop top 5 %). Set to `100` to disable.
- `--frustum-scale F` — initial frustum size in meters (also adjustable live).
- `--no-mesh` — skip the scene mesh entirely (faster startup for large scenes).

## Tests + type checking

```bash
conda run -n mitsuba pytest                  # fast suite
conda run -n mitsuba pytest -m mitsuba_slow  # opt-in end-to-end render test
conda run -n mitsuba mypy src                # strict type-check
```

## Project layout

```
src/
  scene_utils.py        Mitsuba init, bbox, up-axis, raycasting, inside-room test
  scene_blender.py      Blender flavor: addon install, scene load, bbox, raycast
  camera_sampling.py    Pose dataclass, orientation sampling, depth filter
  pose_utils.py         Renderer-free pose_to_c2w (numpy reimplementation)
  render.py             Mitsuba sensor build, beauty + AOV passes, material LUT
  render_blender.py     Cycles camera + render + AOV plumbing
  tonemap.py            Per-scene exposure derivation, gamma encode
  io_utils.py           Output layout, JSON schemas, EXR I/O, resume
  mesh_utils.py         Tessellate-then-bake PLY exporter (renderer-free)
  render_scene.py       CLI entry point (Mitsuba)
  render_scene_blender.py  CLI entry point (Blender / Cycles)
  visualize.py          Viser viewer
tests/
  test_*.py             Hermetic unit tests
  test_render.py        @pytest.mark.mitsuba_slow end-to-end smoke test
render_all.sh           Loops the CLI over bedroom + kitchen
pyproject.toml          pytest config, mypy --strict config
```
