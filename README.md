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

`transforms.json` is OpenCV convention (camera +Z forward, +Y down).
`coordinate_convention: "opencv"` is set explicitly. Depth values are
`ray.t`, NOT perspective Z-depth; this is recorded in `metadata.depth_convention`.

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
  camera_sampling.py    Pose dataclass, orientation sampling, depth filter
  render.py             Sensor build, beauty + AOV passes, material LUT
  tonemap.py            Per-scene exposure derivation, gamma encode
  io_utils.py           Output layout, JSON schemas, EXR I/O, resume
  render_scene.py       CLI entry point
  visualize.py          Viser viewer
tests/
  test_*.py             Hermetic unit tests
  test_render.py        @pytest.mark.mitsuba_slow end-to-end smoke test
render_all.sh           Loops the CLI over bedroom + kitchen
pyproject.toml          pytest config, mypy --strict config
```
