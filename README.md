# pbr-capture-synthetic

Multi-view synthetic indoor capture for PBR / inverse-rendering benchmarks.
Samples N cameras inside a Mitsuba scene, filters by raycast + low-res depth,
and renders a NeRF-style dataset (RGB LDR + HDR, depth, normal, albedo,
specular albedo, per-pixel roughness + metallic, emission, material
segmentation, optional per-pixel envmaps) plus a colored tessellated PLY.
Two backends in one CLI: **Mitsuba 3** beauty + **Blender Cycles** beauty &
AOVs. Designed to consume the Mitsuba-formatted indoor scenes from
[Benedikt Bitterli's resource page](https://benedikt-bitterli.me/resources/).


## Install
```bash
conda create -n mitsuba-blender python=3.11 -y
conda activate mitsuba-blender
./install_blender_deps.sh
```

## Render a scene

```bash
python scripts/render_scene.py \
    --scene /path/to/scene_v3.xml \
    --output /path/to/out \
    --num-cameras 200 \
    --backend both \
    --remove-transparent
```

Useful flags (see `--help` for the rest):

- `--backend {mitsuba,blender,both}` — which renderer produces beauty.
  Default `blender`. AOVs are always Cycles.
- `--sampler {uniform,wall_walk}` — pose sampler. `wall_walk` is FIPT-style.
- `--remove-transparent` — strip dielectric / null BSDFs before rendering.
- `--debug` — 160×120 + low SPP for fast iteration. Envmaps off by default
  in debug; force on with `--envmap`.
- `--cycles-device {OPTIX,CUDA,CPU}` / `--denoiser {OPTIX,OPENIMAGEDENOISE,NONE}`
- `--envmap-patch-size N` — spatial-grid spacing for per-pixel envmaps.

See `docs/operational.md` for `scene_config.yaml`, resume behavior, and
`--only-ply`.

## Output schema

```
<output>/
  scene.ply                       colored tessellated mesh (Z-up)
  preview.png                     contact sheet
  transforms.json                 NeRF-style poses + intrinsics
  metadata.json                   render config, seeds, exposure, stats
  rgb_<backend>/   0000.png       LDR uint8       (one dir per backend)
  rgb_hdr_<backend>/ 0000.exr     HDR linear radiance float32
  depth/           0000.exr       perspective z-depth in meters
  normal/          0000.exr       world-space surface normal
  albedo/          0000.exr       Cycles DiffCol (k_d)
  specular_albedo/ 0000.exr       Cycles GlossCol (k_s)
  emission/        0000.exr       Cycles Emit
  material_index/  0000.exr       per-material ID (uint16-ish)
  roughness/       0000.exr       per-pixel (shader AOV)
  metallic/        0000.exr       per-pixel (shader AOV)
  envmap/          0000.exr       tiled per-pixel envmap mosaic (optional)
```

## Camera + world convention

- **Camera frame:** OpenCV — `+X` right, `+Y` down, `+Z` forward (look
  direction). `transform_matrix` columns 0/1/2 are the camera-frame basis
  vectors in world space; column 3 is camera position.
- **World frame:** Z-up. `scene.ply` is exported in the same frame as the
  poses, so it overlays the fused TSDF mesh and frustums without rotation.
- **Depth:** perspective z-depth in meters (camera-space `+Z`), not ray-t.
  Open3D / COLMAP / NeRF / 3DGS consume this directly.
- Header fields in `transforms.json`: `coordinate_convention: "opencv"`,
  `world_up_axis: "z"`, `depth_format: "z_depth_meters"`.

## Visualizer

```bash
python scripts/visualize.py --data /path/to/out --port 8080
```

Viser server showing the colored mesh + every camera frustum (colored by
index, image on the far plane). Sidebar toggles the modality drawn on the
frustums (RGB / depth / normal). `--ceiling-cutoff-percentile 95` drops the
top 5% of the mesh so top-down views aren't occluded.

## TSDF fusion (coordinate-system sanity check)

```bash
python scripts/tsdf_fuse.py --data /path/to/out
```

Drops every frame's RGB + depth into an Open3D `ScalableTSDFVolume` and
writes `fused.ply` next to `scene.ply`. If poses, depth, and intrinsics
agree, the two PLYs overlap directly in MeshLab.

## Tests + type-check

```bash
pytest -m "not mitsuba_slow and not blender_slow"   # fast suite
pytest -m mitsuba_slow                               # opt-in slow Mitsuba
mypy src                                             # strict type-check

pytest -m blender_slow                               # opt-in slow Blender
```
