#!/usr/bin/env bash
# Create a self-contained conda env for the Blender (Cycles) rendering
# pipeline at src/render_scene_blender.py.
#
# Why a separate env from `mitsuba`:
# - `bpy` (Blender as a Python module) pins to a specific Python version.
# - The mitsuba-blender add-on v0.4.0 (latest, Mar 2024) was written for
#   Mitsuba 3.5; later 3.x releases dropped APIs it uses. The main `mitsuba`
#   env uses Mitsuba 3.8, so the Blender pipeline can't share it.
# - PyPI no longer hosts bpy < 4.2, so we pin to bpy 4.2 LTS + apply
#   compatibility patches to the add-on at install time (see
#   src/scene_blender.py `_patch_addon_for_compat`).
#
# Usage:
#   ./install_blender_env.sh [env_name]
#
# Default env name is `pbr-capture-blender`. The script is idempotent: if the
# env already exists it skips creation and only ensures the pip packages.

set -euo pipefail

ENV_NAME="${1:-pbr-capture-blender}"
PYTHON_VERSION="3.11"  # required by bpy 4.2-4.5

# Locate conda. Prefer `conda` in PATH; fall back to common install location.
if ! command -v conda >/dev/null 2>&1; then
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "/opt/conda/etc/profile.d/conda.sh"
    else
        echo "ERROR: conda is not on PATH and miniconda3 wasn't found in a standard location." >&2
        exit 1
    fi
fi

echo "==> Target conda env: $ENV_NAME"

# Create env (idempotent — `conda create` errors if it already exists, so
# we check first).
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "==> Env $ENV_NAME already exists; will install/update packages into it."
else
    echo "==> Creating new conda env with Python $PYTHON_VERSION..."
    conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
fi

echo "==> Installing pip packages..."
# bpy 4.2 LTS is the lowest 4.x on PyPI; bpy < 4 has been removed entirely.
# mitsuba 3.5.* still on PyPI, and that's what the addon expects.
# imageio[freeimage] brings in the FreeImage backend used for EXR I/O; the
# binary itself is downloaded lazily by imageio on first use, but the
# extras_require pulls in the imageio-ffmpeg etc. wrappers it needs.
conda run -n "$ENV_NAME" --no-capture-output pip install \
    "mitsuba==3.5.*" \
    "bpy==4.2.*" \
    numpy \
    "imageio[freeimage]" \
    trimesh \
    pyyaml \
    pytest \
    mypy \
    viser

echo "==> Triggering FreeImage binary download (one-time, ~5 MB)..."
conda run -n "$ENV_NAME" --no-capture-output python -c "
import imageio.plugins.freeimage as fi
try:
    fi.download()
    print('FreeImage ready.')
except Exception as e:
    print(f'FreeImage download skipped: {e}')
"

echo "==> Installing + patching the mitsuba-blender add-on..."
# Runs src/scene_blender.py's ensure_mitsuba_addon, which downloads the
# v0.4.0 zip and applies all compatibility patches (version pin,
# ThreadEnvironment, xml_to_props polyfill, bpy 4.x BSDF/normals/uvsphere).
# Note: bpy sometimes segfaults at interpreter shutdown when mitsuba was
# also loaded in-process (a Cycles/drjit memory-cleanup interaction).
# The patches are applied synchronously before that, so we tolerate any
# non-zero exit and verify with a post-check below.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
set +e
conda run -n "$ENV_NAME" --no-capture-output python -m src.render_scene_blender --install-addon-only
addon_rc=$?
set -e
# Verify the addon directory ended up where Blender expects it. Use python
# to resolve bpy.utils.user_resource so the version subdir is picked up
# correctly (Blender 4.2 vs 4.5 etc.).
addon_dir=$(conda run -n "$ENV_NAME" --no-capture-output python -c "
import bpy, pathlib
p = pathlib.Path(bpy.utils.user_resource('SCRIPTS', path='addons')) / 'mitsuba_blender'
print(p if (p / '__init__.py').exists() else '')
" 2>/dev/null | tail -1)
if [ -n "$addon_dir" ]; then
    echo "==> Addon installed at: $addon_dir  (install RC=$addon_rc; non-zero RC is typically a benign shutdown crash)"
else
    echo "ERROR: addon failed to install (no __init__.py under user-addons/mitsuba_blender)." >&2
    exit "$addon_rc"
fi

echo ""
echo "==> Done. Activate the env with:"
echo "      conda activate $ENV_NAME"
echo "==> Then run a debug render with:"
echo "      python -m src.render_scene_blender \\"
echo "          --scene /path/to/scene_v3.xml \\"
echo "          --output /tmp/out --num-cameras 4 --debug --seed 42 \\"
echo "          --cycles-device CPU --denoiser NONE"
