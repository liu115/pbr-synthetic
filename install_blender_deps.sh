#!/usr/bin/env bash
# Install Blender-pipeline dependencies into the currently active Python env.
#
# Requirements:
#   - Python 3.11 (bpy 4.2 LTS pins this version)
#   - `python` and `pip` on PATH and pointing at the env you want
#
# Usage:
#   ./install_blender_deps.sh

set -euo pipefail

# bpy 4.2 wheels on PyPI only build for Python 3.11. Fail fast otherwise.
py_minor=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "$py_minor" != "3.11" ]]; then
    echo "ERROR: bpy 4.2 LTS requires Python 3.11; current python ($py_minor) is at $(command -v python)." >&2
    echo "Create / activate a Python 3.11 env first, then re-run this script." >&2
    exit 1
fi

echo "==> Installing into env with python = $(command -v python)"
echo "==> Installing pip packages..."
# bpy 4.2 LTS is the lowest 4.x on PyPI; bpy < 4 has been removed entirely.
# mitsuba 3.5.* is what the addon expects.
pip install \
    "mitsuba==3.5.*" \
    "bpy==4.2.*" \
    numpy \
    scipy \
    "imageio[freeimage]" \
    trimesh \
    pyyaml \
    pytest \
    mypy \
    viser

echo "==> Triggering FreeImage binary download (one-time, ~5 MB)..."
python -c "
import imageio.plugins.freeimage as fi
try:
    fi.download()
    print('FreeImage ready.')
except Exception as e:
    print(f'FreeImage download skipped: {e}')
"

echo "==> Installing + patching the mitsuba-blender add-on..."
# Runs src/scene_blender.py's ensure_mitsuba_addon, which downloads the
# v0.4.0 zip and applies all compatibility patches (see docs/blender-env.md).
# bpy sometimes segfaults at interpreter shutdown when mitsuba was also
# loaded in-process (Cycles/drjit cleanup race). The patches are applied
# synchronously before that, so we tolerate any non-zero exit and verify
# the result with a post-check below.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
set +e
python scripts/render_scene.py --install-addon-only
addon_rc=$?
set -e

addon_dir=$(python -c "
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
echo "==> Done. Run a debug render with:"
echo "      python scripts/render_scene.py \\"
echo "          --scene /path/to/scene_v3.xml \\"
echo "          --output /tmp/out --num-cameras 4 --debug --seed 42 \\"
echo "          --cycles-device CPU --denoiser NONE"
