#!/usr/bin/env bash
# Render the known scenes into /cluster_HDD/umoja/yliu/pbr-test/rendered/.
#
# Defaults to --backend=both (renders Mitsuba beauty AND Blender beauty into
# the same output dir, with shared AOVs from Blender) and --sampler=wall_walk
# (FIPT-style camera placement). Override via env vars; flip --remove-transparent
# off by setting REMOVE_TRANSPARENT=0.
#
# Usage:
#   ./render_all.sh                          # full quality, both backends, wall_walk
#   DEBUG=1 ./render_all.sh                  # debug mode (low res / spp / no envmap)
#   N_CAMERAS=50 ./render_all.sh             # override camera count
#   BACKEND=blender ./render_all.sh          # blender only
#   SAMPLER=uniform ./render_all.sh          # use uniform sampler
#   REMOVE_TRANSPARENT=0 ./render_all.sh     # keep glass shapes
#   ENV_NAME=pbr-capture-blender ./render_all.sh   # custom conda env
set -euo pipefail

RAW_ROOT="${RAW_ROOT:-/cluster_HDD/umoja/yliu/pbr-test/raw}"
OUT_ROOT="${OUT_ROOT:-/cluster_HDD/umoja/yliu/pbr-test/rendered}"
N_CAMERAS="${N_CAMERAS:-200}"
SEED="${SEED:-42}"
SCENES=("${SCENES:-bedroom kitchen}")
BACKEND="${BACKEND:-both}"
SAMPLER="${SAMPLER:-wall_walk}"
REMOVE_TRANSPARENT="${REMOVE_TRANSPARENT:-1}"
ENV_NAME="${ENV_NAME:-pbr-capture-blender-test}"

EXTRA_FLAGS=()
if [[ "${DEBUG:-0}" == "1" ]]; then
  EXTRA_FLAGS+=("--debug")
fi
if [[ "$REMOVE_TRANSPARENT" == "1" ]]; then
  EXTRA_FLAGS+=("--remove-transparent")
fi

mkdir -p "$OUT_ROOT"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

for scene in ${SCENES[@]}; do
  xml="${RAW_ROOT}/${scene}/scene_v3.xml"
  out="${OUT_ROOT}/${scene}"
  if [[ ! -f "$xml" ]]; then
    echo "[skip] $scene: missing $xml" >&2
    continue
  fi
  echo "==> Rendering $scene -> $out (backend=$BACKEND sampler=$SAMPLER)"
  conda run -n "$ENV_NAME" --no-capture-output python -m src.render_scene \
    --scene "$xml" \
    --output "$out" \
    --num-cameras "$N_CAMERAS" \
    --seed "$SEED" \
    --backend "$BACKEND" \
    --sampler "$SAMPLER" \
    "${EXTRA_FLAGS[@]}"
done
