#!/usr/bin/env bash
# Render the known scenes into /cluster_HDD/umoja/yliu/pbr-test/rendered/.
# Usage:
#   ./render_all.sh                # full quality, default 200 cameras
#   DEBUG=1 ./render_all.sh        # debug mode (low res / spp)
#   N_CAMERAS=50 ./render_all.sh   # override camera count
set -euo pipefail

RAW_ROOT="${RAW_ROOT:-/cluster_HDD/umoja/yliu/pbr-test/raw}"
OUT_ROOT="${OUT_ROOT:-/cluster_HDD/umoja/yliu/pbr-test/rendered}"
N_CAMERAS="${N_CAMERAS:-200}"
SEED="${SEED:-42}"
SCENES=("${SCENES:-bedroom kitchen}")

EXTRA_FLAGS=()
if [[ "${DEBUG:-0}" == "1" ]]; then
  EXTRA_FLAGS+=("--debug")
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
  echo "==> Rendering $scene -> $out"
  conda run -n mitsuba --no-capture-output python "${SCRIPT_DIR}/src/render_scene.py" \
    --scene "$xml" \
    --output "$out" \
    --num-cameras "$N_CAMERAS" \
    --seed "$SEED" \
    "${EXTRA_FLAGS[@]}"
done
