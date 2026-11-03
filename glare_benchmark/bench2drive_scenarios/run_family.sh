#!/usr/bin/env bash
# Run DeepSight with headlight glare on a set of Bench2Drive night routes.
#
#   bash run_family.sh <target> <glare level>
#
#   target       a family from night_routes.json (overtaking, junction_left_turns,
#                other_junctions, crossings, controls, other_night), "smoke"
#                (route 3514), "night" (all 53), or route ids like 3410,3436
#   glare level  0, 0.5, 1.0, 1.5 or 2.0; 0 runs DeepSight unchanged
#
# Needs DEEPSIGHT_DIR, CARLA_ROOT and MODEL_PATH (see check_setup.sh).
# Optional: OUT_ROOT (default ~/glare_benchmark_runs), PORT, TM_PORT, GPU_RANK,
# GLARE_STRICT (default 1: stop a route if a frame would miss its glare).
#
# To check the glare on these routes without DeepSight, use run_glare_check.sh.
set -euo pipefail

if [ $# -ne 2 ]; then
    sed -n '2,15p' "$0"
    exit 2
fi
TARGET=$1
LEVEL=$2

HERE=$(cd "$(dirname "$0")" && pwd)
: "${MODEL_PATH:?set MODEL_PATH to the DeepSight checkpoint folder}"
source "$HERE/leaderboard_env.sh"

OUT_ROOT=${OUT_ROOT:-$HOME/glare_benchmark_runs}
SUBSET=$(python3 "$HERE/routes.py" ids "$TARGET")
SAVE=$OUT_ROOT/$(run_label "$TARGET")/severity_$LEVEL
export GLARE_SEVERITY=$LEVEL

echo "Routes:     $SUBSET"
echo "Glare:      $LEVEL (strict=$GLARE_STRICT)"
echo "Output:     $SAVE"
run_leaderboard "$REPO/glare_qwen_agent.py" "$MODEL_PATH" "$SUBSET" "$SAVE"
