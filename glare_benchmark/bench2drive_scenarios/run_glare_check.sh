#!/usr/bin/env bash
# Check the glare on Bench2Drive night routes, with no DeepSight model.
#
#   bash run_glare_check.sh <target> <glare level>
#
#   target       a family (overtaking, junction_left_turns, other_junctions,
#                crossings, controls, other_night), "smoke" (route 3514),
#                "night" (all 53), or route ids like 3410,3436
#   glare level  0, 0.5, 1.0, 1.5 or 2.0
#
# The leaderboard's NpcAgent drives the route; glare_check_agent.py saves
# DeepSight's camera input with your glare, plus a clean front-camera copy.
# Then: python3 summarize_glare_check.py <the output folder printed below>
#
# Needs DEEPSIGHT_DIR and CARLA_ROOT (no MODEL_PATH, no model is loaded).
# Optional: OUT_ROOT (default ~/glare_check_runs), PORT, TM_PORT, GPU_RANK,
# GLARE_STRICT (default 1: stop a route if a frame would miss its glare).
set -euo pipefail

if [ $# -ne 2 ]; then
    sed -n '2,18p' "$0"
    exit 2
fi
TARGET=$1
LEVEL=$2

HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/leaderboard_env.sh"

OUT_ROOT=${OUT_ROOT:-$HOME/glare_check_runs}
SUBSET=$(python3 "$HERE/routes.py" ids "$TARGET")
SAVE=$OUT_ROOT/$(run_label "$TARGET")/severity_$LEVEL
export GLARE_SEVERITY=$LEVEL

echo "Routes:     $SUBSET"
echo "Glare:      $LEVEL (strict=$GLARE_STRICT), no model"
echo "Output:     $SAVE"
run_leaderboard "$HERE/glare_check_agent.py" glare_check "$SUBSET" "$SAVE"
echo "Summarize:  python3 $HERE/summarize_glare_check.py $SAVE"
