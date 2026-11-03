#!/usr/bin/env bash
# Run several families at several glare levels, one after another.
#
#   bash run_sweep.sh
#
# FAMILIES   default "overtaking junction_left_turns controls" (the first sweep)
# LEVELS     default "0 0.5 1.0 1.5 2.0"
# REPEATS    default 1; each repeat goes to OUT_ROOT/rep_<n>/, since traffic
#            differs a little between runs
# Everything else is passed through to run_family.sh (DEEPSIGHT_DIR,
# CARLA_ROOT, MODEL_PATH, OUT_ROOT, PORT, TM_PORT, GPU_RANK, GLARE_STRICT).
#
# A route that is already in a results.json is skipped (the leaderboard
# resumes), so a sweep can be restarted after a crash.
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
FAMILIES=${FAMILIES:-overtaking junction_left_turns controls}
LEVELS=${LEVELS:-0 0.5 1.0 1.5 2.0}
REPEATS=${REPEATS:-1}
BASE_OUT=${OUT_ROOT:-$HOME/glare_benchmark_runs}
: "${DEEPSIGHT_DIR:?set DEEPSIGHT_DIR to your DeepSight checkout}"

failed=()
for rep in $(seq 1 "$REPEATS"); do
    for fam in $FAMILIES; do
        for level in $LEVELS; do
            echo "=== rep $rep  $fam  glare $level ==="
            OUT_ROOT=$BASE_OUT/rep_$rep bash "$HERE/run_family.sh" "$fam" "$level" \
                || failed+=("rep $rep $fam $level")
            # CARLA processes outlive crashed runs; DeepSight ships a cleaner.
            bash "$DEEPSIGHT_DIR/bench2drive/tools/clean_carla.sh" || true
            sleep 10
        done
    done
done

if [ ${#failed[@]} -gt 0 ]; then
    printf 'Failed: %s\n' "${failed[@]}"
    exit 1
fi
echo "Sweep done: $BASE_OUT"
