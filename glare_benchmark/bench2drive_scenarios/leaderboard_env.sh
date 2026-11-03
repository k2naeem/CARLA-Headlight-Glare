# Shared by run_family.sh and run_glare_check.sh; sourced, not run.
#
# The settings DeepSight's run_evaluation.sh exports, but with your own
# CARLA_ROOT (that script hard-codes the authors' path), and one function that
# calls the leaderboard. Expects HERE (the calling script's folder).

: "${DEEPSIGHT_DIR:?set DEEPSIGHT_DIR to your DeepSight checkout}"
: "${CARLA_ROOT:?set CARLA_ROOT to your CARLA 0.9.15 folder}"

REPO=$(cd "$HERE/../.." && pwd)
B2D=$(cd "$DEEPSIGHT_DIR/bench2drive" && pwd)
PORT=${PORT:-30000}
TM_PORT=${TM_PORT:-50000}
GPU_RANK=${GPU_RANK:-0}

export CARLA_ROOT
export CARLA_SERVER=$CARLA_ROOT/CarlaUE4.sh
# $B2D: DeepSight's agent imports team_code.*; $REPO: the glare modules.
export PYTHONPATH=${PYTHONPATH:-}:$CARLA_ROOT/PythonAPI:$CARLA_ROOT/PythonAPI/carla:$B2D:$B2D/leaderboard:$B2D/leaderboard/team_code:$B2D/scenario_runner:$REPO
export SCENARIO_RUNNER_ROOT=$B2D/scenario_runner
export LEADERBOARD_ROOT=$B2D/leaderboard
export CHALLENGE_TRACK_CODENAME=SENSORS
export DEBUG_CHALLENGE=0
export IS_BENCH2DRIVE=True
export PLANNER_TYPE=only_traj
export ROUTES=$B2D/leaderboard/data/bench2drive220.xml
export GLARE_STRICT=${GLARE_STRICT:-1}

# Label for an output folder: the family name, or routes_<ids> for listed ids.
run_label() {
    case "$1" in
        *,*|[0-9]*) echo "routes_${1//,/_}" ;;
        *) echo "$1" ;;
    esac
}

# run_leaderboard <agent .py> <agent config> <route ids> <save path>
run_leaderboard() {
    export SAVE_PATH=$4
    mkdir -p "$SAVE_PATH"
    (cd "$B2D" && CUDA_VISIBLE_DEVICES=$GPU_RANK python leaderboard/leaderboard/leaderboard_evaluator.py \
        --routes="$ROUTES" \
        --routes-subset="$3" \
        --repetitions=1 \
        --track=SENSORS \
        --checkpoint="$SAVE_PATH/results.json" \
        --agent="$1" \
        --agent-config="$2" \
        --debug=0 \
        --resume=True \
        --port="$PORT" \
        --traffic-manager-port="$TM_PORT" \
        --gpu-rank="$GPU_RANK")
}
