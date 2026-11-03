#!/usr/bin/env bash
# Check that a machine can run DeepSight with glare on Bench2Drive.
#
# Run this on the Linux GPU machine, from anywhere:
#
#   export DEEPSIGHT_DIR=~/DeepSight          # your DeepSight checkout
#   export CARLA_ROOT=~/CARLA_0.9.15           # CARLA 0.9.15 with AdditionalMaps
#   export MODEL_PATH=~/checkpoints/deepsight  # only for DeepSight runs
#   bash check_setup.sh
#
# It only reads and reports; it installs nothing. Pass --clone to clone
# DeepSight into $DEEPSIGHT_DIR when it is missing.
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
DEEPSIGHT_URL=https://github.com/hotdogcheesewhite/DeepSight
fails=0

ok()   { printf '  ok    %s\n' "$1"; }
warn() { printf '  WARN  %s\n' "$1"; }
bad()  { printf '  FAIL  %s\n' "$1"; fails=$((fails + 1)); }

echo "Environment variables"
for v in DEEPSIGHT_DIR CARLA_ROOT; do
    if [ -n "${!v:-}" ]; then ok "$v=${!v}"; else bad "$v is not set"; fi
done

echo "DeepSight"
if [ -n "${DEEPSIGHT_DIR:-}" ] && [ ! -d "$DEEPSIGHT_DIR" ] && [ "${1:-}" = "--clone" ]; then
    git clone --depth 1 "$DEEPSIGHT_URL" "$DEEPSIGHT_DIR" && ok "cloned $DEEPSIGHT_URL"
fi
B2D=${DEEPSIGHT_DIR:-/nonexistent}/bench2drive
if [ -f "$B2D/leaderboard/leaderboard/leaderboard_evaluator.py" ]; then
    ok "leaderboard found in $B2D"
else
    bad "no $B2D/leaderboard (clone it: bash check_setup.sh --clone)"
fi
[ -f "$B2D/team_code/qwen_b2d_agent.py" ] && ok "DeepSight agent found" || bad "no team_code/qwen_b2d_agent.py"
[ -f "$B2D/leaderboard/data/bench2drive220.xml" ] && ok "bench2drive220.xml found" || bad "no bench2drive220.xml"
# DeepSight's evaluator starts CARLA with a hard-coded -graphicsadapter=4,
# which fails on machines with fewer GPUs; see the README.
if grep -q 'graphicsadapter={int(4)}' "$B2D/leaderboard/leaderboard/leaderboard_evaluator.py" 2>/dev/null; then
    warn "leaderboard_evaluator.py starts CARLA on -graphicsadapter=4 (see README, 'GPU number')"
fi

echo "CARLA"
if [ -x "${CARLA_ROOT:-/nonexistent}/CarlaUE4.sh" ]; then ok "CarlaUE4.sh found"; else bad "no CarlaUE4.sh in CARLA_ROOT"; fi
maps=${CARLA_ROOT:-/nonexistent}/CarlaUE4/Content/Carla/Maps
for town in Town12 Town13; do
    if ls "$maps" 2>/dev/null | grep -q "^$town"; then ok "$town installed"
    else bad "$town missing: extract AdditionalMaps_0.9.15 into CARLA_ROOT"; fi
done

echo "Model (only for run_family.sh; run_glare_check.sh loads none)"
if [ -z "${MODEL_PATH:-}" ]; then
    warn "MODEL_PATH not set: fine for run_glare_check.sh"
elif [ -d "$MODEL_PATH" ] && ls "$MODEL_PATH" 2>/dev/null | grep -q 'config.json'; then
    ok "checkpoint folder has config.json"
else
    bad "MODEL_PATH is not a Hugging Face model folder (no config.json)"
fi

echo "Glare code"
for f in glare_qwen_agent.py glare_injector.py glare_mask_closed_loop.py; do
    [ -f "$REPO/$f" ] && ok "$f" || bad "no $REPO/$f (check out branch bench2drive-glare-agent)"
done
[ -f "$HERE/night_routes.json" ] && ok "night_routes.json" || bad "no night_routes.json: python3 routes.py build $B2D/leaderboard/data/bench2drive220.xml"

echo "Python (the interpreter that will run the leaderboard: $(command -v python || echo none))"
PYTHONPATH="${PYTHONPATH:-}:${CARLA_ROOT:-}/PythonAPI/carla:$B2D" python - <<'EOF' || bad "python check could not run"
import importlib
# Glare check: the glare code, the stand-in agent and CARLA's BasicAgent.
# DeepSight runs: also the model's packages.
groups = (('glare check', ('carla', 'cv2', 'numpy', 'PIL', 'networkx', 'shapely')),
          ('DeepSight only', ('torch', 'transformers', 'qwen_vl_utils', 'scipy')))
for label, names in groups:
    for name in names:
        try:
            importlib.import_module(name)
            print('  ok    import %s (%s)' % (name, label))
        except Exception as exc:
            print('  FAIL  import %s (%s; %s)' % (name, label, exc.__class__.__name__))
try:
    import torch
    print('  ok    CUDA available' if torch.cuda.is_available() else '  FAIL  CUDA not available')
except Exception:
    pass
EOF

echo "GPU"
if command -v nvidia-smi >/dev/null; then ok "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else bad "nvidia-smi not found: CARLA and DeepSight need an NVIDIA GPU"; fi

echo
if [ "$fails" -eq 0 ]; then echo "All checks passed. Next: bash run_glare_check.sh smoke 1.0"
else echo "$fails check(s) failed."; exit 1; fi
