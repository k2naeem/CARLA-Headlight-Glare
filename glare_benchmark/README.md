# Glare benchmark

A library of closed-loop tests that measure how DeepSight drives when oncoming
headlight glare is added to its camera images. Glare comes from
`glare_qwen_agent.py` at the repo root; this folder holds the scenarios.

```
glare_benchmark/
├── bench2drive_scenarios/   Bench2Drive night routes that run as they are
└── custom_scenarios/        scenarios Bench2Drive cannot provide; we build these
```

**bench2drive_scenarios/** needs no new scenario code. Bench2Drive's night
routes already contain lit oncoming traffic and the decisions that glare
affects (overtaking, junction turns, crossings), and Bench2Drive scores them.
These scripts pick the routes and either check the glare on them with no model
(`run_glare_check.sh`) or run DeepSight with glare at each level
(`run_family.sh`), keeping each level's results apart.

**custom_scenarios/** covers what Bench2Drive cannot guarantee: glare and the
hazard at the same moment, high beams, unlit roads, a fixed number of oncoming
cars, and sun glare.

## Where this runs

CARLA 0.9.15 and DeepSight need Linux and an NVIDIA GPU, so the scripts run on
the GPU machine, not on a Mac. Check out this repo on that machine on branch
`bench2drive-glare-agent`, then follow
[bench2drive_scenarios/README.md](bench2drive_scenarios/README.md).
