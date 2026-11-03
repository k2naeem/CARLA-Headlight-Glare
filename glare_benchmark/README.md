# Glare benchmark

Closed-loop tests of oncoming headlight glare on DeepSight's camera input.

```
glare_benchmark/
├── bench2drive_scenarios/   Bench2Drive night scenarios, run with run.py
└── custom_scenarios/        scenarios Bench2Drive cannot provide; we build these
```

**bench2drive_scenarios/** saves the Bench2Drive night routes worth testing
under glare and runs them, one route at a time, with your glare on DeepSight's
six cameras. The leaderboard's NpcAgent drives for now; DeepSight can replace
it later. See [its README](bench2drive_scenarios/README.md).

**custom_scenarios/** covers what Bench2Drive cannot guarantee: glare and the
hazard at the same moment, high beams, unlit roads, a fixed number of oncoming
cars, and sun glare.

Everything runs on the Ubuntu GPU machine (CARLA needs Linux and an NVIDIA
GPU); copy the results to your Mac to view them.
