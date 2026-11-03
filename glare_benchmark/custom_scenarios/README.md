# Custom scenarios

Scenarios Bench2Drive cannot guarantee, so we build them. Nothing here is
implemented yet; this is the plan for each.

| # | Scenario | Why Bench2Drive can't | What to build | Start from |
| --- | --- | --- | --- | --- |
| 6 | Cyclist emerging inside the glare | Its crossing scenarios never place a lit oncoming car beside the cyclist or pedestrian, so glare and hazard overlap only by chance | Lit oncoming stack; cyclist released from the lane line beside the headlamps when the lead car is 15–25 m out; packaged as a route DeepSight's leaderboard runs | `carla_script_closed_loop.py` (Town07) already stages this for input checks |
| 7 | Sudden high beams on a bend | Traffic Manager picks the beams, usually low; a scenario cannot request high beams | A high-beam switch in `glare_qwen_agent.py`, plus a route where one car appears around a bend at 30–50 m | Town07 rural bends (`spawn_points.py` to pick them), rural Town12 |
| 8 | Dark-road versions | Bench2Drive turns street lights on within 100 m of the ego at night | A street-light switch, applied to the Bench2Drive families | The pole-disabling code in `carla_script_closed_loop.py` |
| 9 | Controlled oncoming stack | Traffic density varies from run to run | Exactly 1, 2 or 4 lit oncoming cars at fixed spacing | The oncoming batches in `carla_script_closed_loop.py` |
| 10 | Sun glare at dawn or dusk | The glare code only uses headlights as light sources | A sun light source in the glare code, from the weather's sun angles; future work | — |

Suggested order: 6 first (it tests the core claim that headlight glare hides
road users), then 7 and 8 (switches that also apply to the Bench2Drive runs),
then 9.
