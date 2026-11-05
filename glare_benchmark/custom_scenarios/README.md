# Custom scenarios

Scenarios Bench2Drive cannot guarantee, so we build them. Bench2Drive's
pop-out scenarios (DynamicObjectCrossing, ParkingCrossingPedestrian,
VehicleOpensDoor) never place a lit oncoming car where the hazard appears, and
its oncoming-traffic scenarios have no pop-out. These do both, on purpose,
every run.

| File | |
| --- | --- |
| `glare_popout.py` | The scenarios, as Bench2Drive leaderboard scenario classes |
| `make_routes.py` | Finds straight two-way roads in CARLA's maps and writes a route file per scenario |
| `routes/` | Made by `make_routes.py`: `glare_popout/<id>_<scenario>_<town>.xml` and `index.json` |

## The scenarios

Both run at night on a straight road with one lane each way. The glare
source is a car the scenario drives itself, headlights on, so the timing is
the same every run, whatever speed the ego drives. A vehicle glares only
while its headlights are on: the glare agent never switches on the lights of
a vehicle the scenario drives, so one it drives dark stays dark.

**`GlarePedestrianPopOut`** — a pedestrian steps out from behind an oncoming
car, inside its glare. Nobody stands in the road beforehand: the pedestrian
stays hidden until the lit oncoming car is 30 m from the ego. Then they appear
just behind its rear bumper, right beside its flank (between the car and the
centre line), inside its glare. They wait in the dark, then hurry into the
ego's lane at 2.5 m/s, timed from the ego's speed so they stop right in front
of it 0.5 s before it reaches them: too late to stop for a driver who only
sees them then. They stand there 3 s and walk off the road. Because the
release follows the car, and the step-out follows the ego, it plays out the
same way at any ego speed. The clean frame shows the pedestrian throughout.

How long the glare hides them is limited by the glare itself. Measured with
`glare()`, a pedestrian loses most of their contrast only within about 40 px
of a headlamp, and is nearly back to normal by 80 px. With one car, the
pedestrian starts at the edge of that zone (about 44 px) and is out of it
within half a second: the car's approach sweeps its lamps outward while the
pedestrian walks inward. By the time they step out, 8–14 m ahead of the ego,
the car is beside the ego and its lamps are at the image edge. So the glare
hides them appearing, and darkness (`PedestrianDark`) hides them stepping
out. Every frame is labelled (below), so the dataset records exactly which
frames hide them.

**Two cars (`PedestrianTwoCarsDark`).** A second lit car starts 24 m behind
the first. Once the pedestrian is in place, its speed is set every tick so
that its inner headlamp lines up with them a quarter of the way across
(`line_up` 0.25). That keeps it just a few metres behind the pedestrian as
they step out: about 6 m at an ego speed of 8 m/s, 3 m at 4 m/s, never under
2 m. Its glare covers the wait and the step-out itself. In the offline sim, at
`--glare-size 2` and ego speeds of 3–12 m/s, the pedestrian is inside that
glare for 1–8 s while waiting and 0.4–0.6 s of their 1.1 s crossing, and
still stops 0.4 s before impact. A lower `line_up` brings the car closer
still. For a moment mid-crossing they stand in front of the inner lamp. The
injector then skips that lamp's glare (a covered lamp does not glare), and
the outer lamp still glares.

**Bigger glare.** `python run.py run ... --glare-size 2` makes every part of
the glare twice as large: the zone where a pedestrian keeps under half their
contrast grows from about 37 px to 80 px at 25 m, and from 50 to 108 px at
15 m. At 3 the glare starts dimming the ego's lane centre itself. Runs are
named `..._glare1x2_...`. It applies to any run, Bench2Drive's too; the
default is 1.

**`GlareOncomingOvertake`** — an oncoming car pulls out of another car's glare,
into the ego's lane. A slow lit car (5 m/s) comes toward the ego, with a
second lit car catching up behind it, hidden in its glare. When the ego is
75 m away, the second car pulls out to overtake, head-on in the ego's lane, at
up to 15 m/s, and pulls back in once past. An ego that brakes keeps about
20 m clear; one at 8 m/s that doesn't react misses it by about 2 m.

**Head-on (`OvertakeHeadOn`, `OvertakeHeadOnMoto`).** The same, but the
overtaker pulls out later, at 55 m (the lead is then about 45 m away), and
stays in the ego's lane until it is past the ego (`pull_back="ego"`): a
head-on course. It pulls out only once the ego is that close: if it catches
up with the lead sooner (a slow ego), it follows the lead 5 m behind until
then. It does not bail out, and its path runs past the ego, so it hits an
ego that stays in its lane, even one that brakes to a stop: about 2.7–4.5 s
after the swing at ego speeds of 2–8 m/s in the offline sim. A late bail-out
is planned for when DeepSight drives.

Seen from the ego, perspective puts the follower's inner headlamp just beside
the lead's inner lamp while it follows. In the offline sim at
`--glare-size 2`, it sits inside the lead's glare for 2–3 s before the
swing, then pulls out of it. The motorcycle (`follower="motorcycle"`) has
one central headlamp: it stays behind the lead's body until it pulls out,
then emerges into the glare. With `follower_lights="off"` the overtaker
drives dark and does not glare itself, so only the lead's glare and the
street lights show it. Use it with the street lights on: with them off too,
darkness hides it in the clean frame as well, which is not glare.

**Dark head-on (`OvertakeHeadOnDark`, `OvertakeHeadOnMotoDark`).** The two
head-on routes with the street lights off, as in `PedestrianDark`: the road is
lit only by headlights. The overtaker keeps its lights on, so the clean frame
still shows it by its own lamps; the glared frame merges them into the lead's
glare. Compare each with its lit twin.

Variants are route-file parameters, no code changes needed:

| Parameter | Scenario | Default | Other values |
| --- | --- | --- | --- |
| `adversary` | pedestrian | `walker` | `bicycle` (rides straight across) |
| `release_dist` | pedestrian | 30 m | the car's distance to the ego when the pedestrian appears |
| `behind`, `ped_offset` | pedestrian | 0.5 m, `auto` (0.15 m clear of the car's side) | where they appear |
| `speed`, `arrive_before` | pedestrian | 2.5 m/s, 0.5 s | how fast they cross; how long before impact they stop in the ego's lane |
| `hold`, `car_speed` | pedestrian | 3 s, 9 m/s | |
| `second_gap`, `second_dist`, `line_up` | pedestrian | 0 (none), `auto`, 0.25 | a second lit car this far behind the first; its distance to the ego at the step-out; how far across the pedestrian is when its lamp lines up with them (lower = closer behind them) |
| `street_lights` | both | `on` | `off`: no street lights along the stretch, lit only by headlights |
| `lead_speed`, `pass_speed` | overtake | 5, 15 m/s | |
| `swing_dist`, `pull_out_gap`, `catch_gap`, `return_gap` | overtake | 75, 5, 25, 3 m | |
| `pull_back` | overtake | `lead` (back in once past the lead) | `ego`: stays in the ego's lane until past the ego, head-on |
| `follower` | overtake | `car` | `motorcycle` |
| `follower_lights` | overtake | `on` | `off`: the overtaker drives dark and does not glare |
| `beams` | both | `low` | `high` (the lit cars') |

The full list, with what each means, is in each class's docstring in
`glare_popout.py`.

## Run (on the Ubuntu machine)

Once, to write the routes (reads the maps' OpenDRIVE files, no CARLA server):

```bash
python make_routes.py
```

It prints each spot it picked, two per town in Town01, 02, 03, 05 and 10HD.
Each spot gets one route per variant:

| Ids | Variant | |
| --- | --- | --- |
| 91001+ | `PedestrianDark` | the pedestrian with street lights off, lit only by headlights (the hard case) |
| 92001+ | `Overtake` | the oncoming overtake |
| 93001+ | `PedestrianLit` | the pedestrian under Bench2Drive's street lights (the easier case) |
| 94001+ | `PedestrianTwoCarsDark` | street lights off, plus a second lit car whose glare covers the step-out |
| 95001+ | `OvertakeHeadOn` | a car pulls out of the lead's glare onto a head-on course (edit `follower_lights` to `off` for a dark one) |
| 96001+ | `OvertakeHeadOnMoto` | the same with a motorcycle |
| 97001+ | `OvertakeHeadOnDark` | `OvertakeHeadOn` with the street lights off |
| 98001+ | `OvertakeHeadOnMotoDark` | `OvertakeHeadOnMoto` with the street lights off |

Then, from `../bench2drive_scenarios`:

```bash
python run.py run glare_popout --glare 1.0 --glare-size 2 --headlights scene
```

```bash
python run.py run glare_popout --glare 0 --headlights scene
```

The second run is the clean baseline: same routes, no glare. `--headlights
scene` keeps the background traffic as Bench2Drive lights it (mostly dark), so
the scenario's cars are the only glare. One route is a family member like any
other: `python run.py run 91001`.

Each run folder is laid out like a Bench2Drive run, plus these label files:
- `scenario_events.jsonl`: one line per moment (`street_lights_off`,
  `popout_start`, `second_car_at_mark`, `popout_walk`, `popout_in_lane`, `popout_clear`, `overtake_swing`,
  `overtake_return`) with its CARLA frame, which matches the `frame` in
  `meta/<step>.json`. That is where the hazard is in the video.
- `road_user_glare.jsonl`: for every step with a pedestrian or cyclist in
  CAM_FRONT, how much the glare washes them out. The segmentation camera gives
  their exact pixels. For each pixel, `cover` is how far the glare pushed it
  toward white (0 = untouched, 1 = white), measured on the saved JPEGs
  DeepSight reads. The line records the pixel count, the mean cover, the share
  of pixels at least half way to white (`washed`), and `hidden` (half or more
  washed). It needs glare on, since the segmentation camera only runs then.
- `hazard_glare.jsonl`: the same for the overtaker, on every step it is in
  front of CAM_FRONT within 70 m, lights on or off. `footprint_px` is the
  area of its projected box, and `pixels` is how much of it the camera sees
  (segmentation plus matching depth, so the lead car in front of it is not
  counted). Few `pixels` means the lead's body hides it; `hidden` means what
  the camera does see is mostly washed out by glare. The line also records
  its distance and whether its lights are on.
- Both files also record `clean_contrast` and `glared_contrast`: how much the
  road user or vehicle stands out from the 8 px ring around it, in the clean
  and the glared frame (0 = not at all). `visible_clean` means it stands out
  in the clean frame (contrast at least 0.1); when it does not, darkness hides
  it, not glare. `glare_occluded` means it is visible clean but hidden, or no
  longer standing out, once glared: the frames that are glare occlusion.

`python run.py summary` adds a `hidden` column, steps with a hidden
pedestrian or cyclist out of steps with one in view, and a `hazard` column:
steps the overtaker is `glare_occluded` out of steps it is visible in the clean
frame. A motorcycle's rider also counts in
`road_user_glare.jsonl`, as a rider.

NpcAgent ignores pedestrians, so it drives into the pedestrian in every run.
It brakes for a vehicle in its lane, but on the head-on routes that is not
enough, so those runs usually end in a collision too. This is expected: for
now the runs are about the images. Brakes and
collisions mean something once DeepSight drives.

## How it plugs into Bench2Drive

The leaderboard loads scenario classes from every
`$SCENARIO_RUNNER_ROOT/srunner/scenarios/*.py`. For a route of ours, `run.py`
builds a temporary folder with links to Bench2Drive's scenario files plus
`glare_popout.py`, and points `SCENARIO_RUNNER_ROOT` at it. Bench2Drive's
files stay unedited. DeepSight's agent will run these routes the same way.

## Branching off

A new scenario is a new class in `glare_popout.py` built from the same
pieces:
- `_GlareScenario`: the meet point M on a two-way road, and road position `s`
- `OncomingCar`: a lit car, optionally timed to be at a place when the ego is
- `HAZARD_ROLE`: spawn a vehicle with it and the glare agent labels how hidden it is
- `LogEvent`: a frame-stamped moment for the dataset

Then add the class and its default parameters to `SCENARIOS` in
`make_routes.py`.

## Still to build

| # | Scenario | Why Bench2Drive can't | What to build | Start from |
| --- | --- | --- | --- | --- |
| 7 | Sudden high beams on a bend | Traffic Manager picks the beams, usually low; a scenario cannot request high beams | A route where one car appears around a bend at 30–50 m with `beams=high` | Town07 rural bends (`spawn_points.py` to pick them), rural Town12 |
| 8 | Dark-road Bench2Drive routes | Bench2Drive turns street lights on near the ego at night | Our scenarios have the switch (`street_lights="off"`); the Bench2Drive families still need it, e.g. from the agent | `StreetLightsOff` in `glare_popout.py` |
| 10 | Sun glare at dawn or dusk | The glare code only uses headlights as light sources | A sun light source in the glare code, from the weather's sun angles; future work | — |
