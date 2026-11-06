"""Something pops out of oncoming headlight glare: Bench2Drive leaderboard scenarios.

Two scenarios on a straight two-way road at night. In each, one lit oncoming
car is the glare source, timed so the hazard appears beside or behind its
headlamps as seen from the ego's front camera:

GlarePedestrianPopOut
    A lit oncoming car drives toward the ego. When it is `release_dist` metres
    from the ego, a pedestrian (or cyclist) appears just behind its rear
    bumper, right beside its flank (between the car and the centre line). Seen
    from CAM_FRONT they appear next to the car's inner headlamp, inside its
    glare. They wait there, then hurry into the ego's lane, timed from the
    ego's speed to stop right in front of it `arrive_before` seconds before
    the ego reaches them.

    The glare only takes a pedestrian's contrast away within ~40 px of a lamp
    (measured with glare(): contrast against the local brightness drops to
    0.2-0.6 there, and is nearly back by 80 px). The pedestrian is that close
    to the lamp only while the car is 25-40 m away; nearer, the lamps sweep to
    the edge of the image. Hence release_dist 30 and a start tight against the
    car. The release follows the car, not the ego, so it lines up at any ego
    speed. The clean frame shows the pedestrian throughout; only the glare
    hides them. A larger glare (run.py --glare-size 2) widens that zone to
    ~80 px at 25 m.

    With second_gap > 0 a second lit car follows the first. It is timed so
    its inner lamp lines up with the pedestrian a quarter of the way across
    (line_up), a few metres behind them, so its glare covers the wait and the
    step-out itself, which the first car, by then beside the ego, cannot.

GlareOncomingOvertake
    A slow lit oncoming car with a second vehicle close behind it, hidden in
    its glare. When the ego is within `swing_dist` of the second one, it pulls
    out into the ego's lane to overtake, head-on, and pulls back in once past
    the lead, or (pull_back="ego") only once past the ego: a head-on course
    the ego must avoid. The second vehicle is a car or a motorcycle (one
    headlamp, narrow, easily lost beside the lead's lamps), with its lights on
    or off.

The oncoming cars drive a fixed plan with their own headlights on (low or
high beams), so they glare under any --headlights mode, and the role name
'scenario no lights' keeps Bench2Drive's lights behaviour from switching them
off. With street_lights="off" the street lights along the stretch stay off
(Bench2Drive switches them on near the ego at night), so the road is lit only
by headlights. With glare 0 the same run is the clean baseline: any change in
the ego's behaviour is the glare's doing.

Every parameter can be set in the route file, as Bench2Drive's scenarios do
(<release_dist value="20"/>); custom_scenarios/make_routes.py writes route
files with all of them. Each timing moment (pop-out start, in the ego's lane,
overtake swing and return) is appended to $GLARE_OUTPUT/scenario_events.jsonl
with its CARLA frame, which matches meta/<step>.json of the run.

The leaderboard finds scenario classes in $SCENARIO_RUNNER_ROOT/srunner/
scenarios/*.py; bench2drive_scenarios/run.py links this file in there for the
run, so Bench2Drive's own files stay unedited.
"""
import json
import math
import os

import carla
import py_trees

from agents.navigation.local_planner import LocalPlanner, RoadOption

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorDestroy,
                                                                      AtomicBehavior,
                                                                      Idle,
                                                                      KeepVelocity,
                                                                      MovePedestrianWithEgo)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (DriveDistance,
                                                                               InTriggerDistanceToVehicle)
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.background_manager import ChangeOppositeBehavior, ChangeRoadBehavior, LeaveSpaceInFront

BEAMS = {
    'low': carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam,
    'high': (carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam
             | carla.VehicleLightState.HighBeam),
}
# Bench2Drive's RouteLightsBehavior switches the lights of 'scenario' cars on
# and off by distance; it leaves this role alone, so the cars keep our lights.
ROLE = 'scenario no lights'
# The same, marking the vehicle the glare agent labels (hazard_glare.jsonl).
HAZARD_ROLE = 'scenario no lights hazard'
CAR_MODEL = 'vehicle.lincoln.mkz_2017'   # the oncoming model of carla_script_closed_loop.py
BICYCLE_MODEL = 'vehicle.diamondback.century'
# BackgroundActivity keeps this many of its cars ahead of the ego in its lane;
# our scenarios set it to 0 while they run and put it back after.
BA_FRONT_VEHICLES = 2
# street_lights="off" darkens every street light this close to the meet point,
# which covers the whole stretch make_routes.py picks.
DARK_RADIUS_M = 130.0


def get_value_parameter(config, name, p_type, default):
    if name in config.other_parameters:
        return p_type(config.other_parameters[name]['value'])
    return default


def along(wp, dist):
    """The waypoint `dist` metres further along wp's lane (behind it if negative)."""
    if abs(dist) < 0.01:
        return wp
    found = wp.next(dist) if dist > 0 else wp.previous(-dist)
    if not found:
        raise ValueError('lane ends within %.0f m of %s' % (abs(dist), wp.transform.location))
    return found[0]


def lane_plan(wp, length, step=2.0):
    """wp's lane ahead of it, every `step` metres for `length` metres, for a LocalPlanner."""
    plan = [wp]
    while len(plan) * step < length:
        found = plan[-1].next(step)
        if not found:
            break
        plan.append(found[0])
    return [(w, RoadOption.LANEFOLLOW) for w in plan]


def oncoming_lane(wp):
    """The oncoming lane beside wp; the scenarios need a two-way road."""
    left = wp.get_left_lane()
    if left is None or left.lane_type != carla.LaneType.Driving or left.lane_id * wp.lane_id > 0:
        raise ValueError('no oncoming lane left of the ego lane at %s: these scenarios need a '
                         'two-way road (make_routes.py finds them)' % wp.transform.location)
    return left


def motorcycle_model():
    """A motorcycle blueprint, one with modelled lights if there is one (always the same pick)."""
    bikes = sorted((bp for bp in CarlaDataProvider.get_world().get_blueprint_library().filter('vehicle.*')
                    if bp.has_attribute('base_type') and bp.get_attribute('base_type').as_str() == 'motorcycle'),
                   key=lambda bp: (not (bp.has_attribute('has_lights')
                                        and bp.get_attribute('has_lights').as_bool()), bp.id))
    if not bikes:
        raise ValueError('no motorcycle blueprint in this CARLA build')
    return bikes[0].id


def planner(vehicle, max_throttle=0.75):
    """A LocalPlanner tuned for the leaderboard's tick rate (10 Hz for DeepSight)."""
    dt = CarlaDataProvider.get_world().get_settings().fixed_delta_seconds or 0.1
    return LocalPlanner(vehicle, opt_dict={
        'dt': dt, 'max_throttle': max_throttle, 'max_brake': 0.5,
        'lateral_control_dict': {'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.2, 'dt': dt},
        'longitudinal_control_dict': {'K_P': 1.0, 'K_I': 0.05, 'K_D': 0, 'dt': dt},
    }, map_inst=CarlaDataProvider.get_map())


def log_event(event, **data):
    """Append a timing moment to $GLARE_OUTPUT/scenario_events.jsonl (no-op without it)."""
    out = os.environ.get('GLARE_OUTPUT')
    if not out:
        return
    entry = {'event': event, 'frame': CarlaDataProvider.get_world().get_snapshot().frame,
             'time_s': round(GameTime.get_time(), 2)}
    entry.update(data)
    with open(os.path.join(out, 'scenario_events.jsonl'), 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(entry) + '\n')


class LogEvent(AtomicBehavior):
    """Logs one event (with data_fn()'s numbers) and succeeds."""

    def __init__(self, event, data_fn=None, name='LogEvent'):
        super().__init__(name)
        self._event = event
        self._data_fn = data_fn

    def update(self):
        log_event(self._event, **(self._data_fn() if self._data_fn else {}))
        return py_trees.common.Status.SUCCESS


class OncomingCar(AtomicBehavior):
    """Drives a lit car down its plan at `speed` m/s and keeps its headlights on.

    Never finishes on its own.
    """

    def __init__(self, actor, plan, speed, lights, name='OncomingCar'):
        super().__init__(name, actor)
        self._plan = plan
        self._speed = speed
        self._lights = lights
        self._planner = None

    def initialise(self):
        self._planner = planner(self._actor)
        self._planner.set_global_plan(self._plan)
        super().initialise()

    def _target_speed(self):
        return self._speed

    def update(self):
        self._actor.set_light_state(carla.VehicleLightState(self._lights))
        self._planner.set_speed(self._target_speed() * 3.6)
        self._actor.apply_control(self._planner.run_step())
        return py_trees.common.Status.RUNNING


class TimedOncomingCar(OncomingCar):
    """An OncomingCar timed to be `target_dist()` metres from the ego at a moment.

    `seconds_left()` says how far off the moment is, or None while it is not
    known yet (the car then drives at `speed`). Once known, the speed is set
    every tick so the car closes the remaining distance in that time, given
    the ego's current speed; after the moment it keeps its last speed.
    """

    def __init__(self, actor, plan, speed, lights, ego, target_dist, seconds_left,
                 speed_range=(2.0, 15.0), name='TimedOncomingCar'):
        super().__init__(actor, plan, speed, lights, name=name)
        self._ego = ego
        self._target_dist = target_dist
        self._seconds_left = seconds_left
        self._speed_range = speed_range
        self._timed = False

    def _target_speed(self):
        if self._timed:
            return self._speed
        seconds = self._seconds_left()
        if seconds is None:
            return self._speed
        dist = CarlaDataProvider.get_location(self._actor).distance(CarlaDataProvider.get_location(self._ego))
        if seconds <= 0.1:
            self._timed = True
            log_event('second_car_at_mark', car_to_ego_m=round(dist, 1), target_m=round(self._target_dist(), 1))
            return self._speed
        closing = (dist - self._target_dist()) / seconds
        self._speed = min(max(closing - CarlaDataProvider.get_velocity(self._ego), self._speed_range[0]),
                          self._speed_range[1])
        return self._speed


class StreetLightsOff(AtomicBehavior):
    """Keeps the street lights near `location` off. Never finishes on its own.

    Bench2Drive's RouteLightsBehavior switches on, every tick, the street
    lights near the ego at night. It ticks before the scenarios do, so
    switching them off again every tick wins for the frame that gets rendered.
    """

    def __init__(self, location, radius, name='StreetLightsOff'):
        super().__init__(name)
        self._location = location
        self._radius = radius
        self._manager = None
        self._lights = []

    def initialise(self):
        self._manager = CarlaDataProvider.get_world().get_lightmanager()
        self._lights = [light for light in self._manager.get_all_lights(carla.LightGroup.Street)
                        if light.location.distance(self._location) < self._radius]
        log_event('street_lights_off', lights=len(self._lights), radius_m=self._radius)
        super().initialise()

    def update(self):
        if self._lights:
            self._manager.turn_off(self._lights)
        return py_trees.common.Status.RUNNING


class StepIntoPath(AtomicBehavior):
    """Waits, then crosses `distance` metres into the ego's path, timed to arrive just ahead of it.

    It sets off once the ego's time to reach it (gap / ego speed) is down to its
    own crossing time plus `arrive_before` seconds, so an ego that keeps its
    speed finds it stopped right in front, `arrive_before` seconds before
    impact. It sets off anyway after `max_wait` seconds, for an ego that has
    stopped. It moves along its heading at `speed` m/s and succeeds on
    arrival: a walker standing still, a bicycle still riding.
    """

    def __init__(self, actor, ego, s_of, distance, speed, arrive_before, max_wait=8.0,
                 name='StepIntoPath'):
        super().__init__(name, actor)
        self._ego = ego
        self._s_of = s_of
        self._distance = distance
        self._speed = speed
        self._arrive_before = arrive_before
        self._max_wait = max_wait
        self._walker = 'walker' in actor.type_id
        self._start = self._origin = self._direction = None

    def initialise(self):
        self._start = GameTime.get_time()
        self._origin = None
        fwd = CarlaDataProvider.get_transform(self._actor).get_forward_vector()
        self._direction = carla.Vector3D(fwd.x, fwd.y, 0.0)
        super().initialise()

    def _move(self, speed):
        if self._walker:
            self._actor.apply_control(carla.WalkerControl(direction=self._direction, speed=speed))
        else:
            self._actor.set_target_velocity(self._direction * speed)
            self._actor.apply_control(carla.VehicleControl(throttle=1.0))    # makes the rider pedal

    def _ego_time_to_reach(self):
        """(seconds until the ego's front reaches this actor at its speed, the gap, the ego's speed)."""
        ego = CarlaDataProvider.get_location(self._ego)
        ego_speed = CarlaDataProvider.get_velocity(self._ego)
        gap = (self._s_of(CarlaDataProvider.get_location(self._actor)) - self._s_of(ego)
               - self._ego.bounding_box.extent.x)
        return gap / max(ego_speed, 0.1), gap, ego_speed

    def seconds_to_start(self):
        """Seconds until it sets off at the ego's current speed; None before it is in place."""
        if self._start is None:
            return None
        if self._origin is not None:
            return 0.0
        ttc = self._ego_time_to_reach()[0]
        wait_left = self._max_wait - (GameTime.get_time() - self._start)
        return max(min(ttc - self._distance / self._speed - self._arrive_before, wait_left), 0.0)

    def update(self):
        location = CarlaDataProvider.get_location(self._actor)
        if self._origin is None:
            ttc, gap, ego_speed = self._ego_time_to_reach()
            waited = GameTime.get_time() - self._start
            if ttc > self._distance / self._speed + self._arrive_before and waited < self._max_wait:
                return py_trees.common.Status.RUNNING
            self._origin = location
            log_event('popout_walk', ego_time_to_reach_s=round(ttc, 2), gap_m=round(gap, 1),
                      ego_speed_ms=round(ego_speed, 1), waited_s=round(waited, 1))
        if location.distance(self._origin) >= self._distance:
            if self._walker:
                self._move(0.0)
            return py_trees.common.Status.SUCCESS
        self._move(self._speed)
        return py_trees.common.Status.RUNNING


class PlaceBehindCar(AtomicBehavior):
    """Puts the hidden adversary just behind `car`, facing across the ego's lane, then succeeds.

    `place(car_location)` gives the transform. Physics comes back on once the
    adversary is there, as Bench2Drive's ActorTransformSetter does.
    """

    def __init__(self, actor, car, place, name='PlaceBehindCar'):
        super().__init__(name, actor)
        self._car = car
        self._place = place
        self._transform = None

    def initialise(self):
        self._transform = self._place(CarlaDataProvider.get_location(self._car))
        self._actor.set_target_velocity(carla.Vector3D(0, 0, 0))
        self._actor.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
        self._actor.set_transform(self._transform)
        super().initialise()

    def update(self):
        location = CarlaDataProvider.get_location(self._actor)
        if location is None or location.distance(self._transform.location) > 1.0:
            self._actor.set_transform(self._transform)
            return py_trees.common.Status.RUNNING
        self._actor.set_simulate_physics(True)
        return py_trees.common.Status.SUCCESS


class OncomingOvertaker(AtomicBehavior):
    """A lit car that catches up with `lead` in the oncoming lane and overtakes it in the ego's lane.

    It starts behind the lead and closes in, its speed set every tick so that it
    is `pull_out_gap` metres behind the lead (inside its glare, seen from the
    ego) when the ego is `swing_dist` away; if it gets there first it follows
    the lead at that gap. Only then, however slow the ego, it pulls out into the ego's
    lane at `pass_speed`, already moving faster than the lead, as a real
    overtake does. With pull_back 'lead' it pulls back into its own lane
    `return_gap` metres ahead of the lead, or at once if it is already past
    the ego. With 'ego' it stays in the ego's lane, head-on, until it is past
    the ego: its plan runs past the ego, so it reaches it even if the ego
    stops. Never finishes on its own.
    """

    CHANGE_M = 10.0      # metres travelled while changing lane

    def __init__(self, actor, lead, ego, s_of, pull_out_gap, swing_dist, pass_speed, return_gap,
                 lights, pull_back='lead', name='OncomingOvertaker'):
        super().__init__(name, actor)
        self._lead = lead
        self._ego = ego
        self._s_of = s_of
        self._pull_out_gap = pull_out_gap
        self._swing_dist = swing_dist
        self._pass_speed = pass_speed
        self._return_gap = return_gap
        self._lights = lights
        self._pull_back = pull_back
        self._map = CarlaDataProvider.get_map()
        self._phase = 'approach'
        self._planner = None

    def initialise(self):
        self._planner = planner(self._actor, max_throttle=1.0)
        self._planner.set_global_plan(lane_plan(self._here(), 200.0)[2:])
        super().initialise()

    def _here(self):
        return self._map.get_waypoint(CarlaDataProvider.get_location(self._actor))

    def _change_lane(self, into_ego_lane, length=120.0):
        """Plan from here into the other lane, then `length` m along it in this car's direction."""
        here = self._here()
        if into_ego_lane:
            target = along(here, self.CHANGE_M).get_left_lane()      # oncoming lane's left = ego lane
            if target is None:
                return
            plan = [target]
            for _ in range(int(length / 2.0)):                      # the ego lane runs the other way
                found = plan[-1].previous(2.0)
                if not found:
                    break
                plan.append(found[0])
            plan = [(w, RoadOption.CHANGELANELEFT) for w in plan]
        else:
            back = here.get_left_lane()                             # ego lane's left = oncoming lane
            if back is None:
                return                                              # keep the current plan
            plan = lane_plan(along(back, self.CHANGE_M), 120.0)
        self._planner.set_global_plan(plan)

    def update(self):
        self._actor.set_light_state(carla.VehicleLightState(self._lights))
        loc = CarlaDataProvider.get_location(self._actor)
        ego_loc = CarlaDataProvider.get_location(self._ego)
        lead_loc = CarlaDataProvider.get_location(self._lead)
        lead_speed = CarlaDataProvider.get_velocity(self._lead)
        # Both drive toward decreasing s; `ahead` > 0 once this car is past the lead.
        ahead = self._s_of(lead_loc) - self._s_of(loc)
        lengths = self._actor.bounding_box.extent.x + self._lead.bounding_box.extent.x

        if self._phase == 'approach':
            gap = -ahead - lengths                  # bumper to bumper, behind the lead
            ego_dist = loc.distance(ego_loc)
            if ego_dist <= self._swing_dist:
                self._phase = 'pass'
                # Head-on, the plan must reach past the ego: a planner stops where its plan ends.
                length = ego_dist + 60.0 if self._pull_back == 'ego' else 120.0
                self._change_lane(into_ego_lane=True, length=max(length, 120.0))
                log_event('overtake_swing', ego_dist=round(ego_dist, 1), lead_gap=round(gap, 1),
                          lead_to_ego=round(lead_loc.distance(ego_loc), 1),
                          speed=round(CarlaDataProvider.get_velocity(self._actor), 1))
                speed = self._pass_speed
            else:
                # A standing ego is assumed to pull away; the next ticks correct for it.
                closing = (max(CarlaDataProvider.get_velocity(self._ego), 4.0)
                           + max(CarlaDataProvider.get_velocity(self._actor), lead_speed))
                seconds = max(ego_dist - self._swing_dist, 1.0) / closing
                if gap > self._pull_out_gap:
                    speed = min(max(lead_speed + (gap - self._pull_out_gap) / seconds, lead_speed),
                                self._pass_speed)
                else:                               # caught up early: follow the lead at the gap
                    speed = max(lead_speed + (gap - self._pull_out_gap), 0.0)
        elif self._phase == 'pass':
            speed = self._pass_speed
            past_ego = self._s_of(loc) < self._s_of(ego_loc)
            past_lead = self._pull_back == 'lead' and ahead - lengths >= self._return_gap
            if past_lead or past_ego:
                self._phase = 'back'
                self._change_lane(into_ego_lane=False)
                log_event('overtake_return', ego_dist=round(loc.distance(ego_loc), 1),
                          past_ego=past_ego)
        else:
            speed = self._pass_speed

        self._planner.set_speed(speed * 3.6)
        self._actor.apply_control(self._planner.run_step())
        return py_trees.common.Status.RUNNING


class _GlareScenario(BasicScenario):
    """Shared layout: the meet point M is `distance` metres past the trigger point."""

    DISTANCE = 90.0

    def __init__(self, name, world, ego_vehicles, config, debug_mode, criteria_enable, timeout):
        self._map = CarlaDataProvider.get_map()
        trigger_wp = self._map.get_waypoint(config.trigger_points[0].location)
        self._distance = get_value_parameter(config, 'distance', float, self.DISTANCE)
        self._meet_wp = along(trigger_wp, self._distance)
        self._oncoming_wp = oncoming_lane(self._meet_wp)
        beams = get_value_parameter(config, 'beams', str, 'low')
        if beams not in BEAMS:
            raise ValueError("'beams' must be 'low' or 'high', not %r" % beams)
        self._lights = BEAMS[beams]
        self._street_lights = get_value_parameter(config, 'street_lights', str, 'on')
        if self._street_lights not in ('on', 'off'):
            raise ValueError("'street_lights' must be 'on' or 'off', not %r" % self._street_lights)
        self.timeout = timeout
        super().__init__(name, ego_vehicles, config, world, debug_mode, criteria_enable=criteria_enable)

    def _s(self, location):
        """Metres along the ego's direction from the meet point."""
        m = self._meet_wp.transform
        f = m.get_forward_vector()
        return (location.x - m.location.x) * f.x + (location.y - m.location.y) * f.y

    def _oncoming_at(self, s):
        """The oncoming lane at road position s (it runs toward decreasing s)."""
        return along(self._oncoming_wp, -s)

    def _spawn_car(self, s, model, role=ROLE, lights=None):
        """A vehicle in the oncoming lane at s, moved further out if the spot is taken."""
        for shift in (0.0, 5.0, 10.0, 15.0):
            wp = self._oncoming_at(s + shift)
            car = CarlaDataProvider.request_new_actor(model, wp.transform, rolename=role)
            if car is not None:
                car.set_light_state(carla.VehicleLightState(self._lights if lights is None else lights))
                return car, s + shift
        raise ValueError('could not spawn %s in the oncoming lane near s=%.0f m' % (model, s))

    def _car_plan(self, s):
        """The oncoming lane from s to 60 m behind the trigger point."""
        return lane_plan(self._oncoming_at(s), s + self._distance + 60.0)

    def _add_street_lights(self, parallel):
        """With street_lights="off", keep the stretch's street lights off while `parallel` runs."""
        if self._street_lights == 'off':
            parallel.add_child(StreetLightsOff(self._meet_wp.transform.location, DARK_RADIUS_M))

    def _create_test_criteria(self):
        if self.route_mode:
            return []
        return [CollisionTest(self.ego_vehicles[0])]

    def __del__(self):
        self.remove_all_actors()


class GlarePedestrianPopOut(_GlareScenario):
    """A pedestrian or cyclist steps into the ego's lane from just behind an oncoming car, inside its glare.

    Parameters (route file, all optional):
        distance       trigger point to the meet point M, m (90)
        car_ahead      where the car waits, m past M (35)
        car_speed      the car's speed, m/s (9)
        release_dist   the car's distance to the ego when the adversary appears, m (30)
        behind         how far behind the car's rear bumper it appears, m (0.5)
        ped_offset     how far past the centre line, toward the car, it appears, m
                       ('auto': 0.15 m clear of the car's flank)
        adversary      'walker' or 'bicycle' (walker)
        speed          the adversary's speed, m/s (2.5 walker, a brisk jog; 3.5 bicycle)
        arrive_before  seconds before the ego would reach it that it stops in the ego's lane (0.5)
        hold           seconds a walker stands in the ego's lane before walking on (3)
        second_gap     a second lit car this far behind the first at the start, m (0 = none)
        second_dist    its distance to the ego when the adversary steps out, m ('auto':
                       where its inner lamp lines up with the adversary at line_up)
        line_up        how far across the adversary is when the second car's inner lamp
                       lines up with them, 0-1 (0.25; lower brings the car closer behind them)
        street_lights  'on' (Bench2Drive's) or 'off' along the stretch (on)
        beams          the car's 'low' or 'high' beams (low)
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False,
                 criteria_enable=True, timeout=60):
        self._car_ahead = get_value_parameter(config, 'car_ahead', float, 35.0)
        self._car_speed = get_value_parameter(config, 'car_speed', float, 9.0)
        self._release_dist = get_value_parameter(config, 'release_dist', float, 30.0)
        self._behind = get_value_parameter(config, 'behind', float, 0.5)
        self._ped_offset = get_value_parameter(config, 'ped_offset', str, 'auto')
        self._kind = get_value_parameter(config, 'adversary', str, 'walker')
        self._speed = get_value_parameter(config, 'speed', float, 2.5 if self._kind == 'walker' else 3.5)
        self._arrive_before = get_value_parameter(config, 'arrive_before', float, 0.5)
        self._hold = get_value_parameter(config, 'hold', float, 3.0)
        self._second_gap = get_value_parameter(config, 'second_gap', float, 0.0)
        self._second_dist = get_value_parameter(config, 'second_dist', str, 'auto')
        self._line_up = min(max(get_value_parameter(config, 'line_up', float, 0.25), 0.0), 0.9)
        if self._kind not in ('walker', 'bicycle'):
            raise ValueError("'adversary' must be 'walker' or 'bicycle', not %r" % self._kind)
        super().__init__('GlarePedestrianPopOut', world, ego_vehicles, config, debug_mode,
                         criteria_enable, timeout)

    def _lateral(self):
        """The adversary's start, in metres left of the ego lane's centre (toward the car)."""
        to_line = self._meet_wp.lane_width / 2.0
        if self._ped_offset != 'auto':
            return to_line + float(self._ped_offset)
        flank = to_line + self._oncoming_wp.lane_width / 2.0 - self._car.bounding_box.extent.y
        return max(flank - 0.15, to_line)

    def _place(self, car_location):
        """Where the adversary appears, given the car's location: behind it, facing the ego's lane."""
        s = self._s(car_location) + self._car.bounding_box.extent.x + self._behind
        lane = along(self._meet_wp, s).transform              # the ego's lane, beside that spot
        right = lane.get_right_vector()                       # from the car's side into the ego's lane
        left = self._lateral()
        location = carla.Location(x=lane.location.x - right.x * left, y=lane.location.y - right.y * left,
                                  z=lane.location.z + (1.0 if self._kind == 'walker' else 0.3))
        return carla.Transform(location, carla.Rotation(yaw=math.degrees(math.atan2(right.y, right.x))))

    def _initialize_actors(self, config):
        self._car, self._car_s = self._spawn_car(self._car_ahead, CAR_MODEL)
        self._second = None
        if self._second_gap > 0:
            behind = self._car_s + 2 * self._car.bounding_box.extent.x + self._second_gap
            self._second, self._second_s = self._spawn_car(behind, CAR_MODEL)
            self.other_actors.append(self._second)
        # Spawn the adversary once at a spot like the one it will appear at, to
        # check it can stand there, then hide it until the release.
        first = self._place(self._oncoming_at(-self._release_dist).transform.location)
        model = 'walker.pedestrian.*' if self._kind == 'walker' else BICYCLE_MODEL
        adversary = CarlaDataProvider.request_new_actor(model, first, rolename=ROLE)
        if adversary is None:
            raise ValueError('could not spawn the %s at %s' % (self._kind, first.location))
        if self._kind == 'walker':
            # Bench2Drive's fix for walkers on large maps (DynamicObjectCrossing):
            # respawn it under the ego and keep it there until it appears.
            type_id = adversary.type_id
            adversary.destroy()
            hidden = self.ego_vehicles[0].get_transform()
            hidden.location.z -= 50
            adversary = CarlaDataProvider.request_new_actor(type_id, hidden, rolename=ROLE)
            if adversary is None:
                raise ValueError("couldn't spawn the walker substitute")
        else:
            adversary.set_transform(carla.Transform(first.location + carla.Location(z=-200), first.rotation))
        adversary.set_simulate_physics(False)
        self._adversary = adversary
        self.other_actors += [self._adversary, self._car]

    def _second_mark(self):
        """How far from the ego the second car should be as the adversary steps out.

        With 'auto': far enough that, seen from the ego, its inner headlamp lines
        up with the adversary when they are line_up of the way across. The
        adversary steps out about (crossing time + arrive_before) x ego speed
        ahead of the ego, which is why the mark moves with the ego's speed.
        Lining up early (0.25) keeps the car a few metres behind them; at 0 it
        would reach them as they set off.
        """
        if self._second_dist != 'auto':
            return float(self._second_dist)
        ego_speed = CarlaDataProvider.get_velocity(self.ego_vehicles[0])
        walk_s = self._lateral() / self._speed
        depth = ego_speed * (walk_s + self._arrive_before) + self.ego_vehicles[0].bounding_box.extent.x
        lamp = (self._meet_wp.lane_width / 2.0 + self._oncoming_wp.lane_width / 2.0
                - 0.6)                                    # inner lamp, ~0.6 m in from the car's centre line
        # Line up line_up x walk_s after the step-out, then add how far both close in meanwhile.
        f = self._line_up
        closing = (ego_speed + CarlaDataProvider.get_velocity(self._second)) * f * walk_s
        mark = lamp * (depth - ego_speed * f * walk_s) / (self._lateral() * (1.0 - f)) + closing
        # Never closer than 4 m behind the adversary as they set off.
        behind = depth + self._second.bounding_box.extent.x + 4.0
        return min(max(mark, behind), 45.0)

    def _state(self):
        ego = CarlaDataProvider.get_location(self.ego_vehicles[0])
        car = CarlaDataProvider.get_location(self._car)
        adversary = CarlaDataProvider.get_location(self._adversary)
        state = {'car_to_ego_m': round(car.distance(ego), 1),
                 'adversary_to_ego_m': round(adversary.distance(ego), 1),
                 'ego_speed_ms': round(CarlaDataProvider.get_velocity(self.ego_vehicles[0]), 1)}
        if self._second is not None:
            state['second_car_to_ego_m'] = round(CarlaDataProvider.get_location(self._second).distance(ego), 1)
        return state

    def _create_behavior(self):
        ego = self.ego_vehicles[0]
        root = py_trees.composites.Sequence(name='GlarePedestrianPopOut')
        if self.route_mode:
            root.add_child(LeaveSpaceInFront(self._distance + 20.0))
            # No background car ahead of the ego in its lane: the hazard must meet the ego.
            root.add_child(ChangeRoadBehavior(num_front_vehicles=0))
            root.add_child(ChangeOppositeBehavior(active=False))

        main = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE,
                                            name='PopOutOfGlare')
        main.add_child(OncomingCar(self._car, self._car_plan(self._car_s), self._car_speed, self._lights))
        self._add_street_lights(main)

        # Hidden until the car is close; walkers are kept under the ego meanwhile.
        release = InTriggerDistanceToVehicle(self._car, ego, self._release_dist)
        if self.route_mode and self._kind == 'walker':
            wait = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE,
                                                name='WaitForCar')
            wait.add_child(MovePedestrianWithEgo(ego, self._adversary, 100))
            wait.add_child(release)
            release = wait

        to_centre = self._lateral()
        off_road = self._meet_wp.lane_width / 2.0 + 2.0
        step = StepIntoPath(self._adversary, ego, self._s, to_centre, self._speed, self._arrive_before)
        if self._second is not None:
            # Timed to be second_dist from the ego as the adversary steps out:
            # from there its lamps line up with the adversary mid-crossing.
            main.add_child(TimedOncomingCar(self._second, self._car_plan(self._second_s), self._car_speed,
                                            self._lights, ego, self._second_mark, step.seconds_to_start))
        popout = py_trees.composites.Sequence(name='Adversary')
        popout.add_child(release)
        popout.add_child(PlaceBehindCar(self._adversary, self._car, self._place))
        popout.add_child(LogEvent('popout_start', self._state))
        popout.add_child(step)
        popout.add_child(LogEvent('popout_in_lane', self._state))
        if self._kind == 'bicycle':      # a cyclist rides on across
            popout.add_child(KeepVelocity(self._adversary, self._speed, force_speed=True, distance=off_road))
        else:                            # a walker stands in the ego's lane, then walks off the road
            popout.add_child(Idle(self._hold))
            popout.add_child(KeepVelocity(self._adversary, self._speed, distance=off_road))
        popout.add_child(LogEvent('popout_clear', self._state))
        popout.add_child(DriveDistance(ego, 40.0))
        main.add_child(popout)
        root.add_child(main)

        if self.route_mode:
            root.add_child(ChangeOppositeBehavior(active=True))
            root.add_child(ChangeRoadBehavior(num_front_vehicles=BA_FRONT_VEHICLES))
        root.add_child(ActorDestroy(self._adversary))
        root.add_child(ActorDestroy(self._car))
        if self._second is not None:
            root.add_child(ActorDestroy(self._second))
        return root


class GlareOncomingOvertake(_GlareScenario):
    """An oncoming car or motorcycle pulls out of a slower lit car's glare into the ego's lane to overtake it.

    Parameters (route file, all optional):
        distance      trigger point to the meet point M, m (60)
        lead_ahead    where the slow lead car waits, m past M (50)
        lead_speed    the lead car's speed, m/s (5)
        catch_gap     the overtaker's starting gap behind the lead, m (25)
        pull_out_gap  its gap behind the lead when it pulls out, m (5)
        swing_dist    the ego's distance to the overtaker at that moment, m (75)
        pass_speed    the overtaker's top speed, m/s (15)
        return_gap    how far past the lead it pulls back in, m (3)
        pull_back     'lead': once past the lead; 'ego': only once past the ego, head-on (lead)
        follower      the overtaker: 'car' or 'motorcycle' (car)
        follower_lights  the overtaker's headlights, 'on' or 'off' (on); off, it does not glare
        street_lights 'on' (Bench2Drive's) or 'off' along the stretch (on)
        beams         the lit cars' 'low' or 'high' beams (low)

    The overtaker has HAZARD_ROLE, so the glare agent labels how hidden it is.
    """

    DISTANCE = 60.0

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False,
                 criteria_enable=True, timeout=60):
        self._lead_ahead = get_value_parameter(config, 'lead_ahead', float, 50.0)
        self._lead_speed = get_value_parameter(config, 'lead_speed', float, 5.0)
        self._catch_gap = get_value_parameter(config, 'catch_gap', float, 25.0)
        self._pull_out_gap = get_value_parameter(config, 'pull_out_gap', float, 5.0)
        self._swing_dist = get_value_parameter(config, 'swing_dist', float, 75.0)
        self._pass_speed = get_value_parameter(config, 'pass_speed', float, 15.0)
        self._return_gap = get_value_parameter(config, 'return_gap', float, 3.0)
        self._pull_back = get_value_parameter(config, 'pull_back', str, 'lead')
        self._follower = get_value_parameter(config, 'follower', str, 'car')
        self._follower_lights = get_value_parameter(config, 'follower_lights', str, 'on')
        for name, value, allowed in (('pull_back', self._pull_back, ('lead', 'ego')),
                                     ('follower', self._follower, ('car', 'motorcycle')),
                                     ('follower_lights', self._follower_lights, ('on', 'off'))):
            if value not in allowed:
                raise ValueError('%r must be %s, not %r' % (name, ' or '.join(map(repr, allowed)), value))
        super().__init__('GlareOncomingOvertake', world, ego_vehicles, config, debug_mode,
                         criteria_enable, timeout)

    def _initialize_actors(self, config):
        self._lead, self._lead_s = self._spawn_car(self._lead_ahead, CAR_MODEL)
        behind = self._lead_s + self._catch_gap + 2 * self._lead.bounding_box.extent.x
        model = motorcycle_model() if self._follower == 'motorcycle' else CAR_MODEL
        self._overtaker_lights = self._lights if self._follower_lights == 'on' else carla.VehicleLightState.NONE
        self._overtaker, _ = self._spawn_car(behind, model, HAZARD_ROLE, self._overtaker_lights)
        self.other_actors += [self._lead, self._overtaker]

    def _create_behavior(self):
        ego = self.ego_vehicles[0]
        root = py_trees.composites.Sequence(name='GlareOncomingOvertake')
        if self.route_mode:
            root.add_child(LeaveSpaceInFront(self._distance + 20.0))
            # No background car ahead of the ego in its lane: the hazard must meet the ego.
            root.add_child(ChangeRoadBehavior(num_front_vehicles=0))
            root.add_child(ChangeOppositeBehavior(active=False))

        main = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE,
                                            name='OvertakeOutOfGlare')
        main.add_child(OncomingCar(self._lead, self._car_plan(self._lead_s), self._lead_speed, self._lights))
        main.add_child(OncomingOvertaker(self._overtaker, self._lead, ego, self._s, self._pull_out_gap,
                                         self._swing_dist, self._pass_speed, self._return_gap,
                                         self._overtaker_lights, self._pull_back))
        main.add_child(DriveDistance(ego, self._distance + 30.0))     # ends once the ego is past M
        self._add_street_lights(main)
        root.add_child(main)

        if self.route_mode:
            root.add_child(ChangeOppositeBehavior(active=True))
            root.add_child(ChangeRoadBehavior(num_front_vehicles=BA_FRONT_VEHICLES))
        root.add_child(ActorDestroy(self._lead))
        root.add_child(ActorDestroy(self._overtaker))
        return root
