"""Something pops out of oncoming headlight glare: Bench2Drive leaderboard scenarios.

Two scenarios on a straight two-way road at night. In each, one lit oncoming
car is the glare source, timed so the hazard appears beside or behind its
headlamps as seen from the ego's front camera:

GlarePedestrianPopOut
    A pedestrian (or cyclist) waits on the centre line, or on the far
    sidewalk, at the meet point M. A lit oncoming car is driven so that it is
    `gap` metres in front of M (between the ego and the pedestrian) when the
    ego is `ped_start` metres from M. At that moment the pedestrian walks into
    the ego's lane, from right behind the car's headlamps: the geometry
    carla_script_closed_loop.py stages for its cyclist.

GlareOncomingOvertake
    A slow lit oncoming car with a second car close behind it, hidden in its
    glare. When the ego is within `swing_dist` of the second car, it pulls out
    into the ego's lane to overtake, head-on, and pulls back in once past.

The oncoming cars drive a fixed plan with their own headlights on (low or
high beams), so they glare under any --headlights mode, and the role name
'scenario no lights' keeps Bench2Drive's lights behaviour from switching them
off. With glare 0 the same run is the clean baseline: any change in the ego's
behaviour is the glare's doing.

Every parameter can be set in the route file, as Bench2Drive's scenarios do
(<ped_start value="30"/>); custom_scenarios/make_routes.py writes route files
with all of them. Each timing moment (pop-out start, in the ego's lane,
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
                                                                      ActorTransformSetter,
                                                                      AtomicBehavior,
                                                                      Idle,
                                                                      KeepVelocity,
                                                                      MovePedestrianWithEgo)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (DriveDistance,
                                                                               InTriggerDistanceToLocation)
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.background_manager import (ChangeOppositeBehavior,
                                              LeaveCrossingSpace,
                                              LeaveSpaceInFront)

BEAMS = {
    'low': carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam,
    'high': (carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam
             | carla.VehicleLightState.HighBeam),
}
# Bench2Drive's RouteLightsBehavior switches the lights of 'scenario' cars on
# and off by distance; it leaves this role alone, so the cars keep our lights.
ROLE = 'scenario no lights'
CAR_MODEL = 'vehicle.lincoln.mkz_2017'   # the oncoming model of carla_script_closed_loop.py
BICYCLE_MODEL = 'vehicle.diamondback.century'


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
    """Drives a lit car down its plan, toward the ego, and keeps its headlights on.

    Road position s is metres along the ego's direction from the meet point
    (`s_of`): the ego starts at s < 0, the oncoming car at s > 0. With
    `meet=(car_s, ego_s)` the car's speed is set every tick so that it reaches
    car_s when the ego reaches ego_s, whatever speed the ego drives; once there
    (or with meet=None) it cruises. Never finishes on its own.
    """

    def __init__(self, actor, plan, ego, s_of, cruise, lights, meet=None,
                 speed_range=(2.0, 16.0), name='OncomingCar'):
        super().__init__(name, actor)
        self._plan = plan
        self._ego = ego
        self._s_of = s_of
        self._cruise = cruise
        self._lights = lights
        self._meet = meet
        self._speed_range = speed_range
        self._planner = None

    def initialise(self):
        self._planner = planner(self._actor)
        self._planner.set_global_plan(self._plan)
        super().initialise()

    def _speed(self):
        if self._meet is None:
            return self._cruise
        car_s, ego_s = self._meet
        car_left = self._s_of(CarlaDataProvider.get_location(self._actor)) - car_s
        ego_left = ego_s - self._s_of(CarlaDataProvider.get_location(self._ego))
        if car_left <= 0.0 or ego_left <= 0.0:
            log_event('oncoming_meet', car_s=round(car_s + car_left, 1), ego_s=round(ego_s - ego_left, 1))
            self._meet = None
            return self._cruise
        # A standing ego is assumed to pull away; the next ticks correct for it.
        seconds = ego_left / max(CarlaDataProvider.get_velocity(self._ego), 4.0)
        return min(max(car_left / seconds, self._speed_range[0]), self._speed_range[1])

    def update(self):
        self._actor.set_light_state(carla.VehicleLightState(self._lights))
        self._planner.set_speed(self._speed() * 3.6)
        self._actor.apply_control(self._planner.run_step())
        return py_trees.common.Status.RUNNING


class OncomingOvertaker(AtomicBehavior):
    """A lit car that catches up with `lead` in the oncoming lane and overtakes it in the ego's lane.

    It starts behind the lead and closes in, its speed set every tick so that it
    is `pull_out_gap` metres behind the lead (inside its glare, seen from the
    ego) when the ego is `swing_dist` away. Then it pulls out into the ego's
    lane at `pass_speed`, already moving faster than the lead, as a real
    overtake does, and pulls back into its own lane `return_gap` metres ahead
    of the lead, or at once if it is already past the ego. Never finishes on
    its own.
    """

    CHANGE_M = 10.0      # metres travelled while changing lane

    def __init__(self, actor, lead, ego, s_of, pull_out_gap, swing_dist, pass_speed, return_gap,
                 lights, name='OncomingOvertaker'):
        super().__init__(name, actor)
        self._lead = lead
        self._ego = ego
        self._s_of = s_of
        self._pull_out_gap = pull_out_gap
        self._swing_dist = swing_dist
        self._pass_speed = pass_speed
        self._return_gap = return_gap
        self._lights = lights
        self._map = CarlaDataProvider.get_map()
        self._phase = 'approach'
        self._planner = None

    def initialise(self):
        self._planner = planner(self._actor, max_throttle=1.0)
        self._planner.set_global_plan(lane_plan(self._here(), 200.0)[2:])
        super().initialise()

    def _here(self):
        return self._map.get_waypoint(CarlaDataProvider.get_location(self._actor))

    def _change_lane(self, into_ego_lane):
        """Plan from here into the other lane, then 120 m along it in this car's direction."""
        here = self._here()
        if into_ego_lane:
            target = along(here, self.CHANGE_M).get_left_lane()      # oncoming lane's left = ego lane
            if target is None:
                return
            plan = [target]
            for _ in range(60):                                     # the ego lane runs the other way
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
            if gap <= self._pull_out_gap or ego_dist <= self._swing_dist:
                self._phase = 'pass'
                self._change_lane(into_ego_lane=True)
                log_event('overtake_swing', ego_dist=round(ego_dist, 1), lead_gap=round(gap, 1),
                          speed=round(CarlaDataProvider.get_velocity(self._actor), 1))
                speed = self._pass_speed
            else:
                # A standing ego is assumed to pull away; the next ticks correct for it.
                closing = (max(CarlaDataProvider.get_velocity(self._ego), 4.0)
                           + max(CarlaDataProvider.get_velocity(self._actor), lead_speed))
                seconds = max(ego_dist - self._swing_dist, 1.0) / closing
                speed = min(max(lead_speed + (gap - self._pull_out_gap) / seconds, lead_speed),
                            self._pass_speed)
        elif self._phase == 'pass':
            speed = self._pass_speed
            past_ego = self._s_of(loc) < self._s_of(ego_loc)
            if ahead - lengths >= self._return_gap or past_ego:
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

    def _spawn_car(self, s, model):
        """A car in the oncoming lane at s, moved further out if the spot is taken."""
        for shift in (0.0, 5.0, 10.0, 15.0):
            wp = self._oncoming_at(s + shift)
            car = CarlaDataProvider.request_new_actor(model, wp.transform, rolename=ROLE)
            if car is not None:
                car.set_light_state(carla.VehicleLightState(self._lights))
                return car, s + shift
        raise ValueError('could not spawn %s in the oncoming lane near s=%.0f m' % (model, s))

    def _car_plan(self, s):
        """The oncoming lane from s to 60 m behind the trigger point."""
        return lane_plan(self._oncoming_at(s), s + self._distance + 60.0)

    def _create_test_criteria(self):
        if self.route_mode:
            return []
        return [CollisionTest(self.ego_vehicles[0])]

    def __del__(self):
        self.remove_all_actors()


class GlarePedestrianPopOut(_GlareScenario):
    """A pedestrian or cyclist steps into the ego's lane from right behind an oncoming car's headlamps.

    Parameters (route file, all optional):
        distance    trigger point to the meet point M, m (90)
        ped_start   the ego's distance to M when the adversary moves, m (30)
        gap         how far in front of M the car is at that moment, m (4)
        car_ahead   where the car waits, m past M (35)
        cruise      the car's speed after the meet, m/s (9)
        adversary   'walker' or 'bicycle' (walker)
        start       'line' (centre line, beside the passing car) or 'curb' (far sidewalk) (line)
        speed       the adversary's speed, m/s (1.4 walker, 3.0 bicycle)
        hold        seconds a walker stands in the ego's lane before walking on (3)
        beams       the car's 'low' or 'high' beams (low)
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False,
                 criteria_enable=True, timeout=60):
        self._ped_start = get_value_parameter(config, 'ped_start', float, 30.0)
        self._gap = get_value_parameter(config, 'gap', float, 4.0)
        self._car_ahead = get_value_parameter(config, 'car_ahead', float, 35.0)
        self._cruise = get_value_parameter(config, 'cruise', float, 9.0)
        self._kind = get_value_parameter(config, 'adversary', str, 'walker')
        self._start = get_value_parameter(config, 'start', str, 'line')
        self._speed = get_value_parameter(config, 'speed', float, 1.4 if self._kind == 'walker' else 3.0)
        self._hold = get_value_parameter(config, 'hold', float, 3.0)
        if self._kind not in ('walker', 'bicycle'):
            raise ValueError("'adversary' must be 'walker' or 'bicycle', not %r" % self._kind)
        if self._start not in ('line', 'curb'):
            raise ValueError("'start' must be 'line' or 'curb', not %r" % self._start)
        super().__init__('GlarePedestrianPopOut', world, ego_vehicles, config, debug_mode,
                         criteria_enable, timeout)

    def _crossing(self):
        """Where the adversary waits, facing the ego's lane, and how far it walks to the lane centre and then off the road."""
        meet = self._meet_wp.transform
        right = meet.get_right_vector()          # from the oncoming side into the ego's lane
        if self._start == 'line':
            to_line = self._meet_wp.lane_width / 2.0
            origin = meet.location - carla.Location(x=right.x * to_line, y=right.y * to_line)
            to_centre = to_line
        else:
            side = self._oncoming_wp.get_right_lane()   # the oncoming lane's right = the far edge
            if side is not None and side.lane_type == carla.LaneType.Sidewalk:
                origin = side.transform.location
            else:
                out = self._meet_wp.lane_width / 2.0 + self._oncoming_wp.lane_width + 1.0
                origin = meet.location - carla.Location(x=right.x * out, y=right.y * out)
            to_centre = math.hypot(origin.x - meet.location.x, origin.y - meet.location.y)
        origin.z = meet.location.z + (1.0 if self._kind == 'walker' else 0.3)
        yaw = math.degrees(math.atan2(right.y, right.x))
        beyond = self._meet_wp.lane_width / 2.0 + 2.0     # from the lane centre off the road
        return carla.Transform(origin, carla.Rotation(yaw=yaw)), to_centre, beyond

    def _initialize_actors(self, config):
        self._car, self._car_s = self._spawn_car(self._car_ahead, CAR_MODEL)
        self._wait_tf, self._to_centre, self._beyond = self._crossing()

        model = 'walker.pedestrian.*' if self._kind == 'walker' else BICYCLE_MODEL
        adversary = CarlaDataProvider.request_new_actor(model, self._wait_tf, rolename=ROLE)
        if adversary is None:
            raise ValueError('could not spawn the %s at %s' % (self._kind, self._wait_tf.location))
        if self._kind == 'walker':
            # Bench2Drive's fix for walkers on large maps (DynamicObjectCrossing):
            # respawn it under the ego and keep it there until the scenario starts.
            type_id = adversary.type_id
            adversary.destroy()
            hidden = self.ego_vehicles[0].get_transform()
            hidden.location.z -= 50
            adversary = CarlaDataProvider.request_new_actor(type_id, hidden)
            if adversary is None:
                raise ValueError("couldn't spawn the walker substitute")
        else:
            hidden = carla.Transform(self._wait_tf.location + carla.Location(z=-200), self._wait_tf.rotation)
            adversary.set_transform(hidden)
        adversary.set_simulate_physics(False)
        self._adversary = adversary
        self.other_actors += [self._adversary, self._car]

    def _setup_scenario_trigger(self, config):
        trigger = super()._setup_scenario_trigger(config)
        if not self.route_mode or self._kind != 'walker':
            return trigger
        parallel = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE,
                                                name='ScenarioTrigger')
        parallel.add_child(MovePedestrianWithEgo(self.ego_vehicles[0], self._adversary, 100))
        parallel.add_child(trigger)
        return parallel

    def _state(self):
        ego = CarlaDataProvider.get_location(self.ego_vehicles[0])
        car = CarlaDataProvider.get_location(self._car)
        return {'ego_s': round(self._s(ego), 1), 'car_s': round(self._s(car), 1),
                'car_to_ego_m': round(car.distance(ego), 1)}

    def _create_behavior(self):
        ego = self.ego_vehicles[0]
        root = py_trees.composites.Sequence(name='GlarePedestrianPopOut')
        if self.route_mode:
            root.add_child(LeaveSpaceInFront(self._distance + 20.0))
            root.add_child(ChangeOppositeBehavior(active=False))
        root.add_child(ActorTransformSetter(self._adversary, self._wait_tf, True))

        main = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE,
                                            name='PopOutOfGlare')
        main.add_child(OncomingCar(self._car, self._car_plan(self._car_s), ego, self._s, self._cruise,
                                   self._lights, meet=(-self._gap, -self._ped_start)))

        popout = py_trees.composites.Sequence(name='Adversary')
        popout.add_child(InTriggerDistanceToLocation(ego, self._meet_wp.transform.location, self._ped_start))
        if self.route_mode:
            popout.add_child(LeaveCrossingSpace(self._meet_wp))
        popout.add_child(LogEvent('popout_start', self._state))
        bicycle = self._kind == 'bicycle'
        if bicycle:      # a cyclist rides straight across
            popout.add_child(KeepVelocity(self._adversary, self._speed, force_speed=True,
                                          distance=self._to_centre + self._beyond))
        else:            # a walker stops in the ego's lane, then walks on off the road
            popout.add_child(KeepVelocity(self._adversary, self._speed, distance=self._to_centre))
            popout.add_child(LogEvent('popout_in_lane', self._state))
            popout.add_child(Idle(self._hold))
            popout.add_child(KeepVelocity(self._adversary, self._speed, distance=self._beyond))
        popout.add_child(LogEvent('popout_clear', self._state))
        popout.add_child(DriveDistance(ego, self._ped_start + 15.0))
        main.add_child(popout)
        root.add_child(main)

        if self.route_mode:
            root.add_child(ChangeOppositeBehavior(active=True))
        root.add_child(ActorDestroy(self._adversary))
        root.add_child(ActorDestroy(self._car))
        return root


class GlareOncomingOvertake(_GlareScenario):
    """An oncoming car pulls out of a slower lit car's glare into the ego's lane to overtake it.

    Parameters (route file, all optional):
        distance      trigger point to the meet point M, m (60)
        lead_ahead    where the slow lead car waits, m past M (50)
        lead_speed    the lead car's speed, m/s (5)
        catch_gap     the overtaker's starting gap behind the lead, m (25)
        pull_out_gap  its gap behind the lead when it pulls out, m (5)
        swing_dist    the ego's distance to the overtaker at that moment, m (75)
        pass_speed    the overtaker's top speed, m/s (15)
        return_gap    how far past the lead it pulls back in, m (3)
        beams         both cars' 'low' or 'high' beams (low)
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
        super().__init__('GlareOncomingOvertake', world, ego_vehicles, config, debug_mode,
                         criteria_enable, timeout)

    def _initialize_actors(self, config):
        self._lead, self._lead_s = self._spawn_car(self._lead_ahead, CAR_MODEL)
        behind = self._lead_s + self._catch_gap + 2 * self._lead.bounding_box.extent.x
        self._overtaker, _ = self._spawn_car(behind, CAR_MODEL)
        self.other_actors += [self._lead, self._overtaker]

    def _create_behavior(self):
        ego = self.ego_vehicles[0]
        root = py_trees.composites.Sequence(name='GlareOncomingOvertake')
        if self.route_mode:
            root.add_child(LeaveSpaceInFront(self._distance + 20.0))
            root.add_child(ChangeOppositeBehavior(active=False))

        main = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE,
                                            name='OvertakeOutOfGlare')
        main.add_child(OncomingCar(self._lead, self._car_plan(self._lead_s), ego, self._s,
                                   self._lead_speed, self._lights))
        main.add_child(OncomingOvertaker(self._overtaker, self._lead, ego, self._s, self._pull_out_gap,
                                         self._swing_dist, self._pass_speed, self._return_gap,
                                         self._lights))
        main.add_child(DriveDistance(ego, self._distance + 30.0))     # ends once the ego is past M
        root.add_child(main)

        if self.route_mode:
            root.add_child(ChangeOppositeBehavior(active=True))
        root.add_child(ActorDestroy(self._lead))
        root.add_child(ActorDestroy(self._overtaker))
        return root
