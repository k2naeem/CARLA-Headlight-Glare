"""Leaderboard agent: NpcAgent drives, DeepSight's cameras record with glare.

The leaderboard's own NpcAgent (CARLA's BasicAgent following the route) drives.
Each step DeepSight's six cameras go through the same input path as in
carla_script_closed_loop.py: GlareMixin adds glare to the three front views,
then the stand-in agent saves them as quality-20 JPEGs and builds the 10-image
input DeepSight would read. No model is loaded.

NpcAgent obeys traffic lights and stops for cars, but ignores pedestrians and
cannot overtake, so it may wait behind a blocked lane until the route times
out. The scores mean nothing; the images are the point.

Only driving cars glare. A vehicle counts as driving once it has moved faster
than DRIVING_SPEED_MS at any point in the run, so a car stopped at a light
that drove there still glares, a car pulling out of a parking spot starts
glaring when it moves, and a parked car never glares, whatever its lights show.

Headlights ($GLARE_HEADLIGHTS). Bench2Drive's night traffic mostly drives
with its headlights off, so there is nothing to glare. With "low" (default) or
"high", every step this agent switches on the headlights of each driving
4-wheeled vehicle within HEADLIGHT_RADIUS_M of the ego and takes its lights
away from the Traffic Manager, which would otherwise switch them back off.
"scene" leaves Bench2Drive's lighting as it is. Lights set during a step show
from the next frame on.

Writes to $GLARE_OUTPUT (run.py sets one folder per route):
    compare_CAM_FRONT.mp4         clean | glared front camera, the whole route
    model_view/step_<step>.jpg    the 10 inputs as DeepSight would see them
    camera/<CAM>/<step:05d>.jpg   what DeepSight would read (front views glared)
    clean/CAM_FRONT/<step>.jpg    the same front frame without glare
    meta/<step>.json              the 10 image slots, CARLA frame, speed
    glare_log.jsonl               every glare decision, per camera and frame
"""
import copy
import json
import os

import carla
import cv2

from leaderboard.autoagents.npc_agent import NpcAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from deepsight_input_agent import DEEPSIGHT_CAMERA_SPECS, JPEG_QUALITY, make_glare_agent_class

# Glare reaches 70 m (GlareInjector.max_distance); light a little beyond so a
# car is already lit when it comes into range.
HEADLIGHT_RADIUS_M = 90.0
DRIVING_SPEED_MS = 1.0             # above this once = driving, not parked
HEADLIGHTS = {
    'low': carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam,
    'high': (carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam
             | carla.VehicleLightState.HighBeam),
}


def get_entry_point():
    return 'GlareAgent'


class GlareAgent(NpcAgent):

    def setup(self, path_to_conf_file, keypoints=None, manager=None):
        # DeepSight's leaderboard passes (config, keypoints, manager); the
        # original Bench2Drive one passes only the config. Accept both.
        super().setup(path_to_conf_file)
        self.save_dir = os.environ.get('GLARE_OUTPUT') or os.path.abspath('glare_output')
        self.clean_dir = os.path.join(self.save_dir, 'clean', 'CAM_FRONT')
        os.makedirs(self.clean_dir, exist_ok=True)
        self.inputs = make_glare_agent_class()(self.save_dir)
        # The ego exists by now: spawn the seg/depth cameras before the first
        # tick, so frame 1 is glared too.
        self.inputs._glare_injector()

        self.headlights = os.environ.get('GLARE_HEADLIGHTS', 'low')
        self._tm = None
        self._driving_ids = set()      # vehicles seen moving; only these glare
        self._parked_ids = set()       # vehicles in range never seen moving
        self._lit_ids = set()          # vehicles whose lights this agent has taken over
        self._max_in_range = 0
        injector = self.inputs._glare
        if injector is not None:
            injector.vehicle_filter = lambda actor: actor.id in self._driving_ids

    def sensors(self):
        return copy.deepcopy(DEEPSIGHT_CAMERA_SPECS) + [
            {'type': 'sensor.speedometer', 'reading_frequency': 20, 'id': 'SPEED'},
        ]

    def _update_traffic(self):
        """Mark nearby vehicles as driving or parked; light the driving ones."""
        wanted = HEADLIGHTS.get(self.headlights)
        hero = CarlaDataProvider.get_hero_actor()
        if hero is None:
            return
        if wanted is not None and self._tm is None:
            self._tm = CarlaDataProvider.get_client().get_trafficmanager(
                CarlaDataProvider.get_traffic_manager_port())
        ego = hero.get_location()
        in_range = 0
        for vehicle in CarlaDataProvider.get_world().get_actors().filter('vehicle.*'):
            try:
                if vehicle.id == hero.id or int(vehicle.attributes.get('number_of_wheels', 4)) < 4:
                    continue
                if vehicle.get_location().distance(ego) > HEADLIGHT_RADIUS_M:
                    continue
                in_range += 1
                if vehicle.id not in self._driving_ids:
                    v = vehicle.get_velocity()
                    if (v.x * v.x + v.y * v.y + v.z * v.z) ** 0.5 > DRIVING_SPEED_MS:
                        self._driving_ids.add(vehicle.id)
                        self._parked_ids.discard(vehicle.id)
                    else:
                        self._parked_ids.add(vehicle.id)
                        continue
                if wanted is None:
                    continue
                if vehicle.id not in self._lit_ids:
                    self._tm.update_vehicle_lights(vehicle, False)
                    self._lit_ids.add(vehicle.id)
                state = int(vehicle.get_light_state())
                if state & int(wanted) != int(wanted):
                    vehicle.set_light_state(carla.VehicleLightState(state | int(wanted)))
            except RuntimeError:
                continue                   # destroyed mid-step
        self._max_in_range = max(self._max_in_range, in_range)

    def run_step(self, input_data, timestamp):
        self._update_traffic()
        self.inputs.run_step(input_data, timestamp)
        # input_data still holds the unglared images: GlareMixin only replaces
        # them in the agent's own tick() result.
        cv2.imwrite(os.path.join(self.clean_dir, '%05d.jpg' % self.inputs.step),
                    input_data['CAM_FRONT'][1][:, :, :3], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return super().run_step(input_data, timestamp)

    def destroy(self):
        inputs = getattr(self, 'inputs', None)
        if inputs is not None:
            inputs.destroy()       # writes glare_log.jsonl, removes the seg/depth cameras
            self._write_video(inputs.step + 1)
            with open(os.path.join(self.save_dir, 'headlights.json'), 'w', encoding='utf-8') as fh:
                json.dump({'mode': self.headlights, 'radius_m': HEADLIGHT_RADIUS_M,
                           'driving_speed_ms': DRIVING_SPEED_MS,
                           'vehicles_driving': len(self._driving_ids),
                           'vehicles_parked': len(self._parked_ids),
                           'vehicles_lit': len(self._lit_ids),
                           'max_vehicles_in_range': self._max_in_range}, fh, indent=1)
            self.inputs = None
        super().destroy()

    def _write_video(self, n_steps, fps=10):
        """Clean | glared CAM_FRONT, decoded from the saved JPEGs."""
        writer = None
        level = os.environ.get('GLARE_SEVERITY', '1.0')
        for step in range(n_steps):
            clean = cv2.imread(os.path.join(self.clean_dir, '%05d.jpg' % step))
            glared = cv2.imread(os.path.join(self.save_dir, 'camera', 'CAM_FRONT', '%05d.jpg' % step))
            if clean is None or glared is None:
                continue
            frame = cv2.hconcat([cv2.resize(clean, (800, 450)), cv2.resize(glared, (800, 450))])
            cv2.putText(frame, 'CLEAN', (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, 'GLARE %s  step %d' % (level, step), (814, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2, cv2.LINE_AA)
            if writer is None:
                writer = cv2.VideoWriter(os.path.join(self.save_dir, 'compare_CAM_FRONT.mp4'),
                                         cv2.VideoWriter_fourcc(*'mp4v'), fps, (1600, 450))
            writer.write(frame)
        if writer is not None:
            writer.release()
