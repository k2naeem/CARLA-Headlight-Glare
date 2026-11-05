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
car or motorcycle within HEADLIGHT_RADIUS_M of the ego and takes its lights
away from the Traffic Manager, which would otherwise switch them back off.
"scene" leaves Bench2Drive's lighting as it is. Lights set during a step show
from the next frame on. Scenario vehicles with the role 'scenario no
lights...' are never switched on: their scenario sets their lights, so a car
it drives dark stays dark. In every mode a vehicle glares only while its
headlights are on.

Writes to $GLARE_OUTPUT (run.py sets one folder per route):
    compare_CAM_FRONT.mp4         clean | glared front camera, the whole route
    model_view/step_<step>.jpg    the 10 inputs as DeepSight would see them
    camera/<CAM>/<step:05d>.jpg   what DeepSight would read (front views glared)
    clean/CAM_FRONT/<step>.jpg    the same front frame without glare
    meta/<step>.json              the 10 image slots, CARLA frame, speed
    glare_log.jsonl               every glare decision, per camera and frame
    road_user_glare.jsonl         per step: how much of the pedestrians and cyclists
                                  in CAM_FRONT the glare washes out (see _label_road_users)
    hazard_glare.jsonl            per step: the same for a scenario's hazard vehicle
                                  (see _label_hazards)
"""
import copy
import json
import math
import os

import carla
import cv2
import numpy as np

from leaderboard.autoagents.npc_agent import NpcAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from deepsight_input_agent import DEEPSIGHT_CAMERA_SPECS, JPEG_QUALITY, make_glare_agent_class
from glare_injector import headlamp_count
from glare_mask_closed_loop import (TWO_WHEELER_TAGS, VEHICLE_TAGS, camera_detected_headlights,
                                    detected_vehicle_mask, project_points_to_image,
                                    project_vehicle_footprint)

# Every N steps, save the front frame with each lit car's predicted position
# and glare verdict drawn on, plus the numbers ($GLARE_DEBUG_EVERY, 0 = off).
DEBUG_EVERY = int(os.environ.get('GLARE_DEBUG_EVERY', '10'))

# Glare reaches 70 m (GlareInjector.max_distance); light a little beyond so a
# car is already lit when it comes into range.
HEADLIGHT_RADIUS_M = 90.0
DRIVING_SPEED_MS = 1.0             # above this once = driving, not parked

# CARLA 0.9.15 semantic classes of the road users glare can hide.
ROAD_USER_LABELS = (12, 13, 19)    # pedestrian, rider, bicycle
WASHED_COVER = 0.5                 # a pixel pushed at least half way to white is washed out
# Contrast of a road user against the RING_PX ring of pixels around it, (a - b) / (a + b + 20)
# on grey levels (the 20 keeps JPEG noise in near-black scenes from counting). Below
# VISIBLE_CONTRAST it does not stand out: in the clean frame that is darkness, not glare.
RING_PX = 8
VISIBLE_CONTRAST = 0.1
# Scenario vehicles whose lights their scenario sets (Bench2Drive's convention),
# and among them the hazard a scenario wants labelled (custom_scenarios/glare_popout.py).
OWN_LIGHTS_ROLE = 'scenario no lights'
HAZARD_ROLE = 'scenario no lights hazard'
HEADLIGHTS = {
    'low': carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam,
    'high': (carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam
             | carla.VehicleLightState.HighBeam),
}


def _contrast(gray, mask, ring):
    """How much `mask` stands out from `ring` in a grey image: (a - b) / (a + b + 20), 0 to 1."""
    if not ring.any():
        return 0.0
    a, b = float(gray[mask].mean()), float(gray[ring].mean())
    return abs(a - b) / (a + b + 20.0)


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

    def _debug_front(self, input_data):
        """Show the glare code's view of CAM_FRONT: where it puts each lit car, and why.

        Writes debug/step_<step>.jpg (the unglared frame, each lit car's centre
        and box drawn where the glare code projects it, labelled with its id,
        verdict and distance) and a line in glare_debug.jsonl with the numbers:
        the camera position the glare code uses (from the seg camera) next to
        where the RGB camera should be (ego pose + its mounting offset).
        """
        injector = self.inputs._glare
        step = self.inputs.step
        if not DEBUG_EVERY or injector is None or step % DEBUG_EVERY:
            return
        cam = injector.cameras.get('CAM_FRONT')
        frame = input_data['CAM_FRONT'][0]
        hero = CarlaDataProvider.get_hero_actor()
        if cam is None or hero is None or frame not in cam.pose:
            return
        cam_matrix, cam_loc = injector.camera_pose(cam, cam.pose[frame])
        records = injector._vehicle_records(frame)
        _, reports = camera_detected_headlights(
            records, cam_matrix, cam.K, (cam.width, cam.height),
            depth_map=cam.depth.get(frame), seg_labels=cam.seg.get(frame), cam_location=cam_loc,
            require_facing=True, max_distance=injector.max_distance,
            min_detected_px=injector.min_detected_px, return_reports=True)

        hero_tf = hero.get_transform()
        spec = DEEPSIGHT_CAMERA_SPECS[0]
        expected = hero_tf.transform(carla.Location(x=spec['x'], y=spec['y'], z=spec['z']))
        expected = (expected.x, expected.y, expected.z)
        m = np.asarray(cam_matrix)
        entry = {
            'step': step, 'frame': frame,
            'camera_location_used': [round(v, 2) for v in cam_loc],
            'camera_location_expected': [round(v, 2) for v in expected],
            'camera_offset_m': round(math.dist(cam_loc, expected), 2),
            'camera_yaw_used': round(math.degrees(math.atan2(m[1][0], m[0][0])), 1),
            'ego_yaw': round(hero_tf.rotation.yaw, 1),
            'cars': [],
        }

        img = input_data['CAM_FRONT'][1][:, :, :3].copy()
        h, w = img.shape[:2]
        cv2.putText(img, 'camera offset %.2f m  yaw used %.1f  ego yaw %.1f' % (
            entry['camera_offset_m'], entry['camera_yaw_used'], entry['ego_yaw']),
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        for rec, rep in zip(records, reports):
            to_cam = np.subtract(cam_loc, rec['location'])
            dist = float(np.linalg.norm(to_cam))
            facing = float(np.dot(rec['forward'], to_cam / max(dist, 1e-6)))
            px, py, depth = project_points_to_image([rec['location']], cam_matrix, cam.K)[0]
            entry['cars'].append({'id': rec['id'], 'status': rep['status'], 'dist_m': round(dist, 1),
                                  'facing': round(facing, 2), 'cam_depth_m': round(float(depth), 1),
                                  'pixel': None if px is None else [round(px), round(py)],
                                  'location': [round(v, 1) for v in rec['location']]})
            color = (0, 255, 0) if rep['status'] == 'detected' else (0, 0, 255)
            if rep['footprint']:
                x0, y0, x1, y1 = rep['footprint']
                cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
            if px is not None and -w < px < 2 * w and -h < py < 2 * h:
                cx, cy = int(min(max(px, 5), w - 5)), int(min(max(py, 5), h - 5))
                cv2.circle(img, (cx, cy), 8, color, -1)
                cv2.putText(img, '%s %s %.0fm' % (rec['id'], rep['status'], dist), (cx + 10, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        os.makedirs(os.path.join(self.save_dir, 'debug'), exist_ok=True)
        cv2.imwrite(os.path.join(self.save_dir, 'debug', 'step_%05d.jpg' % step), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        with open(os.path.join(self.save_dir, 'glare_debug.jsonl'), 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(entry) + '\n')

    def _label_road_users(self, frame):
        """Record how much of the pedestrians and cyclists in CAM_FRONT the glare washes out.

        The seg camera marks which pixels are a pedestrian, rider or bicycle.
        For each one, cover = how far the glare pushed it from its clean
        brightness toward white (0 = untouched, 1 = white), measured on the two
        saved JPEGs, as DeepSight reads them. One line per step with any such
        pixels goes to road_user_glare.jsonl: the pixel count, the mean cover,
        the share of pixels with cover >= WASHED_COVER, and hidden = at least
        half of them washed out. clean_contrast and glared_contrast say how much
        they stand out from the pixels around them in each frame;
        visible_clean = they do in the clean one (else darkness hides them, not
        glare), and glare_occluded = visible clean but hidden or no longer
        standing out once glared. Pixels hidden behind something else are not
        counted: the seg camera does not see them either. Needs glare on (the
        seg camera only runs then).
        """
        cam = self._front_cam()
        seg = cam.seg.get(frame) if cam is not None else None
        if seg is None:
            return
        mask = np.isin(seg, ROAD_USER_LABELS)
        if not mask.any():
            return
        cover = self._cover(mask)
        if cover is not None:
            self._append('road_user_glare.jsonl', dict({'step': self.inputs.step, 'frame': frame}, **cover))

    def _label_hazards(self, frame):
        """Record how much of each hazard vehicle in CAM_FRONT is hidden, and by what.

        A hazard is a vehicle its scenario spawned with HAZARD_ROLE, such as the
        overtaker in GlareOncomingOvertake. One line per step it is in front of
        the camera within 70 m goes to hazard_glare.jsonl: footprint_px is the
        area of its projected box, pixels how many of those the camera sees of
        it (seg + matching depth, so the lead car in front of it does not
        count), then the same cover numbers as _label_road_users over those
        pixels, contrasts included. Few pixels = hidden by the car in front;
        visible_clean false = darkness hides it; glare_occluded = it stands out
        in the clean frame but not once glared. Lights on or off, it is
        labelled. Needs glare on (the seg camera only runs then).
        """
        cam = self._front_cam()
        if cam is None or frame not in cam.pose or cam.seg.get(frame) is None:
            return
        seg, depth = cam.seg[frame], cam.depth.get(frame)
        cam_matrix, cam_loc = self.inputs._glare.camera_pose(cam, cam.pose[frame])
        for actor in CarlaDataProvider.get_world().get_actors().filter('vehicle.*'):
            try:
                if actor.attributes.get('role_name') != HAZARD_ROLE:
                    continue
                bb, tf = actor.bounding_box, actor.get_transform()
                rec = {'matrix': tf.get_matrix(), 'extent': (bb.extent.x, bb.extent.y, bb.extent.z),
                       'bb_center': (bb.location.x, bb.location.y, bb.location.z)}
                lamps = headlamp_count(actor)
                lit = bool(int(actor.get_light_state()) & int(carla.VehicleLightState.LowBeam
                                                              | carla.VehicleLightState.HighBeam))
            except RuntimeError:
                continue                   # destroyed mid-step
            footprint = project_vehicle_footprint(rec, cam_matrix, cam.K, (cam.width, cam.height))
            if footprint is None or footprint[4] > 70.0:
                continue
            x0, y0, x1, y1 = footprint[:4]
            # A tight depth match, so the lead car a few metres in front is not counted.
            own = detected_vehicle_mask(footprint, depth, seg, (cam.width, cam.height), depth_tol_m=1.5,
                                        depth_tol_frac=0.04,
                                        tags=TWO_WHEELER_TAGS if lamps == 1 else VEHICLE_TAGS)
            mask = np.zeros(seg.shape, dtype=bool)
            mask[y0:y1, x0:x1] = own
            entry = {'step': self.inputs.step, 'frame': frame, 'actor': actor.id,
                     'type': actor.type_id, 'lights_on': lit,
                     'distance_m': round(float(footprint[4]), 1), 'footprint_px': (x1 - x0) * (y1 - y0)}
            cover = self._cover(mask) if mask.any() else None
            entry.update(cover or {'pixels': 0, 'mean_cover': 0.0, 'washed': 0.0, 'hidden': False,
                                   'clean_contrast': 0.0, 'glared_contrast': 0.0, 'visible_clean': False,
                                   'glare_occluded': False})
            self._append('hazard_glare.jsonl', entry)

    def _front_cam(self):
        injector = self.inputs._glare
        return injector.cameras.get('CAM_FRONT') if injector is not None else None

    def _cover(self, mask):
        """Pixel count and how far the glare pushed `mask`'s pixels toward white, on this step's JPEGs."""
        step = self.inputs.step
        clean = cv2.imread(os.path.join(self.clean_dir, '%05d.jpg' % step), cv2.IMREAD_GRAYSCALE)
        glared = cv2.imread(os.path.join(self.save_dir, 'camera', 'CAM_FRONT', '%05d.jpg' % step),
                            cv2.IMREAD_GRAYSCALE)
        if clean is None or glared is None or clean.shape != mask.shape or glared.shape != mask.shape:
            return None
        c = clean[mask].astype(np.float32)
        g = glared[mask].astype(np.float32)
        cover = np.clip((g - c) / np.maximum(255.0 - c, 1.0), 0.0, 1.0)
        washed = float(np.mean(cover >= WASHED_COVER))
        # Does it stand out from its surroundings, before and after the glare?
        size = 2 * RING_PX + 1
        near = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
        ring = (near > 0) & ~mask
        clean_contrast, glared_contrast = (_contrast(img, mask, ring) for img in (clean, glared))
        hidden = washed >= 0.5
        visible = clean_contrast >= VISIBLE_CONTRAST
        return {'pixels': int(mask.sum()), 'mean_cover': round(float(cover.mean()), 3),
                'washed': round(washed, 3), 'hidden': hidden,
                'clean_contrast': round(clean_contrast, 3), 'glared_contrast': round(glared_contrast, 3),
                'visible_clean': visible,
                'glare_occluded': visible and (hidden or glared_contrast < VISIBLE_CONTRAST)}

    def _append(self, name, entry):
        with open(os.path.join(self.save_dir, name), 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(entry) + '\n')

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
                if vehicle.id == hero.id or not headlamp_count(vehicle):
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
                if wanted is None or vehicle.attributes.get('role_name', '').startswith(OWN_LIGHTS_ROLE):
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
        self._debug_front(input_data)
        self._label_road_users(input_data['CAM_FRONT'][0])
        self._label_hazards(input_data['CAM_FRONT'][0])
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
        if float(os.environ.get('GLARE_SIZE', '1.0')) != 1.0:
            level += ' x%s' % os.environ['GLARE_SIZE']
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
