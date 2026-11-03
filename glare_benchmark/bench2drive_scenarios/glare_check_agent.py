"""Bench2Drive agent that checks the glare on real routes, with no DeepSight model.

The car is driven along the route by the leaderboard's own NpcAgent (CARLA's
BasicAgent following the route plan). Each step, DeepSight's six cameras go
through the same path as in carla_script_closed_loop.py: GlareMixin adds glare
to the three front views, then the stand-in agent saves them as quality-20
JPEGs and builds the 10-image input DeepSight would read. Nothing is sent to
a model.

The driver obeys traffic lights and stops for vehicles, but ignores
pedestrians and cannot overtake, so it may stop behind a blocked lane until
the route times out. That is fine here: the point is the images, not the score.

Output per route, under $SAVE_PATH/routes/<route>/:
    camera/<CAM>/<step:05d>.jpg   what DeepSight would read (front views glared)
    clean/CAM_FRONT/<step>.jpg    the same front frame without glare
    meta/<step>.json              the 10 image slots, CARLA frame, speed
    model_view/step_<step>.jpg    the 10 inputs as the model would see them
    glare_log.jsonl               every glare decision, per camera and frame
    input_summary.json            per-run counts
"""
import copy
import datetime
import os

import cv2

from leaderboard.autoagents.npc_agent import NpcAgent

from deepsight_input_agent import DEEPSIGHT_CAMERA_SPECS, JPEG_QUALITY, make_glare_agent_class


def get_entry_point():
    return 'GlareCheckAgent'


class GlareCheckAgent(NpcAgent):
    """NpcAgent driving, DeepSight's input path with glare recording."""

    def setup(self, path_to_conf_file, keypoints=None, manager=None):
        # DeepSight's leaderboard passes (config, keypoints, manager); the
        # stock Bench2Drive one passes only the config. Accept both.
        super().setup(path_to_conf_file)
        conf = str(path_to_conf_file)
        route = conf.split('+')[-1] if '+' in conf else datetime.datetime.now().strftime('%m_%d_%H_%M_%S')
        save_root = os.environ.get('SAVE_PATH') or 'glare_check_output'
        self.save_dir = os.path.join(save_root, 'routes', route)
        self.clean_dir = os.path.join(self.save_dir, 'clean', 'CAM_FRONT')
        os.makedirs(self.clean_dir, exist_ok=True)

        self.inputs = make_glare_agent_class()(self.save_dir)
        # The ego exists by now, so spawn the seg/depth cameras before the first
        # tick (strict mode would otherwise stop on frame 1).
        self.inputs._glare_injector()

    def sensors(self):
        return copy.deepcopy(DEEPSIGHT_CAMERA_SPECS) + [
            {'type': 'sensor.speedometer', 'reading_frequency': 20, 'id': 'SPEED'},
        ]

    def run_step(self, input_data, timestamp):
        self.inputs.run_step(input_data, timestamp)
        # input_data still holds the unglared images: GlareMixin only replaces
        # them in the agent's own tick() result.
        cv2.imwrite(os.path.join(self.clean_dir, '%05d.jpg' % self.inputs.step),
                    input_data['CAM_FRONT'][1][:, :, :3],
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return super().run_step(input_data, timestamp)

    def destroy(self):
        inputs = getattr(self, 'inputs', None)
        if inputs is not None:
            inputs.destroy()       # writes glare_log.jsonl, removes the seg/depth cameras
            self.inputs = None
        super().destroy()
