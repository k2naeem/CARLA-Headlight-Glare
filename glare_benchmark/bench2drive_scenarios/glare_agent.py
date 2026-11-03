"""Leaderboard agent: NpcAgent drives, DeepSight's cameras record with glare.

The leaderboard's own NpcAgent (CARLA's BasicAgent following the route) drives.
Each step DeepSight's six cameras go through the same input path as in
carla_script_closed_loop.py: GlareMixin adds glare to the three front views,
then the stand-in agent saves them as quality-20 JPEGs and builds the 10-image
input DeepSight would read. No model is loaded.

NpcAgent obeys traffic lights and stops for cars, but ignores pedestrians and
cannot overtake, so it may wait behind a blocked lane until the route times
out. The scores mean nothing; the images are the point.

Writes to $GLARE_OUTPUT (run.py sets one folder per route):
    compare_CAM_FRONT.mp4         clean | glared front camera, the whole route
    model_view/step_<step>.jpg    the 10 inputs as DeepSight would see them
    camera/<CAM>/<step:05d>.jpg   what DeepSight would read (front views glared)
    clean/CAM_FRONT/<step>.jpg    the same front frame without glare
    meta/<step>.json              the 10 image slots, CARLA frame, speed
    glare_log.jsonl               every glare decision, per camera and frame
"""
import copy
import os

import cv2

from leaderboard.autoagents.npc_agent import NpcAgent

from deepsight_input_agent import DEEPSIGHT_CAMERA_SPECS, JPEG_QUALITY, make_glare_agent_class


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

    def sensors(self):
        return copy.deepcopy(DEEPSIGHT_CAMERA_SPECS) + [
            {'type': 'sensor.speedometer', 'reading_frequency': 20, 'id': 'SPEED'},
        ]

    def run_step(self, input_data, timestamp):
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
