"""A stand-in for DeepSight that reproduces its camera input, without the model.

DeepSight (Bench2Drive team_code/qwen_b2d_agent.py) never reads the camera
arrays it is handed. On every step it:

  1. tick()            slices each camera to BGR: input_data[cam][1][:, :, :3]
  2. save_cur_frame()  writes each camera to save_path/camera/<CAM>/<step>.jpg
                       at JPEG quality 20
  3. format_message()  builds a list of 10 FILE PATHS, not images: CAM_FRONT
                       from 20, 15, 10 and 5 steps ago (a black image until that
                       much history exists), then the six current views
  4. the Qwen processor opens those files and resizes each to 644 x 364
  5. save()            adds this step's CAM_FRONT path to the history list

This agent does steps 1-5 the same way and stops where DeepSight would call
model.generate(). So whatever these files contain is exactly what DeepSight
would see. It does not drive: run_step() returns None, and the test script keeps
the ego on CARLA's Traffic Manager.

With glare, use GlareDeepSightInputAgent. The glare is added in tick(), before
step 2 writes the files, as it would be for the real agent.

Outputs under save_path:
    camera/<CAM>/<step:05d>.jpg   the files DeepSight would load (glared)
    meta/<step:04d>.json          the 10 image slots for that step
    model_view/step_<step>.jpg    the 10 images after the 644x364 resize, tiled
    hisblack.jpg                  the black history placeholder
    input_summary.json            per-run counts, written by destroy()
"""
import json
import os

import cv2
import numpy as np
from PIL import Image

# The six views DeepSight's prompt lists, in prompt order.
SURROUND_CAMERAS = ('CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')

JPEG_QUALITY = 20                  # save_cur_frame() uses quality 20
MODEL_IMAGE_SIZE = (644, 364)      # (width, height) every image is resized to

# DeepSight's format_message(): for i in 4..1, history index = i * frame // 2,
# with self.frame = 10, i.e. 20, 15, 10 and 5 steps back. Its prompt calls them
# 2.0 s, 1.5 s, 1.0 s and 0.5 s ago, which holds if the agent runs at 10 Hz.
HISTORY_STEPS = (20, 15, 10, 5)
HISTORY_LABELS = ('2.0s ago', '1.5s ago', '1.0s ago', '0.5s ago')

# Camera rig from DeepSight's sensors(). The leaderboard sets only size and FOV
# on each camera, so CARLA's default exposure and post-processing apply.
DEEPSIGHT_CAMERA_SPECS = [
    {'type': 'sensor.camera.rgb', 'id': 'CAM_FRONT',
     'x': 0.80, 'y': 0.0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
     'width': 1600, 'height': 900, 'fov': 70},
    {'type': 'sensor.camera.rgb', 'id': 'CAM_FRONT_LEFT',
     'x': 0.27, 'y': -0.55, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': -55.0,
     'width': 1600, 'height': 900, 'fov': 70},
    {'type': 'sensor.camera.rgb', 'id': 'CAM_FRONT_RIGHT',
     'x': 0.27, 'y': 0.55, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 55.0,
     'width': 1600, 'height': 900, 'fov': 70},
    {'type': 'sensor.camera.rgb', 'id': 'CAM_BACK',
     'x': -2.0, 'y': 0.0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
     'width': 1600, 'height': 900, 'fov': 110},
    {'type': 'sensor.camera.rgb', 'id': 'CAM_BACK_LEFT',
     'x': -0.32, 'y': -0.55, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': -110.0,
     'width': 1600, 'height': 900, 'fov': 70},
    {'type': 'sensor.camera.rgb', 'id': 'CAM_BACK_RIGHT',
     'x': -0.32, 'y': 0.55, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 110.0,
     'width': 1600, 'height': 900, 'fov': 70},
]


def load_like_qwen(path):
    """Open an image the way DeepSight's model receives it: RGB, 644 x 364.

    The Qwen processor loads the file with PIL and resizes it. Bicubic is
    Pillow's default for resize; the processor may differ slightly in filter,
    so treat this as a close preview, not a bit-exact copy.
    """
    with Image.open(path) as im:
        return im.convert('RGB').resize(MODEL_IMAGE_SIZE, Image.BICUBIC)


class DeepSightInputAgent:
    """DeepSight's input path with the model removed. See the module docstring.

    save_path         output folder, like DeepSight's self.save_path
    cameras           which of the six views are spawned; the rest are filled
                      with the black image, so the prompt still has 10 slots
    model_view_every  write a tiled view of the 10 model inputs every N steps
    """

    def __init__(self, save_path, cameras=SURROUND_CAMERAS, model_view_every=20):
        self.save_path = str(save_path)
        self.cameras = tuple(c for c in SURROUND_CAMERAS if c in cameras)
        self.model_view_every = model_view_every
        self.step = -1
        self.his_images = []           # CAM_FRONT path per past step, like DeepSight
        self.frames = []               # CARLA frame number per step

        for cam in SURROUND_CAMERAS:
            os.makedirs(os.path.join(self.save_path, 'camera', cam), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'meta'), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'model_view'), exist_ok=True)

        # Local stand-in for DeepSight's hard-coded .../hisblack.jpg.
        self.black_path = os.path.join(self.save_path, 'hisblack.jpg')
        cv2.imwrite(self.black_path, np.zeros((900, 1600, 3), np.uint8),
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

    def sensors(self):
        """DeepSight's camera specs for the views this agent uses."""
        return [s for s in DEEPSIGHT_CAMERA_SPECS if s['id'] in self.cameras]

    def frame_path(self, cam, step):
        return os.path.join(self.save_path, 'camera', cam, '%05d.jpg' % step)

    # --- DeepSight's steps, in order ---------------------------------------

    def tick(self, input_data):
        """Step 1. Same slicing as DeepSight: BGRA buffer -> BGR view."""
        self.step += 1
        imgs = {cam: input_data[cam][1][:, :, :3]
                for cam in self.cameras if cam in input_data}
        frame = next(iter(input_data.values()))[0] if input_data else None
        speed = input_data['SPEED'][1].get('speed') if 'SPEED' in input_data else None
        return {'imgs': imgs, 'frame': frame, 'speed': speed}

    def save_cur_frame(self, tick_data):
        """Step 2. Write every view to disk at JPEG quality 20."""
        for cam, img in tick_data['imgs'].items():
            cv2.imwrite(self.frame_path(cam, self.step), img,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

    def format_message(self, tick_data):
        """Step 3. The 10 file paths, in DeepSight's prompt order."""
        slots = []
        for steps_back, label in zip(HISTORY_STEPS, HISTORY_LABELS):
            if len(self.his_images) < steps_back:
                slots.append(('CAM_FRONT ' + label, self.black_path))
            else:
                slots.append(('CAM_FRONT ' + label, self.his_images[-steps_back]))
        for cam in SURROUND_CAMERAS:
            path = (self.frame_path(cam, self.step) if cam in tick_data['imgs']
                    else self.black_path)
            slots.append((cam, path))
        assert len(slots) == 10, "DeepSight's prompt has exactly 10 image slots"
        return slots

    def save(self, tick_data):
        """Step 5. This step's CAM_FRONT joins the history."""
        self.his_images.append(self.frame_path('CAM_FRONT', self.step))
        self.frames.append(tick_data['frame'])

    def run_step(self, input_data, timestamp):
        tick_data = self.tick(input_data)
        self.save_cur_frame(tick_data)
        slots = self.format_message(tick_data)

        # Step 4. DeepSight's processor fails on a missing file, so check every
        # step; do the full load and resize only when writing a model view.
        missing = [p for _, p in slots if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError("model input missing at step %d: %s"
                                    % (self.step, missing))
        if self.model_view_every and self.step % self.model_view_every == 0:
            self._write_model_view(slots)

        meta = {
            'step': self.step, 'frame': tick_data['frame'], 'timestamp': timestamp,
            'speed': tick_data['speed'],
            'slots': [{'slot': name, 'path': os.path.relpath(p, self.save_path),
                       'black': p == self.black_path} for name, p in slots],
        }
        with open(os.path.join(self.save_path, 'meta', '%04d.json' % self.step),
                  'w', encoding='utf-8') as fh:
            json.dump(meta, fh, indent=1)

        self.save(tick_data)
        return None        # DeepSight returns a control here; this agent does not drive

    def _write_model_view(self, slots):
        """The 10 images as the model receives them, tiled 5 x 2 with labels."""
        tiles = []
        for name, path in slots:
            tile = cv2.cvtColor(np.asarray(load_like_qwen(path)), cv2.COLOR_RGB2BGR)
            cv2.rectangle(tile, (0, 0), (tile.shape[1], 26), (0, 0, 0), -1)
            cv2.putText(tile, name, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
        grid = np.vstack([np.hstack(tiles[:5]), np.hstack(tiles[5:])])
        cv2.imwrite(os.path.join(self.save_path, 'model_view', 'step_%05d.jpg' % self.step),
                    grid, [cv2.IMWRITE_JPEG_QUALITY, 90])

    def destroy(self):
        black_history = sum(1 for s in range(self.step + 1)
                            if s < max(HISTORY_STEPS))
        summary = {
            'steps': self.step + 1,
            'cameras_spawned': list(self.cameras),
            'cameras_filled_black': [c for c in SURROUND_CAMERAS if c not in self.cameras],
            'history_steps_back': list(HISTORY_STEPS),
            'steps_with_some_black_history': black_history,
            'jpeg_quality': JPEG_QUALITY,
            'model_image_size': list(MODEL_IMAGE_SIZE),
        }
        with open(os.path.join(self.save_path, 'input_summary.json'), 'w',
                  encoding='utf-8') as fh:
            json.dump(summary, fh, indent=1)


def make_glare_agent_class():
    """GlareDeepSightInputAgent, built on demand so this file imports without CARLA."""
    from glare_injector import GlareMixin

    class GlareDeepSightInputAgent(GlareMixin, DeepSightInputAgent):
        """DeepSightInputAgent with glare added to the front views in tick()."""

    return GlareDeepSightInputAgent
