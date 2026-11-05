"""DeepSight with live headlight glare, for Bench2Drive.

Copy this file into DeepSight's bench2drive/team_code/ and put this repo on
PYTHONPATH (for glare_injector and glare_mask_closed_loop). Then run DeepSight's
evaluation script with

    TEAM_AGENT=team_code/glare_qwen_agent.py

GLARE_SEVERITY (default 1.0), GLARE_SIZE (default 1.0) and GLARE_STRICT
(default 0) set the glare; see glare_injector.py. Severity 0 runs DeepSight
unchanged.

Besides adding glare, this agent fixes two things that stop DeepSight's own
agent from running outside its authors' machines:

  * It checks SAVE_PATH, IS_BENCH2DRIVE and ROUTES up front. DeepSight needs
    all three and otherwise crashes on the first step with a less clear error.
  * DeepSight's black history image is a hard-coded path on its authors' NAS.
    When that file does not exist, the history slots point to a black image
    written into this route's save folder instead.
"""
import os

import cv2
import numpy as np

from team_code.qwen_b2d_agent import QwenAgent
from glare_injector import GlareMixin

# The black history placeholder hard-coded in qwen_b2d_agent.format_message().
DEEPSIGHT_BLACK_PATH = ('/mnt/nas-data-1/zhanglingjun.zlj1/ad_data_process/'
                        'sft_data_api_explain/hisblack.jpg')

REQUIRED_ENV = ('SAVE_PATH', 'IS_BENCH2DRIVE', 'ROUTES')


def get_entry_point():
    return 'GlareQwenAgent'


class GlareQwenAgent(GlareMixin, QwenAgent):
    """DeepSight's QwenAgent with glare added to the front views in tick()."""

    def setup(self, *args, **kwargs):
        missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
        if missing:
            raise RuntimeError("DeepSight needs these environment variables set: %s"
                               % ", ".join(missing))
        super().setup(*args, **kwargs)

        self._black_path = None
        if not os.path.exists(DEEPSIGHT_BLACK_PATH):
            self._black_path = str(self.save_path / 'hisblack.jpg')
            cv2.imwrite(self._black_path, np.zeros((900, 1600, 3), np.uint8),
                        [cv2.IMWRITE_JPEG_QUALITY, 20])

    def format_message(self, tick_data):
        prompt, images, bevtargetpoints = super().format_message(tick_data)
        if self._black_path is not None:
            images = [self._black_path if p == DEEPSIGHT_BLACK_PATH else p
                      for p in images]
        return prompt, images, bevtargetpoints
