"""DeepSight with live headlight glare, for Bench2Drive.

DeepSight (https://github.com/hotdogcheesewhite/DeepSight, Apache License 2.0)
is by its authors and is not part of this repo: QwenAgent is imported from your
own clone. glare_benchmark/bench2drive_scenarios/run.py --driver deepsight loads
this file as the leaderboard agent and puts both repos on PYTHONPATH.

DEEPSIGHT_AGENT picks the agent class to add glare to, as "module:Class"
(default "team_code.qwen_b2d_agent:QwenAgent", DeepSight's own). A fork whose
agent has another file or class name sets it instead of editing this file;
the module is imported from $B2D_DIR, the fork's bench2drive folder.

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
import importlib
import os

import cv2
import numpy as np

from glare_injector import GlareMixin

DEFAULT_AGENT = 'team_code.qwen_b2d_agent:QwenAgent'


def load_agent_class(spec):
    """The class named by a 'module:Class' spec, e.g. DeepSight's QwenAgent."""
    module_name, _, class_name = spec.partition(':')
    if not module_name or not class_name:
        raise ValueError("DEEPSIGHT_AGENT must look like 'module:Class', got %r" % spec)
    return getattr(importlib.import_module(module_name), class_name)


QwenAgent = load_agent_class(os.environ.get('DEEPSIGHT_AGENT') or DEFAULT_AGENT)

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
