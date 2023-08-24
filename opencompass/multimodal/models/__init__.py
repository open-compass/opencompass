import os.path as osp

from opencompass.utils import satisfy_requirement

if satisfy_requirement('salesforce-lavis'):
    from .instructblip import *  # noqa: F401, F403

if osp.exists('opencompass/multimodal/models/minigpt_4/MiniGPT-4'):
    from .minigpt_4 import *  # noqa: F401, F403

from .llava import *  # noqa: F401, F403
from .openflamingo import *  # noqa: F401, F403
from .visualglm import *  # noqa: F401, F403
