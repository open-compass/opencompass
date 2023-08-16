from opencompass.utils import satisfy_requirement

if satisfy_requirement('salesforce-lavis'):
    from .instructblip import *  # noqa: F401, F403

from .llava import *  # noqa: F401, F403
from .minigpt_4 import *  # noqa: F401, F403
from .visualglm import *  # noqa: F401, F403
