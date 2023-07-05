from mmengine.config import read_base

with read_base():
    from .SuperGLUE_WiC_ppl_4118db import WiC_datasets  # noqa: F401, F403
