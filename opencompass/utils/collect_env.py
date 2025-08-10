from mmengine.utils import get_git_hash
from mmengine.utils.dl_utils import collect_env as collect_base_env

import opencompass


def collect_env():
    """Collect the information of the running environments."""
    env_info = collect_base_env()
    env_info['opencompass'] = opencompass.__version__ + '+' + get_git_hash(
    )[:7]

    # LMDeploy
    try:
        import lmdeploy
        env_info['lmdeploy'] = lmdeploy.__version__
    except ModuleNotFoundError as e:
        env_info['lmdeploy'] = f'not installed:{e}'
    # Transformers
    try:
        import transformers
        env_info['transformers'] = transformers.__version__
    except ModuleNotFoundError as e:
        env_info['transformers'] = f'not installed:{e}'

    return env_info
