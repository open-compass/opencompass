from mmengine.config import read_base

from opencompass.models import OpenAISDK
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    from .opencompass.configs.datasets.perspectivegap.perspectivegap_role_assignment_gen import perspectivegap_role_assignment_datasets
    from .opencompass.configs.datasets.perspectivegap.perspectivegap_prompt_writing_gen import perspectivegap_prompt_writing_datasets

api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
])

models = [
    dict(
        abbr='deepseek-v4-pro',
        type=OpenAISDK,
        path='deepseek-v4-pro',
        key='ENV',
        meta_template=api_meta_template,
        query_per_second=5,
        openai_api_base='https://api.deepseek.com/v1',
        batch_size=16,
        max_seq_len=65536,
        retry=5,
    ),
]

datasets = perspectivegap_role_assignment_datasets + perspectivegap_prompt_writing_datasets

work_dir = './outputs/perspectivegap_test'
