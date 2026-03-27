from mmengine.config import read_base

from opencompass.models import OpenAISDK
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

with read_base():
    # choose a list of datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen import \
        gsm8k_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.race.race_gen import \
        race_datasets  # noqa: F401, E501

API_BASE = 'http://localhost:23333/v1'
MODEL_PATH = 'Qwen/Qwen3-8B'
TOKENIZER_PATH = 'Qwen/Qwen3-8B'

models = [
    dict(
        abbr='mock_test',
        type=OpenAISDK,
        key='EMPTY',
        openai_api_base=API_BASE,
        path=MODEL_PATH,
        tokenizer_path=TOKENIZER_PATH,
        rpm_verbose=True,
        meta_template=dict(round=[
            dict(role='SYSTEM', api_role='SYSTEM'),
            dict(role='HUMAN', api_role='HUMAN'),
            dict(role='BOT', api_role='BOT', generate=True),
        ]),
        temperature=0,
        max_workers=1024,
        mode='mid',
        retry=3,
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]

race_datasets = [race_datasets[1]]
datasets = sum([v for k, v in locals().items() if k.endswith('_datasets')], [])

for d in datasets:
    d['reader_cfg']['test_range'] = '[0:4]'

summarizer = dict(
    dataset_abbrs=[
        'gsm8k',
        'race-middle',
        'race-high',
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)
