
from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.eese.eese_judge_gen import \
        eese_datasets
    # 选择一个感兴趣的模型
    from opencompass.configs.models.openai.gpt_4o_2024_05_13 import \
        models as gpt4

from opencompass.models import OpenAISDK

# 配置评判模型
api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
], )

judge_cfg = dict(
    abbr='model-judge',
    type=OpenAISDK,
    path='model-name',
    key='your-api-key',
    openai_api_base=['openai-url'],
    meta_template=api_meta_template,
    query_per_second=16,
    batch_size=1,
    temperature=0.001,
    tokenizer_path='gpt-4o',
    verbose=True,
    max_out_len=16384,
    max_seq_len=49152,
)

datasets = eese_datasets
models = gpt4

# 为每个数据集增加judge_cfg信息，而不是覆盖
for dataset in datasets:
    if 'eval_cfg' in dataset and 'evaluator' in dataset['eval_cfg']:
        # 获取现有的judge_cfg，如果不存在则创建空字典
        existing_judge_cfg = dataset['eval_cfg']['evaluator'].get('judge_cfg', {})
        # 更新现有的judge_cfg，保留原有配置并添加新配置
        existing_judge_cfg.update(judge_cfg)
        # 将更新后的配置设置回去
        dataset['eval_cfg']['evaluator']['judge_cfg'] = existing_judge_cfg

