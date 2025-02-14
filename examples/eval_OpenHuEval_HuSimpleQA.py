from mmengine.config import read_base

from opencompass.summarizers.subjective.husimpleqa import HuSimpleQASummarizer

with read_base():
    from opencompass.configs.datasets.OpenHuEval.HuSimpleQA.HuSimpleQA import HuSimpleQA_datasets, PROMPT_LANGUAGES

    from opencompass.configs.models.openai.gpt_4o_mini_20240718 import models as gpt_4o_mini_20240718_model
    from opencompass.configs.models.openai.gpt_4o_2024_11_20 import models as gpt_4o_2024_11_20_model
    from opencompass.configs.models.deepseek.deepseek_v3_api_aliyun import models as deepseek_v3_api_aliyun_model

    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_7b_instruct import models as lmdeploy_qwen2_5_7b_instruct_model
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_72b_instruct import models as lmdeploy_qwen2_5_72b_instruct_model
    from opencompass.configs.models.hf_llama.lmdeploy_llama3_1_8b_instruct import models as lmdeploy_llama3_1_8b_instruct_model
    from opencompass.configs.models.hf_llama.lmdeploy_llama3_1_70b_instruct import models as lmdeploy_llama3_1_70b_instruct_model

    # from opencompass.configs.models.hf_internlm.lmdeploy_internlm3_8b_instruct import models as lmdeploy_internlm3_8b_instruct_model

    from opencompass.configs.models.qwq.lmdeploy_qwq_32b_preview import models as lmdeploy_qwq_32b_preview_model
    from opencompass.configs.models.deepseek.deepseek_r1_api_aliyun import models as deepseek_r1_api_aliyun_model
    from opencompass.configs.models.openai.o1_mini_2024_09_12 import models as o1_mini_2024_09_12_model
    # from opencompass.configs.models.openai.o3_mini_2025_01_31 import models as o3_mini_2025_01_31_model

from opencompass.models import OpenAI
from opencompass.partitioners import (
    NumWorkerPartitioner,
    SubjectiveNumWorkerPartitioner,
)
from opencompass.runners import LocalRunner, SlurmSequentialRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask

api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
])

models = sum([v for k, v in locals().items() if k.endswith('_model')], [])

for model in models:
    if model['abbr'].startswith('deepseek_r1_api_'):
        model['return_reasoning_content'] = True
        model['pred_postprocessor'] = {
            'OpenHuEval_*': {
                'type': 'rm_<think>_before_eval'
            }
        }
    if model['abbr'].startswith('QwQ'):
        model['pred_postprocessor'] = {
            'OpenHuEval_*': {
                'type': 'extract_qwq_answer_before_eval_for_husimpleqa'
            }
        }
del model


judge_models = [
    dict(
        abbr='GPT-4o-2024-08-06',
        type=OpenAI,
        path='gpt-4o-2024-08-06',
        key='ENV',
        openai_proxy_url='ENV',
        verbose=True,
        meta_template=api_meta_template,
        query_per_second=2,
        max_out_len=8192,
        max_seq_len=16384,
        batch_size=8,
        temperature=0,
    )
]

datasets = HuSimpleQA_datasets
del HuSimpleQA_datasets

infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        num_worker=8,
    ),
    runner=dict(
        type=SlurmSequentialRunner,
        max_num_workers=16,
        task=dict(type=OpenICLInferTask),
    ),
)

eval = dict(
    partitioner=dict(
        type=SubjectiveNumWorkerPartitioner,
        num_worker=8,
        models=models,
        judge_models=judge_models,
    ),
    runner=dict(type=LocalRunner,
                max_num_workers=16,
                task=dict(type=SubjectiveEvalTask)),
)

summarizer = dict(type=HuSimpleQASummarizer, 
                  prompt_languages=PROMPT_LANGUAGES)

work_dir = (
    './outputs/' + __file__.split('/')[-1].split('.')[0] + '/' 
)  # do NOT modify this line, yapf: disable, pylint: disable
