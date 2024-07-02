from mmengine.config import read_base

from opencompass.models import OpenAI
from opencompass.runners import LocalRunner
from opencompass.partitioners.sub_size import SubjectiveSizePartitioner
from opencompass.tasks.subjective_eval import SubjectiveEvalTask
from opencompass.summarizers import CharmMemSummarizer

with read_base():
    from .datasets.CHARM.charm_memory_gen_bbbd53 import charm_memory_datasets as datasets

    # ------>>>>>> https://arxiv.org/abs/2403.14112
    # from .models.openai.gpt_3_5_turbo_1106 import models as gpt_3_5_turbo_1106_model
    # from .models.openai.gpt_4_1106_preview import models as gpt_4_1106_preview_model
    # from .models.hf_llama.hf_llama2_7b_chat import models as llama2_7b_chat_model
    # from .models.hf_llama.hf_llama2_13b_chat import models as llama2_13b_chat_model
    # from .models.hf_llama.hf_llama2_70b_chat import models as llama2_70b_chat_model
    # from .models.vicuna.hf_vicuna_7b_v15_16k import models as vicuna_7b_v15_16k_model
    # from .models.vicuna.hf_vicuna_13b_v15_16k import models as vicuna_13b_v15_16k_model
    # from .models.chatglm.hf_chatglm3_6b_32k import models as chatglm3_6b_32k_model
    # from .models.baichuan.hf_baichuan2_7b_chat import models as baichuan2_7b_chat_model  # need torch 2.1
    # from .models.baichuan.hf_baichuan2_13b_chat import models as baichuan2_13b_chat_model  # need torch 2.1
    # from .models.hf_internlm.hf_internlm2_chat_7b import models as hf_internlm2_chat_7b_model
    # from .models.hf_internlm.hf_internlm2_chat_20b import models as hf_internlm2_chat_20b_model
    # from .models.yi.hf_yi_6b_chat import models as yi_6b_chat_model
    # from .models.yi.hf_yi_34b_chat import models as yi_34b_chat_model
    # from .models.deepseek.hf_deepseek_7b_chat import models as deepseek_7b_chat_model
    # from .models.deepseek.hf_deepseek_67b_chat import models as deepseek_67b_chat_model
    # from .models.qwen.hf_qwen_7b_chat import models as qwen_7b_chat_model
    # from .models.qwen.hf_qwen_14b_chat import models as qwen_14b_chat_model
    # from .models.qwen.hf_qwen_72b_chat import models as qwen_72b_chat_model
    # <<<<<<------ https://arxiv.org/abs/2403.14112

    # from .models.openai.gpt_3_5_turbo_0125 import models as gpt_3_5_turbo_0125_model
    # from .models.openai.gpt_4o_2024_05_13 import models as gpt_4o_2024_05_13_model
    # from .models.gemini.gemini_1_5_flash import models as gemini_1_5_flash_model
    # from .models.gemini.gemini_1_5_pro import models as gemini_1_5_pro_model

    # from .models.hf_llama.lmdeploy_llama3_8b_instruct import models as lmdeploy_llama3_8b_instruct_model
    # from .models.hf_llama.lmdeploy_llama3_70b_instruct import models as lmdeploy_llama3_70b_instruct_model

    # from .models.hf_internlm.lmdeploy_internlm2_chat_1_8b import models as lmdeploy_internlm2_chat_1_8b_model
    # from .models.hf_internlm.lmdeploy_internlm2_chat_7b import models as lmdeploy_internlm2_chat_7b_model
    # from .models.hf_internlm.lmdeploy_internlm2_chat_20b import models as lmdeploy_internlm2_chat_20b_model

    # from .models.yi.hf_yi_1_5_6b_chat import models as yi_1_5_6b_chat_model
    # from .models.yi.hf_yi_1_5_34b_chat import models as yi_1_5_34b_chat_model

    # from .models.deepseek.hf_deepseek_v2_chat import models as deepseek_v2_chat_model

    # from .models.qwen.hf_qwen1_5_1_8b_chat import models as qwen1_5_1_8b_chat_model
    # from .models.qwen.hf_qwen1_5_7b_chat import models as qwen1_5_7b_chat_model
    # from .models.qwen.hf_qwen1_5_14b_chat import models as qwen1_5_14b_chat_model
    # from .models.qwen.hf_qwen1_5_72b_chat import models as qwen1_5_72b_chat_model

models = sum([v for k, v in locals().items() if k.endswith('_model')], [])

## ------------- JudgeLLM Configuration
api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
])
judge_models = [
    dict(
        abbr='GPT-3.5-turbo-0125',
        type=OpenAI,
        path='gpt-3.5-turbo-0125',
        key='ENV',
        meta_template=api_meta_template,
        query_per_second=16,
        max_out_len=2048,
        max_seq_len=2048,
        batch_size=8,
        temperature=0,
    )
]

## ------------- Evaluation Configuration
eval = dict(
    partitioner=dict(
        type=SubjectiveSizePartitioner,
        max_task_size=1000,
        mode='singlescore',
        models=models,
        judge_models=judge_models,
    ),
    runner=dict(type=LocalRunner,
                max_num_workers=2,
                task=dict(type=SubjectiveEvalTask)),
)

summarizer = dict(type=CharmMemSummarizer)

work_dir = './outputs/CHARM_mem/chat/'
