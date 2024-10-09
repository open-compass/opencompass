import os.path as osp
from copy import deepcopy

from mmengine.config import read_base
from opencompass.models import (HuggingFacewithChatTemplate,
                                TurboMindModelwithChatTemplate)
from opencompass.models.openai_api import OpenAI, OpenAISDK
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.runners import DLCRunner, LocalRunner
from opencompass.summarizers import SubjectiveSummarizer
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask


#######################################################################
#                          PART 0  Essential Configs                  #
#######################################################################
with read_base():
    # Datasets Part
    from opencompass.configs.datasets.subjective.arena_hard.arena_hard_compare import \
        arenahard_datasets
    from opencompass.configs.datasets.subjective.alignbench.alignbench_v1_1_judgeby_critiquellm import \
        alignbench_datasets
    from opencompass.configs.datasets.subjective.multiround.mtbench_single_judge_diff_temp import \
        mtbench_datasets

    # Summarizer

    # Model List
    # from opencompass.configs.models.qwen.lmdeploy_qwen2_1_5b_instruct import models as lmdeploy_qwen2_1_5b_instruct_model
    # from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_5_7b_chat import models as hf_internlm2_5_7b_chat_model


#######################################################################
#                          PART 1  Datasets List                      #
#######################################################################
# datasets list for evaluation

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])


#######################################################################
#                       PART 2  Datset Summarizer                     #
#######################################################################
summarizer = dict(type=SubjectiveSummarizer, function='subjective')

#######################################################################
#                        PART 3  Models  List                         #
#######################################################################

models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='internlm2_5-7b-chat-turbomind',
        path='internlm/internlm2_5-7b-chat',
        engine_config=dict(session_len=16384, max_batch_size=16, tp=1),
        gen_config=dict(top_k=40, temperature=1.0, top_p=0.9, max_new_tokens=4096),
        max_seq_len=16384,
        max_out_len=4096,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
    )
]

models = sum([v for k, v in locals().items() if k.endswith('_model')], models)



#######################################################################
#                 PART 4  Inference/Evaluation Configuaration         #
#######################################################################

# Local Runner
infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        num_worker=8
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=16,
        retry=0, # Modify if needed
        task=dict(type=OpenICLInferTask)
    ),
)

# JudgeLLM
api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
])


judge_models = [
    dict(
        type=OpenAISDK,
        abbr='gpt-4o-2024-08-06',
        path='gpt-4o-2024-08-06',
        # openai_api_base=
        # 'http://10.140.1.86:10001/v1',  # Change to your own url if needed.
        key='YOUR_API_KEY',
        retry=10,
        meta_template=api_meta_template,
        rpm_verbose=True,
        query_per_second=1,
        max_out_len=4096,
        max_seq_len=16384,
        batch_size=16,
        temperature=0.01,
        tokenizer_path='gpt-4o-2024-08-06'
    )
]

# Evaluation with local runner
eval = dict(
    partitioner=dict(
        type=SubjectiveNaivePartitioner,
        models=models,
        judge_models=judge_models,
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=16,
        task=dict(type=SubjectiveEvalTask)),
)



#######################################################################
#                      PART 5  Utils Configuaration                   #
#######################################################################
base_exp_dir = 'outputs/corebench/'
work_dir = osp.join(base_exp_dir, 'chat_subjective')
