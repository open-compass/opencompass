from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.subjective.compass_arena_subjective_bench.singleturn.pairwise_judge import compassarena_subjectivebench_singleturn_datasets
    from opencompass.configs.datasets.subjective.compass_arena_subjective_bench.multiturn.pairwise_judge import compassarena_subjectivebench_multiturn_datasets

    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_5_7b_chat import models as lmdeploy_internlm2_5_7b_chat
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_5_20b_chat import models as lmdeploy_internlm2_5_20b_chat
    from opencompass.configs.models.hf_llama.lmdeploy_llama3_1_8b_instruct import models as lmdeploy_llama3_1_8b_instruct
    from opencompass.configs.models.hf_llama.lmdeploy_llama3_1_70b_instruct import models as lmdeploy_llama3_1_70b_instruct
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_0_5b_instruct import models as lmdeploy_qwen2_5_0_5b_instruct
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_1_5b_instruct import models as lmdeploy_qwen2_5_1_5b_instruct
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_3b_instruct import models as lmdeploy_qwen2_5_3b_instruct
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_7b_instruct import models as lmdeploy_qwen2_5_7b_instruct
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_14b_instruct import models as lmdeploy_qwen2_5_14b_instruct
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_32b_instruct import models as lmdeploy_qwen2_5_32b_instruct
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_72b_instruct import models as lmdeploy_qwen2_5_72b_instruct
    from opencompass.configs.models.qwen.lmdeploy_qwen2_7b_instruct import models as lmdeploy_qwen2_7b_instruct

from opencompass.models import (HuggingFace, HuggingFaceCausalLM,
                                HuggingFaceChatGLM3, OpenAI,
                                TurboMindModelwithChatTemplate)
from opencompass.partitioners import NaivePartitioner, SizePartitioner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.partitioners.sub_num_worker import \
    SubjectiveNumWorkerPartitioner
from opencompass.partitioners.sub_size import SubjectiveSizePartitioner
from opencompass.runners import LocalRunner, SlurmSequentialRunner
from opencompass.summarizers import DefaultSubjectiveSummarizer
from opencompass.tasks import OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask

api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
])

# -------------Inference Stage ----------------------------------------
# For subjective evaluation, we often set do sample for models
# models = [
#     dict(
#         type=TurboMindModelwithChatTemplate,
#         abbr='CompassJudger-1-7B-Instruct',
#         path='opencompass/CompassJudger-1-7B-Instruct',
#         engine_config=dict(session_len=16384, max_batch_size=16, tp=1),
#         gen_config=dict(top_k=1, temperature=1e-6, top_p=0.9, max_new_tokens=2048),
#         max_seq_len=16384,
#         max_out_len=2048,
#         batch_size=16,
#         run_cfg=dict(num_gpus=1),
#     )
# ]

models = [
    *lmdeploy_qwen2_5_14b_instruct, *lmdeploy_qwen2_5_32b_instruct,
    *lmdeploy_qwen2_5_7b_instruct, *lmdeploy_qwen2_7b_instruct
]

datasets = [
    *compassarena_subjectivebench_singleturn_datasets,
    *compassarena_subjectivebench_multiturn_datasets
]  # add datasets you want

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner,
                max_num_workers=16,
                task=dict(type=OpenICLInferTask)),
)
# -------------Evalation Stage ----------------------------------------

## ------------- JudgeLLM Configuration
judge_models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='CompassJudger-1-32B-Instruct',
        path='opencompass/CompassJudger-1-32B-Instruct',
        engine_config=dict(session_len=16384, max_batch_size=16, tp=4),
        gen_config=dict(top_k=1,
                        temperature=1e-6,
                        top_p=0.9,
                        max_new_tokens=2048),
        max_seq_len=16384,
        max_out_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=4),
    )
]

## ------------- Evaluation Configuration
eval = dict(
    partitioner=dict(
        type=SubjectiveNaivePartitioner,
        models=models,
        judge_models=judge_models,
    ),
    runner=dict(type=LocalRunner,
                max_num_workers=16,
                task=dict(type=SubjectiveEvalTask)),
)

summarizer = dict(type=DefaultSubjectiveSummarizer, )
work_dir = 'outputs/subjective/'
