from mmengine.config import read_base

with read_base():
    from .datasets.subjective.creationbench.creationbench_checklist import checklist_datasets
from opencompass.models import OpenAI
from opencompass.models import HuggingFacewithChatTemplate
from opencompass.models import TurboMindModelwithChatTemplate
from opencompass.models import VLLMwithChatTemplate
from opencompass.partitioners import NumWorkerPartitioner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.partitioners.sub_size import SubjectiveSizePartitioner
from opencompass.partitioners.sub_num_worker import SubjectiveNumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.runners import SlurmSequentialRunner
from opencompass.tasks.openicl_infer import OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask
from opencompass.summarizers.subjective.compassbench_v13 import CompassBenchSummarizer

datasets = checklist_datasets

# -------------Evalation Stage ----------------------------------------

## ------------- JudgeLLM Configuration
api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ]
)

models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='glm4-chat-9b-hf',
        path='internlm/internlm2-chat-1_8b',
        engine_config=dict(session_len=7168, max_batch_size=16, tp=1),
        gen_config=dict(top_k=1000, temperature=1, top_p=0.9, max_new_tokens=2048),
        max_seq_len=7168,
        max_out_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
    ),
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='qwen1.5-7b-chat-hf',
        path='internlm/internlm2-chat-1_8b',
        engine_config=dict(session_len=7168, max_batch_size=16, tp=1),
        gen_config=dict(top_k=1000, temperature=1, top_p=0.9, max_new_tokens=2048),
        max_seq_len=7168,
        max_out_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
    ),
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='qwen2-7b-instruct-hf',
        path='internlm/internlm2-chat-1_8b',
        engine_config=dict(session_len=7168, max_batch_size=16, tp=1),
        gen_config=dict(top_k=1000, temperature=1, top_p=0.9, max_new_tokens=2048),
        max_seq_len=7168,
        max_out_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
    )
]


judge_models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='qwen2-72b-instruct-turbomind',
        path='Qwen/Qwen2-72B-Instruct',
        engine_config=dict(session_len=7168, max_batch_size=16, tp=4),
        gen_config=dict(top_k=1, temperature=1e-6, top_p=0.9, max_new_tokens=1024),
        max_seq_len=7168,
        max_out_len=1024,
        batch_size=16,
        run_cfg=dict(num_gpus=4),
    )
]

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=1),
    runner=dict(
        type=SlurmSequentialRunner,
        max_num_workers=32,
        quotatype='reserved',
        partition='llmeval',
        task=dict(type=OpenICLInferTask)),
)
## ------------- Evaluation Configuration
eval = dict(
    partitioner=dict(type=SubjectiveNumWorkerPartitioner, num_worker=1, models=models, judge_models=judge_models,),
    runner=dict(
        type=SlurmSequentialRunner,
        max_num_workers=32,
        quotatype='reserved',
        partition='llmeval',
        task=dict(type=SubjectiveEvalTask)),
    #runner=dict(type=LocalRunner, max_num_workers=16, task=dict(type=SubjectiveEvalTask)),
)

summarizer = dict(type=CompassBenchSummarizer)
work_dir = 'outputs/creationbench/'
