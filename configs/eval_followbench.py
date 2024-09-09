from mmengine.config import read_base

with read_base():
    from .datasets.subjective.followbench.followbench_llmeval import followbench_llmeval_dataset
    from .datasets.subjective.followbench.followbench_rulebase import followbench_rulebase_dataset
from opencompass.models import TurboMindModelwithChatTemplate
from opencompass.models import HuggingFaceCausalLM, HuggingFace, HuggingFaceChatGLM3, OpenAI
from opencompass.partitioners import NaivePartitioner, SizePartitioner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.partitioners.sub_size import SubjectiveSizePartitioner
from opencompass.partitioners.sub_num_worker import SubjectiveNumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.runners import SlurmSequentialRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask
from opencompass.summarizers import SubjectiveSummarizer

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ]
)

# -------------Inference Stage ----------------------------------------
# For subjective evaluation, we often set do sample for models
models = [    
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='0822_internlm2.5_7b_lb_r8_nosft_addray1:0.5_selfcollect0826_e2_lr2e-5',
        path='/mnt/petrelfs/caomaosong/backup_hwfile/compassjudger/xtuner/work_dir/0822_internlm2.5_7b_lb_r8_nosft_addray1:0.5_selfcollect0826_e2_lr2e-5/hf',
        engine_config=dict(session_len=7168, max_batch_size=16, tp=1),
        gen_config=dict(top_k=1, temperature=1e-6, top_p=0.9, max_new_tokens=1024),
        max_seq_len=7168,
        max_out_len=1024,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
    )
]


datasets = [*followbench_llmeval_dataset]

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner, max_num_workers=16, task=dict(type=OpenICLInferTask)),
)
# -------------Evalation Stage ----------------------------------------

## ------------- JudgeLLM Configuration
judge_models = models

## ------------- Evaluation Configuration
eval = dict(
    partitioner=dict(type=SubjectiveNaivePartitioner, models=models, judge_models=judge_models,),
    runner=dict(type=LocalRunner, max_num_workers=16, task=dict(type=SubjectiveEvalTask)),
)

summarizer = dict(type=SubjectiveSummarizer, function='subjective')
work_dir = 'outputs/followbench/'
