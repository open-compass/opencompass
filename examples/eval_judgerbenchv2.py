from mmengine.config import read_base
with read_base():
    from opencompass.configs.datasets.judge.judgerbenchv2 import get_judgerbenchv2_dataset
    from opencompass.configs.summarizers.judgerbenchv2 import summarizer
from opencompass.models import HuggingFaceCausalLM, HuggingFace, HuggingFaceChatGLM3, OpenAI
from opencompass.partitioners import NaivePartitioner, SizePartitioner, NumWorkerPartitioner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.partitioners.sub_size import SubjectiveSizePartitioner
from opencompass.partitioners.sub_num_worker import SubjectiveNumWorkerPartitioner
from opencompass.runners import LocalRunner, DLCRunner, VOLCRunner
from opencompass.runners import SlurmSequentialRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ]
)
datasets = [*get_judgerbenchv2_dataset]

from opencompass.models import TurboMindModelwithChatTemplate

models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='qwen-7b-hf',
        path='Qwen/Qwen-7B',
        engine_config=dict(session_len=16384, max_batch_size=16, tp=1),
        gen_config=dict(top_k=1, temperature=1e-6, top_p=0.9, max_new_tokens=2048),
        max_seq_len=16384,
        max_out_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
    ),
]


infer = dict(
    # partitioner=dict(type=NaivePartitioner),
    partitioner=dict(type=NumWorkerPartitioner, num_worker=2),
    runner=dict(
        type=LocalRunner,
        max_num_workers=72,
        task=dict(type=OpenICLInferTask),
    ),
)



work_dir = './outputs/judgerbenchv2/'
