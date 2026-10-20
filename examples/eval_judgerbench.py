from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.subjective.judgerbench.judgerbench import judgerbench_datasets

from opencompass.models import (HuggingFace, HuggingFaceCausalLM,
                                HuggingFaceChatGLM3, OpenAI,
                                TurboMindModelwithChatTemplate)
from opencompass.partitioners import NaivePartitioner, SizePartitioner
from opencompass.runners import LocalRunner, SlurmSequentialRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
])

# -------------Inference Stage ----------------------------------------
# For subjective evaluation, we often set do sample for models
models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='CompassJudger-1-7B-Instruct',
        path='opencompass/CompassJudger-1-7B-Instruct',
        engine_config=dict(session_len=16384, max_batch_size=16, tp=1),
        gen_config=dict(top_k=1,
                        temperature=1e-6,
                        top_p=0.9,
                        max_new_tokens=2048),
        max_seq_len=16384,
        max_out_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
    )
]

datasets = judgerbench_datasets

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner,
                max_num_workers=16,
                task=dict(type=OpenICLInferTask)),
)
# -------------Evalation Stage ----------------------------------------

## ------------- Evaluation Configuration
eval = dict(
    partitioner=dict(
        type=NaivePartitioner,
        n=10,
    ),
    runner=dict(type=LocalRunner,
                max_num_workers=16,
                task=dict(type=OpenICLEvalTask)),
)

work_dir = 'outputs/judgerbench/'
