# flake8: noqa

from mmengine.config import read_base

from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask
from opencompass.models import OpenAISDKRollout

#######################################################################
#                          PART 0  Essential Configs                  #
#######################################################################
with read_base():
    # Datasets
    from opencompass.configs.datasets.aime2025.aime2025_cascade_eval_gen_5e9f4f import aime2025_datasets

#######################################################################
#                          PART 1  Datasets List                      #
#######################################################################

# datasets list for evaluation
datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')),
               [])

num_repeat = 4
for item in datasets:
    item['abbr'] += f'_rollout_rep{num_repeat}'
    item['n'] = num_repeat

#######################################################################
#                        PART 2  Models  List                         #
#######################################################################

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ]
)

models = [
    dict(
        abbr='YOUR_MODEL',
        key='YOUR_API_KEY',
        openai_api_base='YOUR_API_BASE',
        type=OpenAISDKRollout,
        path='YOUR_MODEL_PATH',
        temperature=1.0,
        meta_template=api_meta_template,
        query_per_second=1,
        batch_size=16,
        max_out_len=65536,
        max_seq_len=65536,
        retry=10,
        extra_body=dict(
            top_k=20,
        ),
        openai_extra_kwargs=dict(
            top_p=0.95,
        ),
    )
]

#######################################################################
#                 PART 3  Inference/Evaluation Configuaration         #
#######################################################################

# infer with local runner
infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=8),
    runner=dict(
        type=LocalRunner,
        max_num_workers=16,
        retry=0,  # Modify if needed
        task=dict(type=OpenICLInferTask),
    ),
)

# eval with local runner
eval = dict(
    partitioner=dict(type=NaivePartitioner, n=10),
    runner=dict(type=LocalRunner,
                max_num_workers=16,
                task=dict(type=OpenICLEvalTask)),
)

#######################################################################
#                      PART 4  Utils Configuaration                   #
#######################################################################

work_dir = './outputs/oc_rollout_eval'