from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask


with read_base():
    # If you want to evaluate the full scireasoner dataset (more than one million samples)
    from opencompass.configs.datasets.SciReasoner1_5.scireasoner1_5_gen import scireasoner1_5_datasets

    # If you only want to evaluate the miniset
    from opencompass.configs.datasets.SciReasoner1_5.scireasoner1_5_gen import mini_scireasoner1_5_datasets

    from opencompass.configs.summarizers.scireasoner1_5 import SciReasoner15Summarizer


datasets = sum(
    (v for k, v in locals().items() if k.endswith('_datasets')),
    [],
)

summarizer = dict(
    type=SciReasoner15Summarizer,
    mini_set=False,  # When evaluating miniset, please set True
    show_details=False  # Whether you want to see the detailed results for each subset
)

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=8),
    runner=dict(
        type=LocalRunner,
        max_num_workers=16,
        task=dict(type=OpenICLInferTask),
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner, n=10),
    runner=dict(
        type=LocalRunner,
         max_num_workers=16,
        task=dict(type=OpenICLEvalTask)
    ),
)


work_dir = './outputs/eval_scireasoner1_5'


