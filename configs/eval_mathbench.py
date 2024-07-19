from mmengine.config import read_base

with read_base():

    # Import models
    from .models.hf_llama.hf_llama3_8b_instruct import models as llama3_8b_instruct_model
    from .models.hf_internlm.hf_internlm2_chat_7b import models as internlm2_chat_7b_model

    # Import datasets
    from .datasets.MathBench.mathbench_gen import mathbench_datasets

    # Import summarizers for display results
    from .summarizers.groups.mathbench_v1_2024 import summarizer # Grouped results for MathBench-A and MathBench-T separately
    # from .summarizers.mathbench_v1 import summarizer # Detailed results for every sub-dataset
    # from .summarizers.groups.mathbench_v1_2024_lang import summarizer # Grouped results for bilingual results

datasets = sum([v for k, v in locals().items() if k.endswith('_datasets')], [])
models = sum([v for k, v in locals().items() if k.endswith('_model')], [])

from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

eval = dict(
    partitioner=dict(type=NaivePartitioner, n=8),
    runner=dict(
        type=LocalRunner,
        max_num_workers=256,
        task=dict(type=OpenICLEvalTask)
    ),
)

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=4),
    runner=dict(
        type=LocalRunner,
        max_num_workers=256,
        task=dict(type=OpenICLInferTask)
    ),
)

work_dir = './outputs/mathbench_results'
