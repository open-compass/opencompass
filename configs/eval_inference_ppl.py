from mmengine.config import read_base

with read_base():
    # Inference PPL datasets
    from .datasets.inference_ppl.inference_ppl import inference_ppl_datasets

    # Model configs
    from .models.qwen.hf_qwen1_5_7b import models as qwen1_5_7b
    from .models.qwen.hf_qwen1_5_14b import models as qwen1_5_14b
    from .models.hf_llama.hf_llama2_7b import models as llama2_7b
    from .models.hf_llama.hf_llama2_13b import models as llama2_13b


from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask


# -------------Inference Stage ----------------------------------------

datasets = [*inference_ppl_datasets]
workdir = 'outputs/inference_ppl'

models = [
    *qwen1_5_7b,
    *qwen1_5_14b,
    *llama2_7b,
    *llama2_13b,
]



# Set custom batch_size and num_gpus for faster loss calculation
# Smaller batch_size should give more precise results, at the cost of worse efficiency
model_cfg = dict(
    batch_size=8,
    run_cfg=dict(num_gpus=4, num_procs=1)
)

for mdl in models:
    mdl.update(model_cfg)


infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLInferTask),
        max_num_workers=256,  # Maximum concurrent evaluation task count
    ),
)


# -------------Evaluation Stage ----------------------------------------
eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLEvalTask),
        max_num_workers=256,
    )
)
