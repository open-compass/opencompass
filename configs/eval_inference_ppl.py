from mmengine.config import read_base

with read_base():
    # LLM compression datasets
    from .datasets.inference_ppl.inference_ppl import inference_ppl_datasets

    # Model configs
    from .models.qwen.hf_qwen1_5_7b import models as qwen1_5_7b
    from .models.qwen.hf_qwen1_5_14b import models as qwen1_5_14b
    from .models.hf_llama.hf_llama2_7b import models as llama2_7b
    from .models.hf_llama.hf_llama2_13b import models as llama2_13b
    from .models.hf_llama.hf_llama2_70b import models as llama2_70b
    from .models.hf_llama.hf_llama_7b import models as llama_7b
    from .models.hf_llama.hf_llama_13b import models as llama_13b
    from .models.hf_llama.hf_llama_30b import models as llama_30b
    from .models.yi.hf_yi_6b import models as yi_6b
    from .models.yi.hf_yi_34b import models as yi_34b
    from .models.yi.hf_yi_1_5_6b import models as yi_1_5_6b
    from .models.yi.hf_yi_1_5_9b import models as yi_1_5_9b
    from .models.yi.hf_yi_1_5_34b import models as yi_1_5_34b
    from .models.deepseek.hf_deepseek_7b_base import models as deepseek_7b_base
    from .models.deepseek.hf_deepseek_67b_base import models as deepseek_67b_base
    from .models.qwen.hf_qwen_1_8b import models as qwen_1_8b
    from .models.qwen.hf_qwen_7b import models as qwen_7b
    from .models.qwen.hf_qwen_14b import models as qwen_14b
    from .models.qwen.hf_qwen_72b import models as qwen_72b
    from .models.hf_internlm.hf_internlm_7b import models as internlm_7b
    from .models.hf_internlm.hf_internlm_20b import models as internlm_20b
    from .models.hf_internlm.hf_internlm2_7b import models as internlm2_7b
    from .models.hf_internlm.hf_internlm2_20b import models as internlm2_20b
    from .models.qwen.hf_qwen1_5_4b_chat import models as qwen1_5_4b_chat
    from .models.qwen.hf_qwen1_5_7b_chat import models as qwen1_5_7b_chat
    from .models.qwen.hf_qwen1_5_14b_chat import models as qwen1_5_14b_chat
    



from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask


# -------------Inference Stage ----------------------------------------
datasets = [*inference_ppl_datasets]
workdir = 'outputs/inference_ppl'

models = [
    *qwen1_5_7b,
    # *qwen1_5_14b,
    # *llama2_7b,
    # *llama2_13b,
    # *llama2_70b,
    # *llama_7b,
    # *llama_13b,
    # *llama_30b
    # *yi_6b,
    # *yi_34b,
    # *yi_1_5_6b,
    # *yi_1_5_9b,
    # *yi_1_5_34b
    # *deepseek_7b_base,
    # *deepseek_67b_base
    # *qwen_1_8b,
    # *qwen_7b,
    # *qwen_14b,
    # *qwen_72b,
    # *internlm_7b,
    # *internlm_20b,
    # *internlm2_7b,
    # *internlm2_20b,
    # *qwen1_5_4b_chat,
    # *qwen1_5_7b_chat,
    # *qwen1_5_14b_chat
]



# Set custom batch_size and num_gpus for faster loss calculation
# Smaller batch_size should give more precise results, at the cost of worse performance
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

