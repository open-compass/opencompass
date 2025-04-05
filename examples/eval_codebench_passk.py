from mmengine.config import read_base
import os.path as osp
from opencompass.runners import LocalRunner, VOLCRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    # Datasets Part
    # bigcodebench
    from opencompass.configs.datasets.bigcodebench.bigcodebench_full_instruct_gen_c3d5ad import (
        bigcodebench_full_instruct_datasets
    )
    from opencompass.configs.datasets.bigcodebench.bigcodebench_hard_instruct_gen_c3d5ad import (
        bigcodebench_hard_instruct_datasets
    )
    # livecodebench code generation lite v5
    from opencompass.configs.datasets.livecodebench.livecodebench_time_split_gen import (
        LCB_datasets
    )
    # huamneval
    from opencompass.configs.datasets.humaneval.humaneval_passk_gen_8e312c import (
        humaneval_datasets
    )
    from opencompass.configs.datasets.humaneval_pro.humaneval_pro_gen import (
        humanevalpro_datasets
    )
    # mbpp
    from opencompass.configs.datasets.mbpp.mbpp_passk_gen_830460 import (
        mbpp_datasets
    )
    from opencompass.configs.datasets.mbpp_pro.mbpp_pro_gen import (
        mbpppro_datasets
    )
    # multipl-e
    from opencompass.configs.datasets.multipl_e.multiple_top_ten_gen import (
        multiple_datasets
    )

    # Models Part
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_7b_instruct import (
        models as lmdeploy_qwen2_5_7b_instruct_model,
    )
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm3_8b_instruct import (
        models as lmdeploy_internlm3_8b_instruct_model,
    )

# models config
models = sum([v for k, v in locals().items() if k.endswith('_model')], [])
for model in models:
    model['max_seq_len'] = 16384
    model['max_out_len'] = 8192

# datasets config
datasets = sum(
    (v for k, v in locals().items() if k.endswith('_datasets')),
    [],
)
num_repeats = 5
k = (1, 3, 5)
for dataset in datasets:
    dataset['infer_cfg']['inferencer']['max_out_len'] = 8192
    # openai pass@k config: the current setting is pass@5 (n=10).
    if not any(exclude in dataset['abbr'] for exclude in ('mbpp', 'humaneval')):
        dataset['eval_cfg']['evaluator']['num_repeats'] = num_repeats
    dataset['eval_cfg']['evaluator']['k'] = k
    dataset['num_repeats'] = num_repeats
    dataset['abbr'] += f'_passk'

work_dir = 'outputs/code_passk'
