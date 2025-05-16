# This config is used to test all the code benchmarks
from mmengine.config import read_base
import os.path as osp
from opencompass.runners import LocalRunner, VOLCRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    # Datasets Part
    # bigcodebench
    from opencompass.configs.datasets.bigcodebench.bigcodebench_full_instruct_gen import (
        bigcodebench_full_instruct_datasets
    )
    from opencompass.configs.datasets.bigcodebench.bigcodebench_hard_instruct_gen import (
        bigcodebench_hard_instruct_datasets
    )
    # livecodebench code generation lite v5
    from opencompass.configs.datasets.livecodebench.livecodebench_time_split_gen_a4f90b import (
        LCB_datasets
    )
    # huamneval series
    from opencompass.configs.datasets.humaneval.humaneval_openai_sample_evals_gen_dcae0e import (
        humaneval_datasets
    )
    from opencompass.configs.datasets.humaneval_pro.humaneval_pro_gen import (
        humanevalpro_datasets
    )
    from opencompass.configs.datasets.humanevalx.humanevalx_gen_620cfa import (
        humanevalx_datasets
    )
    from opencompass.configs.datasets.humaneval_plus.humaneval_plus_gen import (
        humaneval_plus_datasets
    )
    # mbpp series
    from opencompass.configs.datasets.mbpp.mbpp_gen import (
        mbpp_datasets
    )
    from opencompass.configs.datasets.mbpp_pro.mbpp_pro_gen import (
        mbpppro_datasets
    )
    # multipl-e
    from opencompass.configs.datasets.multipl_e.multiple_gen import (
        multiple_datasets
    )
    # ds1000
    from opencompass.configs.datasets.ds1000.ds1000_service_eval_gen_cbc84f import (
        ds1000_datasets
    )

    # Models Part
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_7b_instruct import (
        models as lmdeploy_qwen2_5_7b_instruct_model,
    )

    # Summary Groups
    from opencompass.configs.summarizers.groups.ds1000 import (
        ds1000_summary_groups,
    )
    from opencompass.configs.summarizers.groups.multipl_e import (
        multiple_summary_groups,
    )
    from opencompass.configs.summarizers.groups.humanevalx import (
        humanevalx_summary_groups,
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

for item in humanevalx_datasets:
    item['eval_cfg']['evaluator'][
        'ip_address'
    ] = 'codeeval.opencompass.org.cn/humanevalx'
    item['eval_cfg']['evaluator']['port'] = ''
for item in ds1000_datasets:
    item['eval_cfg']['evaluator'][
        'ip_address'
    ] = 'codeeval.opencompass.org.cn/ds1000'
    item['eval_cfg']['evaluator']['port'] = ''


for dataset in datasets:
    dataset['infer_cfg']['inferencer']['max_out_len'] = 8192


# summary
summary_groups = sum(
    [v for k, v in locals().items() if k.endswith('_summary_groups')], []
)
summary_groups.append(
    {'name': 'humanevalx', 
    'subsets': ['humanevalx-python', 'humanevalx-cpp', 'humanevalx-java', 'humanevalx-js']}
)
summarizer = dict(
    dataset_abbrs = [
        ['bigcodebench_hard_instruct', 'pass@1'],
        ['bigcodebench_full_instruct', 'pass@1'],
        ['lcb_code_generation', 'pass@1'],
        ['openai_humaneval', 'humaneval_pass@1'],
        ['mbpp', 'score'],
        ['humaneval_pro', 'pass@1'],
        ['mbpp_pro', 'pass@1'],
        ['humaneval_plus', 'humaneval_plus_pass@1'],
        ['multiple', 'naive_average'],
        ['humanevalx', 'naive_average'],
        ['ds1000', 'naive_average'],
        '',
        'humanevalx-python',
        'humanevalx-cpp',
        'humanevalx-java',
        'humanevalx-js',
        '',
        'ds1000_Pandas',
        'ds1000_Numpy',
        'ds1000_Tensorflow',
        'ds1000_Scipy',
        'ds1000_Sklearn',
        'ds1000_Pytorch',
        'ds1000_Matplotlib',
        '',
        'humaneval-multiple-cpp', 
        'humaneval-multiple-cs', 
        'humaneval-multiple-go', 
        'humaneval-multiple-java', 
        'humaneval-multiple-rb', 
        'humaneval-multiple-js', 
        'humaneval-multiple-php', 
        'humaneval-multiple-r', 
        'humaneval-multiple-rs', 
        'humaneval-multiple-sh',
        '',
        'mbpp-multiple-cpp', 
        'mbpp-multiple-cs', 
        'mbpp-multiple-go', 
        'mbpp-multiple-java', 
        'mbpp-multiple-rb', 
        'mbpp-multiple-js', 
        'mbpp-multiple-php', 
        'mbpp-multiple-r', 
        'mbpp-multiple-rs', 
        'mbpp-multiple-sh'
    ],
    summary_groups=summary_groups,
)

work_dir = 'outputs/code'
