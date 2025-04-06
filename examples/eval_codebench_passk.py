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
    # dataset['abbr'] += f'_passk'

# summary
summarizer = dict(
    dataset_abbrs = [
        'pass@1',
        ['bigcodebench_full_instruct_passk', 'pass@1'],
        ['bigcodebench_hard_instruct_passk', 'pass@1'],
        ['lcb_code_generation_passk', 'pass@1'],
        ['openai_humaneval_passk_passk', 'humaneval_pass@1'],
        ['humaneval_pro_passk', 'pass@1'],
        ['mbpp_passk_passk', 'pass@1'],
        ['mbpp_pro_passk', 'pass@1'],
        ['humaneval-multiple-cpp_passk', 'pass@1'],
        ['humaneval-multiple-cs_passk', 'pass@1'],
        ['humaneval-multiple-go_passk', 'pass@1'],
        ['humaneval-multiple-java_passk', 'pass@1'],
        ['humaneval-multiple-rb_passk', 'pass@1'],
        ['humaneval-multiple-js_passk', 'pass@1'],
        ['humaneval-multiple-php_passk', 'pass@1'],
        ['humaneval-multiple-r_passk', 'pass@1'],
        ['humaneval-multiple-rs_passk', 'pass@1'],
        ['humaneval-multiple-sh_passk', 'pass@1'],
        ['mbpp-multiple-cpp_passk', 'pass@1'],
        ['mbpp-multiple-cs_passk', 'pass@1'],
        ['mbpp-multiple-go_passk', 'pass@1'],
        ['mbpp-multiple-java_passk', 'pass@1'],
        ['mbpp-multiple-rb_passk', 'pass@1'],
        ['mbpp-multiple-js_passk', 'pass@1'],
        ['mbpp-multiple-php_passk', 'pass@1'],
        ['mbpp-multiple-r_passk', 'pass@1'],
        ['mbpp-multiple-rs_passk', 'pass@1'],
        ['mbpp-multiple-sh_passk', 'pass@1'],
        '',
        'pass@3',
        ['bigcodebench_full_instruct_passk', 'pass@3'],
        ['bigcodebench_hard_instruct_passk', 'pass@3'],
        ['lcb_code_generation_passk', 'pass@3'],
        ['openai_humaneval_passk_passk', 'humaneval_pass@3'],
        ['humaneval_pro_passk', 'pass@3'],
        ['mbpp_passk_passk', 'pass@3'],
        ['mbpp_pro_passk', 'pass@3'],
        ['humaneval-multiple-cpp_passk', 'pass@3'],
        ['humaneval-multiple-cs_passk', 'pass@3'],
        ['humaneval-multiple-go_passk', 'pass@3'],
        ['humaneval-multiple-java_passk', 'pass@3'],
        ['humaneval-multiple-rb_passk', 'pass@3'],
        ['humaneval-multiple-js_passk', 'pass@3'],
        ['humaneval-multiple-php_passk', 'pass@3'],
        ['humaneval-multiple-r_passk', 'pass@3'],
        ['humaneval-multiple-rs_passk', 'pass@3'],
        ['humaneval-multiple-sh_passk', 'pass@3'],
        ['mbpp-multiple-cpp_passk', 'pass@3'],
        ['mbpp-multiple-cs_passk', 'pass@3'],
        ['mbpp-multiple-go_passk', 'pass@3'],
        ['mbpp-multiple-java_passk', 'pass@3'],
        ['mbpp-multiple-rb_passk', 'pass@3'],
        ['mbpp-multiple-js_passk', 'pass@3'],
        ['mbpp-multiple-php_passk', 'pass@3'],
        ['mbpp-multiple-r_passk', 'pass@3'],
        ['mbpp-multiple-rs_passk', 'pass@3'],
        ['mbpp-multiple-sh_passk', 'pass@3'],
        '',
        'pass@5',
        ['bigcodebench_full_instruct_passk', 'pass@5'],
        ['bigcodebench_hard_instruct_passk', 'pass@5'],
        ['lcb_code_generation_passk', 'pass@5'],
        ['openai_humaneval_passk_passk', 'humaneval_pass@5'],
        ['humaneval_pro_passk', 'pass@5'],
        ['mbpp_passk_passk', 'pass@5'],
        ['mbpp_pro_passk', 'pass@5'],
        ['humaneval-multiple-cpp_passk', 'pass@5'],
        ['humaneval-multiple-cs_passk', 'pass@5'],
        ['humaneval-multiple-go_passk', 'pass@5'],
        ['humaneval-multiple-java_passk', 'pass@5'],
        ['humaneval-multiple-rb_passk', 'pass@5'],
        ['humaneval-multiple-js_passk', 'pass@5'],
        ['humaneval-multiple-php_passk', 'pass@5'],
        ['humaneval-multiple-r_passk', 'pass@5'],
        ['humaneval-multiple-rs_passk', 'pass@5'],
        ['humaneval-multiple-sh_passk', 'pass@5'],
        ['mbpp-multiple-cpp_passk', 'pass@5'],
        ['mbpp-multiple-cs_passk', 'pass@5'],
        ['mbpp-multiple-go_passk', 'pass@5'],
        ['mbpp-multiple-java_passk', 'pass@5'],
        ['mbpp-multiple-rb_passk', 'pass@5'],
        ['mbpp-multiple-js_passk', 'pass@5'],
        ['mbpp-multiple-php_passk', 'pass@5'],
        ['mbpp-multiple-r_passk', 'pass@5'],
        ['mbpp-multiple-rs_passk', 'pass@5'],
        ['mbpp-multiple-sh_passk', 'pass@5'],
    ],
)

work_dir = 'outputs/code_passk'
