from opencompass.partitioners import (
    NaivePartitioner,
    NumWorkerPartitioner,
)
from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    from ..configs.models.qwen.lmdeploy_qwen2_7b_instruct import (
        models as qwen2_7b_instruct_model,
    )
    from ..configs.models.hf_llama.lmdeploy_llama3_8b_instruct import (
        models as llama3_8b_instruct_model,
    )
    from ..configs.models.hf_internlm.lmdeploy_internlm2_5_7b_chat_1m import (
        models as internlm2_5_7b_chat_1m,
    )
    from .datasets.ruler.ruler_niah import niah_datasets  # Niah
    from .datasets.ruler.ruler_vt import vt_datasets  # VT
    from .datasets.ruler.ruler_fwe import fwe_datasets  # FWE
    from .datasets.ruler.ruler_cwe import cwe_datasets  # CWE
    from .datasets.ruler.ruler_qa import qa_datasets  # QA

import_datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

# Evaluation config
NUM_SAMPLES = 500
max_seq_lens = [1024 * 4, 1024 * 8, 1024 * 16, 1024 * 32]
abbr_suffixs = ['4k', '8k', '16k', '32k']
work_dir = './outputs/ruler'

# Model Settings
qwen2_7b_instruct_model[0]['max_seq_len'] = 33792
qwen2_7b_instruct_model[0]['engine_config']['tp'] = 2
qwen2_7b_instruct_model[0]['run_cfg']['num_gpus'] = 2
llama3_8b_instruct_model[0]['max_seq_len'] = 33792
llama3_8b_instruct_model[0]['engine_config']['tp'] = 2
llama3_8b_instruct_model[0]['run_cfg']['num_gpus'] = 2
model_settings = [
    [qwen2_7b_instruct_model[0], 'Qwen/Qwen2-7B-Instruct'],
    [llama3_8b_instruct_model[0], 'meta-llama/Meta-Llama-3-8B-Instruct'],
    [internlm2_5_7b_chat_1m[0], 'internlm/internlm2_5-7b-chat-1m'],
]


# Dataset Model Combination
datasets = []
models = []
model_dataset_combinations = []

# Different seq length
for max_seq_len, abbr_suffix in zip(max_seq_lens, abbr_suffixs):
    for model, model_path in model_settings:
        _tmp_datasets = []
        for dataset in import_datasets:
            tmp_dataset = dataset.deepcopy()
            tmp_dataset['tokenizer_model'] = model_path
            tmp_dataset['abbr'] = tmp_dataset['abbr'] + '_' + abbr_suffix
            tmp_dataset['num_samples'] = NUM_SAMPLES
            tmp_dataset['max_seq_length'] = max_seq_len
            _tmp_datasets.append(tmp_dataset)
        model_dataset_combinations.append(dict(models=[model], datasets=_tmp_datasets))
        models.append(model)
        datasets.extend(_tmp_datasets)


infer = dict(
    partitioner=dict(type=NumWorkerPartitioner),
    runner=dict(
        type=LocalRunner, max_num_workers=16, task=dict(type=OpenICLInferTask), retry=5
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner, max_num_workers=32, task=dict(type=OpenICLEvalTask)),
)


summary_groups = []
for abbr_suffix in abbr_suffixs:
    summary_groups.append(
        {
            'name': abbr_suffix,
            'subsets': [dataset['abbr'] + abbr_suffix for dataset in import_datasets],
        }
    )
summarizer = dict(dataset_abbrs=abbr_suffixs, summary_groups=summary_groups)

# summarizer = dict(
#     dataset_abbrs = [
#         '###### MathBench-A: Application Part ######',
#         'college',
#         'high',
#         'middle',
#         'primary',
#         'arithmetic',
#         'mathbench-a (average)',

#         '###### MathBench-T: Theory Part ######',
#         'college_knowledge',
#         'high_knowledge',
#         'middle_knowledge',
#         'primary_knowledge',
#         'mathbench-t (average)',

#         '###### Overall: Average between MathBench-A and MathBench-T ######',
#         'Overall',
#     ],
#     summary_groups=mathbench_2024_summary_groups,
# )
