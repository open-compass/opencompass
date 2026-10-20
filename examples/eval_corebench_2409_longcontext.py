import os.path as osp
from copy import deepcopy

from mmengine.config import read_base

from opencompass.models import (HuggingFacewithChatTemplate,
                                TurboMindModelwithChatTemplate)
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import DLCRunner, LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

#######################################################################
#                          PART 0  Essential Configs                  #
#######################################################################
with read_base():
    from opencompass.configs.datasets.longbench.longbench import \
        longbench_datasets
    from opencompass.configs.datasets.needlebench.needlebench_8k.needlebench_8k import \
        needlebench_datasets as needlebench_8k_datasets
    from opencompass.configs.datasets.needlebench.needlebench_32k.needlebench_32k import \
        needlebench_datasets as needlebench_32k_datasets
    from opencompass.configs.datasets.needlebench.needlebench_128k.needlebench_128k import \
        needlebench_datasets as needlebench_128k_datasets
    from opencompass.configs.datasets.ruler.ruler_8k_gen import \
        ruler_datasets as ruler_8k_datasets
    from opencompass.configs.datasets.ruler.ruler_32k_gen import \
        ruler_datasets as ruler_32k_datasets
    from opencompass.configs.datasets.ruler.ruler_128k_gen import \
        ruler_datasets as ruler_128k_datasets
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_5_7b_chat_1m import \
        models as lmdeploy_internlm2_5_7b_1m_chat_model
    from opencompass.configs.models.hf_llama.lmdeploy_llama3_1_8b_instruct import \
        models as llama3_1_8b_instruct_model
    # Instruct models
    from opencompass.configs.models.qwen.lmdeploy_qwen2_7b_instruct import \
        models as lmdeploy_qwen2_7b_instruct_model
    # Summary Groups
    from opencompass.configs.summarizers.groups.longbench import \
        longbench_summary_groups
    from opencompass.configs.summarizers.groups.ruler import \
        ruler_summary_groups
    from opencompass.configs.summarizers.needlebench import (
        needlebench_8k_summarizer, needlebench_32k_summarizer,
        needlebench_128k_summarizer)

#######################################################################
#                          PART 1  Datasets List                      #
#######################################################################
# datasets list for evaluation
datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

#######################################################################
#                       PART 2  Datset Summarizer                     #
#######################################################################
needlebench_8k_summary_groups = needlebench_8k_summarizer['summary_groups']
needlebench_32k_summary_groups = needlebench_32k_summarizer['summary_groups']
needlebench_128k_summary_groups = needlebench_128k_summarizer['summary_groups']

# Instruct models summarizer
summarizer = dict(
    dataset_abbrs=[
        ['ruler_8k', 'naive_average'],
        ['ruler_32k', 'naive_average'],
        ['ruler_128k', 'naive_average'],
        ['NeedleBench-Overall-Score-8K', 'weighted_average'],
        ['NeedleBench-Overall-Score-32K', 'weighted_average'],
        ['NeedleBench-Overall-Score-128K', 'weighted_average'],
        ['longbench', 'naive_average'],
        ['longbench_zh', 'naive_average'],
        ['longbench_en', 'naive_average'],
        '',
        'longbench_single-document-qa',
        'longbench_multi-document-qa',
        'longbench_summarization',
        'longbench_few-shot-learning',
        'longbench_synthetic-tasks',
        'longbench_code-completion',
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)

#######################################################################
#                        PART 3  Models  List                         #
#######################################################################

lmdeploy_qwen2_7b_instruct_model[0]['max_seq_len'] = 1048576
lmdeploy_qwen2_7b_instruct_model[0]['engine_config']['session_len'] = 1048576
lmdeploy_qwen2_7b_instruct_model[0]['engine_config']['tp'] = 4
lmdeploy_qwen2_7b_instruct_model[0]['engine_config']['rope_scaling_factor'] = 4
lmdeploy_qwen2_7b_instruct_model[0]['run_cfg']['num_gpus'] = 4

llama3_1_8b_instruct_model[0]['max_seq_len'] = 1048576
llama3_1_8b_instruct_model[0]['engine_config']['session_len'] = 1048576
llama3_1_8b_instruct_model[0]['engine_config']['tp'] = 4
llama3_1_8b_instruct_model[0]['engine_config']['rope_scaling_factor'] = 4
llama3_1_8b_instruct_model[0]['run_cfg']['num_gpus'] = 4

models = sum([v for k, v in locals().items() if k.endswith('_model')], [])

#######################################################################
#                 PART 4  Inference/Evaluation Configuaration         #
#######################################################################

# Local Runner
infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=8),
    runner=dict(
        type=LocalRunner,
        max_num_workers=16,
        retry=0,  # Modify if needed
        task=dict(type=OpenICLInferTask)),
)

# eval with local runner
eval = dict(
    partitioner=dict(type=NaivePartitioner, n=10),
    runner=dict(type=LocalRunner,
                max_num_workers=16,
                task=dict(type=OpenICLEvalTask)),
)

#######################################################################
#                      PART 5  Utils Configuaration                   #
#######################################################################
base_exp_dir = 'outputs/corebench/'
work_dir = osp.join(base_exp_dir, 'long_context')
