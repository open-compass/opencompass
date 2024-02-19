from mmengine.config import read_base
from opencompass.datasets.circular import (CircularCEvalDataset, CircularMMLUDataset, CircularCMMLUDataset, CircularCSQADataset,
                                           CircularARCDataset, CircularHSWAGDataset, CircularOBQADataset, CircularRaceDataset, CircularEvaluator)
from opencompass.summarizers import CircularSummarizer

with read_base():
    from .datasets.ceval.ceval_gen_5f30c7 import ceval_datasets
    from .datasets.mmlu.mmlu_gen_a484b3 import mmlu_datasets
    from .datasets.cmmlu.cmmlu_gen_c13365 import cmmlu_datasets
    from .datasets.hellaswag.hellaswag_gen_6faab5 import hellaswag_datasets
    from .datasets.ARC_e.ARC_e_gen_1e0de5 import ARC_e_datasets
    from .datasets.ARC_c.ARC_c_gen_1e0de5 import ARC_c_datasets
    from .datasets.commonsenseqa.commonsenseqa_gen_1da2d0 import commonsenseqa_datasets
    from .datasets.obqa.obqa_gen_9069e4 import obqa_datasets
    from .datasets.race.race_gen_69ee4f import race_datasets

    from .models.hf_internlm.hf_internlm_chat_7b import models as hf_internlm_chat_7b_model
    from .models.hf_internlm.hf_internlm_chat_20b import models as hf_internlm_chat_20b_model
    from .models.qwen.hf_qwen_7b_chat import models as hf_qwen_7b_chat_model
    from .models.qwen.hf_qwen_14b_chat import models as hf_qwen_14b_chat_model

    from .summarizers.groups.mmlu import mmlu_summary_groups
    from .summarizers.groups.cmmlu import cmmlu_summary_groups
    from .summarizers.groups.ceval import ceval_summary_groups

for ds, t in [
    (ceval_datasets, CircularCEvalDataset),
    (mmlu_datasets, CircularMMLUDataset),
    (cmmlu_datasets, CircularCMMLUDataset),
    (hellaswag_datasets, CircularHSWAGDataset),
    (ARC_e_datasets, CircularARCDataset),
    (ARC_c_datasets, CircularARCDataset),
    (commonsenseqa_datasets, CircularCSQADataset),
    (obqa_datasets, CircularOBQADataset),
    (race_datasets, CircularRaceDataset),
]:
    for d in ds:
        d['type'] = t
        d['abbr'] = d['abbr'] + '-circular-4'
        d['eval_cfg']['evaluator'] = {'type': CircularEvaluator, 'circular_pattern': 'circular'}
        d['circular_patterns'] = 'circular'


datasets = sum([v for k, v in locals().items() if k.endswith('_datasets') or k == 'datasets'], [])
models = sum([v for k, v in locals().items() if k.endswith('_model')], [])

# config summarizer
other_summary_groups = [
    {'name': 'average',
     'subsets': ['ceval', 'mmlu', 'cmmlu', 'hellaswag', 'ARC-e', 'ARC-c', 'commonsense_qa', 'openbookqa_fact', 'race-middle', 'race-high']},
]
origin_summary_groups = sum([v for k, v in locals().items() if k.endswith('_summary_groups')], [])
new_summary_groups = []
for item in origin_summary_groups:
    new_summary_groups.append(
        {
            'name': item['name'] + '-circular-4',
            'subsets': [i + '-circular-4' for i in item['subsets']],
        }
    )
summarizer = dict(
    type=CircularSummarizer,
    metric_types=['acc_origin', 'perf_circular'],
    dataset_abbrs = [
        'average-circular-4',
        'ceval-circular-4',
        'mmlu-circular-4',
        'cmmlu-circular-4',
        'hellaswag-circular-4',
        'ARC-e-circular-4',
        'ARC-c-circular-4',
        'commonsense_qa-circular-4',
        'openbookqa_fact-circular-4',
        'race-middle-circular-4',
        'race-high-circular-4',
        'ceval-humanities-circular-4',
        'ceval-stem-circular-4',
        'ceval-social-science-circular-4',
        'ceval-other-circular-4',
        'mmlu-humanities-circular-4',
        'mmlu-stem-circular-4',
        'mmlu-social-science-circular-4',
        'mmlu-other-circular-4',
        'cmmlu-humanities-circular-4',
        'cmmlu-stem-circular-4',
        'cmmlu-social-science-circular-4',
        'cmmlu-other-circular-4',
        'cmmlu-china-specific-circular-4',
    ],
    summary_groups=new_summary_groups,
)
