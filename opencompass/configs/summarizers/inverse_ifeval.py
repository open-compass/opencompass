from mmengine.config import read_base

with read_base():
    from .groups.inverse_ifeval import (inverse_ifeval_instruction_type_abbrs,
                                        inverse_ifeval_subsets,
                                        inverse_ifeval_summary_groups)

summarizer = dict(
    dataset_abbrs=[
        ['InverseIFEval', 'weighted_average'],
        ['InverseIFEval_zh', 'weighted_average'],
        ['InverseIFEval_en', 'weighted_average'],
        ['InverseIFEval_macro', 'naive_average'],
        *[[
            f'InverseIFEval_{instruction_type}', 'naive_average'
        ] for instruction_type in inverse_ifeval_instruction_type_abbrs],
        *[[subset, 'accuracy'] for subset in inverse_ifeval_subsets],
    ],
    summary_groups=inverse_ifeval_summary_groups,
)
