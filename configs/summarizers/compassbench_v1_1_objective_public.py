
from mmengine.config import read_base

with read_base():
    from .compassbench_v1_1_objective import summarizer

for dataset_abbr in summarizer['dataset_abbrs']:
    if isinstance(dataset_abbr, str):
        continue
    else:
        dataset_abbr[0] = dataset_abbr[0] + '_public'
for summary_group in summarizer['summary_groups']:
    summary_group['name'] = summary_group['name'] + '_public'
    replaced_subset = []
    for subset in summary_group['subsets']:
        if isinstance(subset, str):
            replaced_subset.append(subset + '_public')
        else:
            replaced_subset.append([subset[0] + '_public', subset[1]])
    summary_group['subsets'] = replaced_subset
    if 'weights' in summary_group:
        summary_group['weights'] = {k + '_public': v for k, v in summary_group['weights'].items()}
