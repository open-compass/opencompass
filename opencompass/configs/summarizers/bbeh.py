from mmengine.config import read_base

with read_base():
    from .groups.bbeh import bbeh_summary_groups

# Get all the BBEH subset names from the imported bbeh_summary_groups
bbeh_subsets = []
for group in bbeh_summary_groups:
    if group['name'] == 'bbeh':
        bbeh_subsets = group['subsets']
        break

summarizer = dict(
    # Include both individual datasets and the summary metrics we want to see
    dataset_abbrs=bbeh_subsets + ['bbeh'] + ['bbeh_harmonic_mean', 'bbeh_standard_deviation', 'bbeh_sum'],
    
    # Define the summary group for bbeh
    summary_groups=[
        {
            'name': 'bbeh',
            'subsets': bbeh_subsets,
            'metric': 'score'  # Explicitly specify the metric to use
        },
        {
            'name': 'bbeh_harmonic_mean',
            'subsets': bbeh_subsets,
            'metric': 'harmonic_mean'
        },
        {
            'name': 'bbeh_standard_deviation',
            'subsets': bbeh_subsets,
            'metric': 'standard_deviation'
        },
        {
            'name': 'bbeh_sum',
            'subsets': bbeh_subsets,
            'metric': 'sum'
        }
    ]
)