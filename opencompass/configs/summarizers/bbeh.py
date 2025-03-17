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
    dataset_abbrs=bbeh_subsets + ['bbeh_naive_average'] + ['bbeh_harmonic_mean'],
    
    # Define the summary group for bbeh
    summary_groups=[
        {
            'name': 'bbeh_naive_average',
            'subsets': bbeh_subsets,
            'metric': 'naive_average'  # Explicitly specify the metric to use
        },
        {
            'name': 'bbeh_harmonic_mean',
            'subsets': bbeh_subsets,
            'metric': 'harmonic_mean'
        }
    ]
)