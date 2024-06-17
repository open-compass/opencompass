charm_tasks = [
    'Anachronisms_Judgment',
    'Movie_and_Music_Recommendation',
    'Natural_Language_Inference',
    'Reading_Comprehension',
    'Sequence_Understanding',
    'Sport_Understanding',
    'Time_Understanding',
]
regions = [
    'Chinese',
    'Global',
]
prompts = [
    'Direct',
    'ZH-CoT',
    'EN-CoT',
    'XLT',
    'Translate-EN',
]


charm_reason_summary_groups = []
for prompt in prompts:
    for region in regions:
        subsets = ['charm-reason-' + region + '_' + task + '_' + prompt for task in charm_tasks]
        charm_reason_summary_groups.append({'name': 'charm-reason-' + region + '_' + prompt, 'subsets': subsets})

for prompt in prompts:
    subsets = ['charm-reason-' + region + '_' + prompt for region in regions]
    charm_reason_summary_groups.append({'name': 'charm-reason-' + prompt, 'subsets': subsets})

charm_reason_summary_groups.append(
    {'name': 'charm-reason-CoT', 'subsets': ['charm-reason-ZH-CoT', 'charm-reason-EN-CoT']}
)
