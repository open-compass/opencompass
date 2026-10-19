categories = ['Math', 'Physics', 'Chemistry', 'Law', 'Engineering', 'Other', 'Economics', 'Health', 'Psychology', 'Business', 'Biology', 'Philosophy', 'Computer_Science', 'History']

mmlu_cf_summary_groups = [
    {'name': 'mmlu_cf', 'subsets': ['mmlu_cf_' + c.replace(' ', '_') for c in categories]},
]
