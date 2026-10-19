categories = ['math', 'physics', 'chemistry', 'law', 'engineering', 'other', 'economics', 'health', 'psychology', 'business', 'biology', 'philosophy', 'computer science', 'history']

mmlu_pro_summary_groups = [
    {'name': 'mmlu_pro', 'subsets': ['mmlu_pro_' + c.replace(' ', '_') for c in categories]},
]
