
code_passk_summary_groups = [
    # rename
    {'name': 'humaneval_pass@1(greedy)', 'subsets': [['openai_humaneval', 'humaneval_pass@1']]},
    {'name': 'humaneval_pass@10', 'subsets': [['openai_humaneval_passk', 'humaneval_pass@10']]},
    {'name': 'humaneval_pass@10', 'subsets': [['openai_humaneval_repeat10', 'humaneval_pass@10']]},
    {'name': 'humaneval_cn_pass@1(greedy)', 'subsets': [['openai_humaneval_cn', 'humaneval_pass@1']]},
    {'name': 'humaneval_cn_pass@10', 'subsets': [['openai_humaneval_cn_passk', 'humaneval_pass@10']]},
    {'name': 'humaneval_cn_pass@10', 'subsets': [['openai_humaneval_cn_repeat10', 'humaneval_pass@10']]},
    {'name': 'humaneval_plus_pass@1(greedy)', 'subsets': [['humaneval_plus', 'humaneval_plus_pass@1']]},
    {'name': 'humaneval_plus_pass@10', 'subsets': [['humaneval_plus_passk', 'humaneval_plus_pass@10']]},
    {'name': 'humaneval_plus_pass@10', 'subsets': [['humaneval_plus_repeat10', 'humaneval_plus_pass@10']]},
    {'name': 'mbpp_pass@1(greedy)', 'subsets': [['mbpp', 'score']]},
    {'name': 'mbpp_pass@10', 'subsets': [['mbpp_passk', 'pass@10']]},
    {'name': 'mbpp_pass@10', 'subsets': [['mbpp_repeat10', 'pass@10']]},
    {'name': 'mbpp_cn_pass@1(greedy)', 'subsets': [['mbpp_cn', 'score']]},
    {'name': 'mbpp_cn_pass@10', 'subsets': [['mbpp_cn_passk', 'pass@10']]},
    {'name': 'mbpp_cn_pass@10', 'subsets': [['mbpp_cn_repeat10', 'pass@10']]},
    {'name': 'sanitized_mbpp_pass@1(greedy)', 'subsets': [['sanitized_mbpp', 'score']]},
    {'name': 'sanitized_mbpp_pass@10', 'subsets': [['sanitized_mbpp_passk', 'pass@10']]},
    {'name': 'sanitized_mbpp_pass@10', 'subsets': [['sanitized_mbpp_repeat10', 'pass@10']]},
    # real add
    {'name': 'humanevalx', 'subsets': ['humanevalx-python', 'humanevalx-cpp', 'humanevalx-go', 'humanevalx-java', 'humanevalx-js']},
    # {'name': 'code', 'subsets': ['humaneval_plus_pass@1(greedy)', 'sanitized_mbpp_pass@1(greedy)', 'humaneval_cn_pass@1(greedy)', 'mbpp_cn_pass@1(greedy)', 'humanevalx']}
    {'name': 'code_cn', 'subsets': ['humaneval_cn_pass@1(greedy)', 'mbpp_cn_pass@1(greedy)']},
    {'name': 'code_en', 'subsets': ['humaneval_plus_pass@1(greedy)', 'sanitized_mbpp_pass@1(greedy)', 'humanevalx']},
    {'name': 'code', 'subsets': ['humaneval_cn_pass@1(greedy)', 'mbpp_cn_pass@1(greedy)', 'humaneval_plus_pass@1(greedy)', 'sanitized_mbpp_pass@1(greedy)', 'humanevalx']},
]

summarizer = dict(
    dataset_abbrs=[
        'code',
        'code_cn',
        'code_en',
        'humaneval_cn_pass@1(greedy)',
        'humaneval_plus_pass@1(greedy)',
        'mbpp_cn_pass@1(greedy)',
        'sanitized_mbpp_pass@1(greedy)',
        'humanevalx',
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], [])
)
