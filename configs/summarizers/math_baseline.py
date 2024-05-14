from mmengine.config import read_base

with read_base():
    from .groups.mathbench import mathbench_summary_groups

summarizer = dict(
    dataset_abbrs=[
        '######## GSM8K Accuracy ########', # category
        ['gsm8k', 'accuracy'],
        '######## MATH Accuracy ########', # category
        ['math', 'accuracy'],
        '######## MathBench-Agent Accuracy ########', # category
        'mathbench',
        'mathbench-circular',
        'mathbench-circular-and-cloze',
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], [])
)
