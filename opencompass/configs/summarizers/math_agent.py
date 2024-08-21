from mmengine.config import read_base

with read_base():
    from .groups.mathbench_agent import mathbench_agent_summary_groups

summarizer = dict(
    dataset_abbrs=[
        '######## GSM8K-Agent Accuracy ########', # category
        ['gsm8k-agent', 'follow_acc'],
        ['gsm8k-agent', 'reasoning_acc'],
        ['gsm8k-agent', 'code_acc'],
        ['gsm8k-agent', 'action_pct'],
        '######## MATH-Agent Accuracy ########', # category
        ['math-agent', 'follow_acc'],
        ['math-agent', 'reasoning_acc'],
        ['math-agent', 'code_acc'],
        ['math-agent', 'action_pct'],
        '######## MathBench-Agent Accuracy ########', # category
        'mathbench-agent',
        'mathbench-circular-agent',
        'mathbench-circular-and-cloze-agent',
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], [])
)
