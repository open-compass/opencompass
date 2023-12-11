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
        ['mathbench-college-single_choice_cn-agent', 'acc_1'],
        ['mathbench-college-cloze_en-agent', 'accuracy'],
        ['mathbench-high-single_choice_cn-agent', 'acc_1'],
        ['mathbench-high-single_choice_en-agent', 'acc_1'],
        ['mathbench-middle-single_choice_cn-agent', 'acc_1'],
        ['mathbench-primary-cloze_cn-agent', 'accuracy'],
        '######## MathBench-Agent CircularEval ########', # category
        ['mathbench-college-single_choice_cn-agent', 'perf_4'],
        ['mathbench-high-single_choice_cn-agent', 'perf_4'],
        ['mathbench-high-single_choice_en-agent', 'perf_4'],
        ['mathbench-middle-single_choice_cn-agent', 'perf_4'],
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith("_summary_groups")], [])
)
