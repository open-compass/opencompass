
subset_list = [
    'test',
    'valid',
]

language_list = [
    'CN',
    'EN',
]

datasets = []

for subset in subset_list:
    for language in language_list:
        datasets.append(
            dict(
                abbr=f'CS-Bench_{language}_{subset}',
                path=f'./data/csbench/CSBench-{language}/{subset}.jsonl',
                evaluator=dict(
                    type='llm_evaluator',
                    judge_cfg=dict(),
                ),
            )
        )