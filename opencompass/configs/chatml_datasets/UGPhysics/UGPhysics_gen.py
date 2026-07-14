
subset_list = [
    'AtomicPhysics',
    'ClassicalElectromagnetism',
    'ClassicalMechanics',
    'Electrodynamics',
    'GeometricalOptics',
    'QuantumMechanics',
    'Relativity',
    'Solid-StatePhysics',
    'StatisticalMechanics',
    'SemiconductorPhysics',
    'Thermodynamics',
    'TheoreticalMechanics',
    'WaveOptics',
]

language_list = [
    'zh',
    'en',
]

datasets = []

for subset in subset_list:
    for language in language_list:
        datasets.append(
            dict(
                abbr=f'UGPhysics_{subset}_{language}',
                path=f'./data/ugphysics/{subset}/{language}.jsonl',
                evaluator=dict(
                    type='llm_evaluator',
                    judge_cfg=dict(),
                ),
            )
        )
