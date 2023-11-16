mathbench_summary_groups = []


mathbench_college = ['single_choice_cn', 'cloze_en']
mathbench_college = ['mathbench-college' + s for s in mathbench_college]

mathbench_high = ['single_choice_cn', 'single_choice_en']
mathbench_high = ['mathbench-high' + s for s in mathbench_high]

mathbench_middle = ['single_choice_cn']
mathbench_middle = ['mathbench-middle' + s for s in mathbench_middle]

mathbench_primary = ['cloze_cn']
mathbench_primary = ['mathbench-primary' + s for s in mathbench_primary]

mathbench_summary_groups.append(
    {'name': 'mathbench',
     'subsets': mathbench_college+mathbench_high+mathbench_middle+mathbench_primary}
)
