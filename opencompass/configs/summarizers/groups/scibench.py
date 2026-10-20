scibench_summary_groups = []

scibench_tasks = ['atkins', 'calculus', 'chemmc', 'class', 'diff', 'fund', 'matter', 'quan', 'stat', 'thermo']
for suffix in ['', '_zs-cot', '_fs', '_fs-cot']:
    subsets = [f'scibench-{subset}{suffix}' for subset in scibench_tasks]
    scibench_summary_groups.append({'name': f'scibench{suffix}', 'subsets': subsets})
