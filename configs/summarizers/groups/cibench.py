
_cibench = ['Pandas', 'Matplotlib', 'Opencv', 'SciPy', 'Seaborn', 'PyTorch']
_cibench = ['cibench_generation_' + i for i in _cibench]
cibench_summary_groups = [{'name': 'cibench_generation', 'subsets': _cibench}]
