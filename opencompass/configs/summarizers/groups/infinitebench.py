infinitebench_summary_groups = []

_infinitebench = ['codedebug', 'coderun', 'endia', 'enmc', 'enqa', 'ensum', 'mathcalc', 'mathfind', 'retrievekv', 'retrievenumber', 'retrievepasskey', 'zhqa']
_infinitebench = ['InfiniteBench_' + s for s in _infinitebench]
infinitebench_summary_groups.append({'name': 'InfiniteBench', 'subsets': _infinitebench})
