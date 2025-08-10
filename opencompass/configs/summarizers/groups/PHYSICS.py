physics_summary_groups = []

# bbh
_physcis = [
    'atomic_dataset_textonly',
    'electro_dataset_textonly',
    'mechanics_dataset_textonly',
    'optics_dataset_textonly',
    'quantum_dataset_textonly',
    'statistics_dataset_textonly',
]

_physcis = ['PHYSICS_' + s for s in _physcis]
physics_summary_groups.append({'name': 'PHYSICS', 'subsets': _physcis})