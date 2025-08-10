korbench_summary_groups = []
categories = ['cipher', 'counterfactual', 'logic', 'operation', 'puzzle']
mixed_categories = ['Multi-Q', 'Multi-R', 'Multi-RQ']
korbench_summary_groups.append({'name': 'korbench_single', 'subsets': [f'korbench_{c}' for c in categories]})
korbench_summary_groups.append({'name': 'korbench_mixed', 'subsets': [f'korbench_{c}' for c in mixed_categories]})
