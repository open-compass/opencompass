bbh_summary_groups = []

# bbh
_bbh = ['temporal_sequences', 'disambiguation_qa', 'date_understanding', 'tracking_shuffled_objects_three_objects', 'penguins_in_a_table','geometric_shapes', 'snarks', 'ruin_names', 'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_five_objects','logical_deduction_three_objects', 'hyperbaton', 'logical_deduction_five_objects', 'logical_deduction_seven_objects', 'movie_recommendation','salient_translation_error_detection', 'reasoning_about_colored_objects', 'multistep_arithmetic_two', 'navigate', 'dyck_languages', 'word_sorting', 'sports_understanding','boolean_expressions', 'object_counting', 'formal_fallacies', 'causal_judgement', 'web_of_lies']
_bbh = ['bbh-' + s for s in _bbh]
bbh_summary_groups.append({'name': 'bbh', 'subsets': _bbh})
