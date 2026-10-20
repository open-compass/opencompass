bbeh_summary_groups = []

# bbeh
_bbeh = [
    'bbeh_boolean_expressions', 'bbeh_disambiguation_qa', 'bbeh_geometric_shapes', 'bbeh_hyperbaton',
    'bbeh_movie_recommendation', 'bbeh_nycc', 'bbeh_shuffled_objects', 'bbeh_boardgame_qa',
    'bbeh_buggy_tables', 'bbeh_causal_understanding', 'bbeh_dyck_languages', 'bbeh_linguini',
    'bbeh_multistep_arithmetic', 'bbeh_object_counting', 'bbeh_object_properties', 'bbeh_sarc_triples',
    'bbeh_spatial_reasoning', 'bbeh_sportqa', 'bbeh_temporal_sequence', 'bbeh_time_arithmetic',
    'bbeh_web_of_lies', 'bbeh_word_sorting', 'bbeh_zebra_puzzles'
]
bbeh_summary_groups.append({'name': 'bbeh', 'subsets': _bbeh, 'metric':'naive_average'})
bbeh_summary_groups.append({'name': 'bbeh', 'subsets': _bbeh, 'metric':'harmonic_mean'})