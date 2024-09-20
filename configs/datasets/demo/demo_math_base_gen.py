from mmengine import read_base

with read_base():
    from ..math.math_4shot_base_gen_db136b import math_datasets

math_datasets[0]['abbr'] = 'demo_' + math_datasets[0]['abbr']
math_datasets[0]['reader_cfg']['test_range'] = '[0:64]'
