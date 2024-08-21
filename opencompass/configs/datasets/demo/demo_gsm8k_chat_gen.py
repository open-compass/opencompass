from mmengine import read_base

with read_base():
    from ..gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets

gsm8k_datasets[0]['abbr'] = 'demo_' + gsm8k_datasets[0]['abbr']
gsm8k_datasets[0]['reader_cfg']['test_range'] = '[0:64]'
