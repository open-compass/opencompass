from mmengine import read_base

with read_base():
    from ..korbench.korbench_cipher_zero_shot_gen import korbench_cipher_zero_shot_dataset as korbench_cipher_zero_shot_gen_dataset
    from ..korbench.korbench_counterfactual_zero_shot_gen import korbench_counterfactual_zero_shot_dataset as korbench_counterfactual_zero_shot_gen_dataset
    from ..korbench.korbench_logic_zero_shot_gen import korbench_logic_zero_shot_dataset as korbench_logic_zero_shot_gen_dataset
    from ..korbench.korbench_operation_zero_shot_gen import korbench_operation_zero_shot_dataset as korbench_operation_zero_shot_gen_dataset
    from ..korbench.korbench_puzzle_zero_shot_gen import korbench_puzzle_zero_shot_dataset as korbench_puzzle_zero_shot_gen_dataset

datasets = [korbench_cipher_zero_shot_gen_dataset, korbench_counterfactual_zero_shot_gen_dataset, korbench_logic_zero_shot_gen_dataset, korbench_operation_zero_shot_gen_dataset, korbench_puzzle_zero_shot_gen_dataset]

    
#korbench_datasets[0]['abbr'] = 'demo_' + korbench_datasets[0]['abbr']
#korbench_datasets[0]['reader_cfg']['test_range'] = '[0:8]'
