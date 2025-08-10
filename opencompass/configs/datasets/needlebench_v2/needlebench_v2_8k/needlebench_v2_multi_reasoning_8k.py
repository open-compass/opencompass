from opencompass.datasets.needlebench_v2.multi import NeedleBenchMultiDataset
from mmengine.config import read_base
with read_base():
    from .needlebench_v2_single_8k import depths_list, context_lengths
    from .needlebench_v2_single_8k import needlebench_reader_cfg, needlebench_infer_cfg
    from opencompass.configs.datasets.needlebench_v2.atc.atc_0shot_nocot_2_power_en import needlebench_atc_eval_cfg as needlebench_eval_cfg


# ----------English Version----------
base_path = 'opencompass/needlebench'
file_list = ['PaulGrahamEssays.jsonl']
needle_file_name = 'names.json'
diff = 10
language = 'English'
length_buffer = 500

# Initialize dataset lists
needlebench_2needle_en_datasets = []
needlebench_3needle_en_datasets = []
needlebench_4needle_en_datasets = []
needlebench_5needle_en_datasets = []

# Create datasets for different numbers of needles
for num_needles in range(2, 6):
    dataset_list_name = f'needlebench_{num_needles}needle_en_datasets'
    
    for original_context_length in context_lengths:
        for depth_percent in depths_list:
            dataset_dict = {
                'abbr': f'Length{original_context_length}'
                f'Depth{int(depth_percent)}_{num_needles}needle_en_8k',
                'type': NeedleBenchMultiDataset,
                'path': base_path,
                'length': original_context_length,
                'depth': int(depth_percent),
                'tokenizer_model': 'gpt-4',
                'file_list': file_list,
                'num_repeats_per_file': 10,
                'length_buffer': length_buffer,
                'language': language,
                'needle_file_name': needle_file_name,
                'num_needles': num_needles,
                'diff': diff,
                'reader_cfg': needlebench_reader_cfg,
                'infer_cfg': needlebench_infer_cfg,
                'eval_cfg': needlebench_eval_cfg,
            }
            
            # Add to the appropriate list using globals()
            globals()[f'needlebench_{num_needles}needle_en_datasets'].append(dataset_dict)

# ----------Chinese Version----------
base_path = 'opencompass/needlebench'
file_list = ['zh_finance.jsonl']
needle_file_name = 'names.json'
diff = 10
language = 'Chinese'
length_buffer = 200

# Initialize dataset lists
needlebench_2needle_zh_datasets = []
needlebench_3needle_zh_datasets = []
needlebench_4needle_zh_datasets = []
needlebench_5needle_zh_datasets = []

# Create datasets for different numbers of needles
for num_needles in range(2, 6):
    dataset_list_name = f'needlebench_{num_needles}needle_zh_datasets'
    
    for original_context_length in context_lengths:
        for depth_percent in depths_list:
            dataset_dict = {
                'abbr': f'Length{original_context_length}'
                f'Depth{int(depth_percent)}_{num_needles}needle_zh_8k',
                'type': NeedleBenchMultiDataset,
                'path': base_path,
                'length': original_context_length,
                'depth': int(depth_percent),
                'tokenizer_model': 'gpt-4',
                'file_list': file_list,
                'num_repeats_per_file': 10,
                'length_buffer': length_buffer,
                'language': language,
                'needle_file_name': needle_file_name,
                'num_needles': num_needles,
                'diff': diff,
                'reader_cfg': needlebench_reader_cfg,
                'infer_cfg': needlebench_infer_cfg,
                'eval_cfg': needlebench_eval_cfg,
            }
            
            # Add to the appropriate list using globals()
            globals()[f'needlebench_{num_needles}needle_zh_datasets'].append(dataset_dict)