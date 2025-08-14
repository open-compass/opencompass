from mmengine.config import read_base

with read_base():
    # Import dataset configuration
    from .datasets.zebra_grid.zebra_grid_gen import zebra_grid_datasets
    # Import model configuration  
    from .models.qwen.vllm_qwen3_8b_thinking import models

# Datasets to evaluate
datasets = zebra_grid_datasets

# Models to use
models = models

# Evaluation configuration
eval = dict(
    partitioner=dict(type='NaivePartitioner'),
    runner=dict(
        type='LocalRunner',
        max_num_workers=1,  # Only 1 worker for tensor_parallel_size=8 model
        task=dict(type='OpenICLEvalTask'),
    ),
)

# Work directory for results
work_dir = './outputs/zebra_grid_qwen3_8b_thinking/' 