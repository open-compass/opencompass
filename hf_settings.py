import os
import huggingface_hub.constants as hf_constants
from huggingface_hub import set_cache_dir
from datasets import get_dataset_config_names  # Optional, if you need dataset-related functionality

# Set a new cache directory
new_cache_dir = "/fs-computility/llm/shared/llmeval/models/opencompass_hf_hub"  # Replace with your desired path
set_cache_dir(new_cache_dir)

# Alternatively, you can set the environment variable
# os.environ["HF_HOME"] = new_cache_dir

# Root cache path for Hugging Face
root_cache_dir = hf_constants.HF_HOME
print(f"Root HF cache path: {root_cache_dir}")

# Dataset cache path (typically under HF_HOME/datasets)
dataset_cache_dir = f"{root_cache_dir}/datasets"
print(f"Dataset cache path: {dataset_cache_dir}")

# Model cache path (typically under HF_HOME/hub)
model_cache_dir = f"{root_cache_dir}/hub"
print(f"Model cache path: {model_cache_dir}")