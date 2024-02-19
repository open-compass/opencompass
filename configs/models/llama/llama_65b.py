from opencompass.models import Llama2

# Please follow the instruction in the Meta AI website https://github.com/facebookresearch/llama/tree/llama_v1
# and download the LLaMA model and tokenizer to the path './models/llama/'.
#
# The LLaMA requirement is also needed to be installed.
# *Note* that the LLaMA-2 branch is fully compatible with LLAMA-1, and the LLaMA-2 branch is used here.
#
# git clone https://github.com/facebookresearch/llama.git
# cd llama
# pip install -e .

models = [
    dict(
        abbr='llama-65b',
        type=Llama2,
        path='./models/llama/65B/',
        tokenizer_path='./models/llama/tokenizer.model',
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=8, num_procs=8),
    ),
]
