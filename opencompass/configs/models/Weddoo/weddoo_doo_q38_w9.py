from opencompass.models import HuggingFacewithChatTemplate

# Weddoo-Doo-Q3.8.W9 is a LoRA-tuned Qwen3-8B for the Chinese wedding
# industry. The weights are published on ModelScope rather than the HF Hub:
#   https://modelscope.cn/models/Weddoo/Weddoo-Doo-Q3.8.W9
# Download the repository and point `path` at the local directory to run this
# config offline.
models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='Weddoo-Doo-Q3.8.W9',
        path='Weddoo/Weddoo-Doo-Q3.8.W9',
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True,
            torch_dtype='auto',
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
            use_fast=False,
        ),
        max_seq_len=8192,
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
