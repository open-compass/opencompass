from opencompass.models import HuggingFaceCausalLM

models = [
    # CodeLlama 7B Python
    dict(
        type=HuggingFaceCausalLM,
        abbr='CodeLlama-7b-Python',
        path="codellama/CodeLlama-7b-Python-hf",
        tokenizer_path='codellama/CodeLlama-7b-Python-hf',
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
        ),
        max_out_len=1024,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(trust_remote_code=True, device_map='auto'),
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
]
