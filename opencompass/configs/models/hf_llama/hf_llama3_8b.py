from opencompass.models import HuggingFaceBaseModel

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr='llama-3-8b-hf',
        path='meta-llama/Meta-Llama-3-8B',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    )
]


# Llama-3.1-8B-Instruct for MMLU benchmark (added by 6taco)
llama3_1_8b_instruct_mmlu = dict(
    type=HubModelForCausalLM,
    abbr='llama-3.1-8b-instruct-mmlu',
    path='meta-llama/Meta-Llama-3.1-8B-Instruct',
    tokenizer_path='meta-llama/Meta-Llama-3.1-8B-Instruct',
    tokenizer_kwargs=dict(
        padding_side='left',
        truncation_side='left',
        trust_remote_code=True,
        use_fast=False,
    ),
    max_out_len=2048,
    max_seq_len=8192,
    batch_size=8,
    generation_kwargs=dict(
        do_sample=False,
        temperature=0.0,
    ),
    run_cfg=dict(num_gpus=1, num_procs=1),
)
