# Only torch >=2.0 is supported for falcon-40b
from opencompass.models import HuggingFaceCausalLM

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='falcon-40b-hf',
        path='tiiuae/falcon-40b',
        tokenizer_path='tiiuae/falcon-40b',
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
        ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(trust_remote_code=True, device_map='auto', revision='561820f7eef0cc56a31ea38af15ca1acb07fab5d'),
        run_cfg=dict(num_gpus=4, num_procs=1),
    )
]
