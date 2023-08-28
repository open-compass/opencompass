from opencompass.models import HuggingFaceCausalLM

# Please note that we have specified the revision here. Recently (on 20230827),
# during our evaluations, we found that the newer revision models have a drop
# of more than 5 points on datasets like GaokaoBench/mbpp.
# We are not yet sure whether this drop is due to incorrect logic in OpenCompass
# calling qwen or some other reasons. We would like to highlight this.

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='qwen-7b-hf',
        path="Qwen/Qwen-7B",
        tokenizer_path='Qwen/Qwen-7B',
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
            use_fast=False,
            revision='39fc5fdcb95c8c367bbdb3bfc0db71d96266de09'
        ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True,
            revision='39fc5fdcb95c8c367bbdb3bfc0db71d96266de09'
        ),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
