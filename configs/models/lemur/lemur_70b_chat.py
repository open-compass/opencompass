from opencompass.models import HuggingFaceCausalLM


_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='\n<|im_start|>user\n', end='<|im_end|>'),
        dict(role='BOT', begin='\n<|im_start|>assistant\n', end='<|im_end|>', generate=True),
    ],
)

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='lemur-70b-chat-v1',
        path='OpenLemur/lemur-70b-chat-v1',
        tokenizer_path='OpenLemur/lemur-70b-chat-v1',
        # tokenizer_kwargs=dict(
        #     padding_side='left',
        #     truncation_side='left',
        #     trust_remote_code=True,
        #     use_fast=False,),
        # pad_token_id=151643,
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        meta_template=_meta_template,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=4, num_procs=1),
    )
]
