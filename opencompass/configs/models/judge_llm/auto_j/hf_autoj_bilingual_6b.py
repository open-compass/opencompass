from opencompass.models import HuggingFaceCausalLM

'''
This is a bilingual 6B version of Auto-J.
It is trained on both the original training data
and its Chinese translation, which can be find in
https://huggingface.co/GAIR/autoj-bilingual-6b
'''

models = [dict(
        type=HuggingFaceCausalLM,
        abbr='autoj-bilingual-6b',
        path='GAIR/autoj-bilingual-6b',
        tokenizer_path='GAIR/autoj-bilingual-6b',
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              trust_remote_code=True,
                              use_fast=False,),
        max_out_len=1024,
        max_seq_len=4096,
        batch_size=8,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )]
