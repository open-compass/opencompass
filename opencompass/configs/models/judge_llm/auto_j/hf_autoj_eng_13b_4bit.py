from opencompass.models import HuggingFaceCausalLM

'''
#This is a 4bits quantized version of Auto-J by using AutoGPTQ,
which is available on huggingface-hub:
https://huggingface.co/GAIR/autoj-13b-GPTQ-4bits
'''

models = [dict(
        type=HuggingFaceCausalLM,
        abbr='autoj-13b-GPTQ-4bits',
        path='GAIR/autoj-13b-GPTQ-4bits',
        tokenizer_path='GAIR/autoj-13b-GPTQ-4bits',
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
