from opencompass.models import HuggingFaceCausalLM

# OPT-350M
opt350m = dict(
       type=HuggingFaceCausalLM,
       # the folowing are HuggingFaceCausalLM init parameters
       path='facebook/opt-350m',
       tokenizer_path='facebook/opt-350m',
       tokenizer_kwargs=dict(
           padding_side='left',
           truncation_side='left',
           proxies=None,
           trust_remote_code=True),
       model_kwargs=dict(device_map='auto'),
       max_seq_len=2048,
       # the folowing are not HuggingFaceCausalLM init parameters
       abbr='opt350m',                    # Model abbreviation
       max_out_len=100,                   # Maximum number of generated tokens
       batch_size=64,
       run_cfg=dict(num_gpus=1),    # Run configuration for specifying resource requirements
    )

models = [opt350m]
