from mmengine.config import read_base
from opencompass.models import LmdeployPytorchModel


with read_base():
    # choose a list of datasets
    from .datasets.mmlu.mmlu_gen_a484b3 import mmlu_datasets
    from .datasets.ceval.ceval_gen_5f30c7 import ceval_datasets
    from .datasets.SuperGLUE_WiC.SuperGLUE_WiC_gen_d06864 import WiC_datasets
    from .datasets.SuperGLUE_WSC.SuperGLUE_WSC_gen_7902a7 import WSC_datasets
    from .datasets.triviaqa.triviaqa_gen_2121ce import triviaqa_datasets
    from .datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    from .datasets.race.race_gen_69ee4f import race_datasets
    from .datasets.crowspairs.crowspairs_gen_381af0 import crowspairs_datasets
    # and output the results in a choosen format
    from .summarizers.medium import summarizer


datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])


meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<|User|>:', end='<eoh>\n'),
        dict(role='BOT', begin='<|Bot|>:', end='<eoa>\n', generate=True),
    ],
    eos_token_id=103028)

# config for internlm-chat-7b
internlm_chat_7b = dict(
    type=LmdeployPytorchModel,
    abbr='internlm-chat-7b-pytorch',
    path='internlm/internlm-chat-7b',
    engine_config=dict(session_len=2048,
                       max_batch_size=16),
    gen_config=dict(top_k=1,
                    top_p=0.8,
                    temperature=1.0,
                    max_new_tokens=100),
    max_out_len=100,
    max_seq_len=2048,
    batch_size=16,
    concurrency=16,
    meta_template=meta_template,
    run_cfg=dict(num_gpus=1, num_procs=1),
    end_str='<eoa>',
)

# config for internlm-chat-20b
internlm_chat_20b = dict(
    type=LmdeployPytorchModel,
    abbr='internlm-chat-20b-pytorch',
    path='internlm/internlm-chat-20b',
    engine_config=dict(session_len=2048,
                       max_batch_size=8),
    gen_config=dict(top_k=1,
                    top_p=0.8,
                    temperature=1.0,
                    max_new_tokens=100),
    max_out_len=100,
    max_seq_len=2048,
    batch_size=8,
    concurrency=8,
    meta_template=meta_template,
    run_cfg=dict(num_gpus=1, num_procs=1),
    end_str='<eoa>',
    )

models = [internlm_chat_20b]
