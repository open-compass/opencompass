from opencompass.models import Llama2Chat

# Please follow the instruction in the Meta AI website https://github.com/facebookresearch/llama
# and download the LLaMA-2-Chat model and tokenizer to the path './models/llama2/llama/'.
#
# The LLaMA requirement is also needed to be installed.
#
# git clone https://github.com/facebookresearch/llama.git
# cd llama
# pip install -e .

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
)

models = [
    dict(
        abbr='llama-2-7b-chat',
        type=Llama2Chat,
        path='./models/llama2/llama/llama-2-7b-chat/',
        tokenizer_path='./models/llama2/llama/tokenizer.model',
        meta_template=api_meta_template,
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
]
