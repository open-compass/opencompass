from mmengine.config import read_base
from opencompass.models import PytorchModel


with read_base():
    # choose a list of datasets
    # from .datasets.mmlu.mmlu_gen_a484b3 import mmlu_datasets
    from .datasets.ceval.ceval_gen_5f30c7 import ceval_datasets
    # from .datasets.SuperGLUE_WiC.SuperGLUE_WiC_gen_d06864 import WiC_datasets
    # from .datasets.SuperGLUE_WSC.SuperGLUE_WSC_gen_6dc406 import WSC_datasets
    # from .datasets.triviaqa.triviaqa_gen_2121ce import triviaqa_datasets
    # from .datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    # from .datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets
    # from .datasets.race.race_gen_69ee4f import race_datasets
    # from .datasets.crowspairs.crowspairs_gen_381af0 import crowspairs_datasets
    # and output the results in a choosen format
    from .summarizers.medium import summarizer

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])


meta_template = dict(
    round=[
        dict(role="HUMAN", begin='[INST] ', end=' [/INST] '),
        dict(role="BOT", begin="", end='', generate=True),
    ],
)

# config for internlm-chat-7b
# models = [
#     dict(
#         type=TurboMindModel,
#         abbr='internlm-chat-7b-turbomind',
#         path="./turbomind",
#         max_out_len=100,
#         max_seq_len=2048,
#         batch_size=32,
#         concurrency=32,
#         meta_template=meta_template,
#         run_cfg=dict(num_gpus=1, num_procs=1),
#     )
# ]

# config for internlm-chat-7b-w4 model
# models = [
#     dict(
#         type=TurboMindModel,
#         abbr='internlm-chat-7b-w4-turbomind',
#         path="./turbomind",
#         max_out_len=100,
#         max_seq_len=2048,
#         batch_size=32,
#         concurrency=32,
#         meta_template=meta_template,
#         run_cfg=dict(num_gpus=1, num_procs=1),
#     )
# ]

# config for internlm-chat-7b-w4kv8 model
# models = [
#     dict(
#         type=TurboMindModel,
#         abbr='internlm-chat-7b-w4kv8-turbomind',
#         path="./turbomind",
#         max_out_len=100,
#         max_seq_len=2048,
#         batch_size=32,
#         concurrency=32,
#         meta_template=meta_template,
#         run_cfg=dict(num_gpus=1, num_procs=1),
#     )
# ]

# config for internlm-chat-20b
# models = [
#     dict(
#         type=TurboMindModel,
#         abbr='internlm-chat-20b-turbomind',
#         path="./turbomind",
#         max_out_len=100,
#         max_seq_len=2048,
#         batch_size=8,
#         concurrency=8,
#         meta_template=meta_template,
#         run_cfg=dict(num_gpus=1, num_procs=1),
#     )
# ]

# config for internlm-chat-20b-w4 model
models = [
    dict(
        type=PytorchModel,
        abbr='llama2-chat-7b-pytorch-poc',
        path="/mnt/142/gaojianfei/quantization/smooth_llama_chat_absmax",
        # path='/mnt/140/InternLM/20B/internlm-20b-chat',
        max_out_len=100,
        max_seq_len=2048,
        batch_size=1,
        concurrency=1,
        meta_template=meta_template,
        run_cfg=dict(num_gpus=1, num_procs=1),
        # stop_words=[103027, 103028],
        # w8a8=True
    )
]

# config for internlm-chat-20b-w4kv8 model
# models = [
#     dict(
#         type=TurboMindModel,
#         abbr='internlm-chat-20b-w4kv8-turbomind',
#         path="./turbomind",
#         max_out_len=100,
#         max_seq_len=2048,
#         batch_size=16,
#         concurrency=16,
#         meta_template=meta_template,
#         run_cfg=dict(num_gpus=1, num_procs=1),
#     )
# ]
