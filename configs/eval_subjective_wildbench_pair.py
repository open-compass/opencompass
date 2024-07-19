from mmengine.config import read_base

with read_base():
    # from .datasets.subjective.multiround.mtbench_single_judge_diff_temp import subjective_datasets
    from .datasets.subjective.wildbench.wildbench_pair_judge import subjective_datasets
    from .models.openai.gpt_4 import models as gpt4_models
    from .models.hf_llama.hf_llama2_70b_chat import models as llama2_models
    # from .models.gemma.hf_gemma_2b_it import models
    # from .models.hf_llama.hf_llama3_70b_instruct import models as llama3_model
    # # from .models.hf_internlm.hf_internlm2_chat_7b import models
    # from .models.yi.hf_yi_1_5_34b_chat import models as yi_model
    # from .models.qwen.hf_qwen1_5_72b_chat import models as qwen_model

from opencompass.models import HuggingFaceCausalLM, HuggingFace, HuggingFaceChatGLM3, OpenAI
from opencompass.partitioners import NaivePartitioner, SizePartitioner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.partitioners.sub_size import SubjectiveSizePartitioner
from opencompass.runners import LocalRunner
from opencompass.runners import SlurmSequentialRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask
from opencompass.summarizers import WildBenchPairSummarizer
from opencompass.models.claude_api.claude_api import Claude
from opencompass.models import HuggingFacewithChatTemplate


models = sum([v for k, v in locals().items() if k.endswith('_model')], [])

api_meta_template = dict(
    round=[
        dict(role='SYSTEM', api_role='SYSTEM'),
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ]
)

# _meta_template = dict(
#     round=[
#         dict(role='HUMAN', begin='\n<|im_start|>user\n', end='<|im_end|>'),
#         dict(role='BOT', begin='\n<|im_start|>assistant\n', end='<|im_end|>', generate=True),
#     ],
# )
# -------------Inference Stage ----------------------------------------
# For subjective evaluation, we often set do sample for models

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='llama-3-8b-instruct-hf',
        path='meta-llama/Meta-Llama-3-8B-Instruct',
        max_out_len=4096,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
    ),
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='yi-1.5-6b-chat-hf',
        path='01-ai/Yi-1.5-6B-Chat',
        max_out_len=4096,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    ),
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='qwen1.5-7b-chat-hf',
        path='Qwen/Qwen1.5-7B-Chat',
        max_out_len=4096,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    ),
    # dict(
    #     type=HuggingFacewithChatTemplate,
    #     abbr='llama-3-70b-instruct-hf',
    #     path='meta-llama/Meta-Llama-3-70B-Instruct',
    #     max_out_len=4096,
    #     batch_size=8,
    #     run_cfg=dict(num_gpus=4),
    #     stop_words=['<|end_of_text|>', '<|eot_id|>'],
    # ),
    #     dict(
    #     type=HuggingFacewithChatTemplate,
    #     abbr='yi-1.5-34b-chat-hf',
    #     path='01-ai/Yi-1.5-34B-Chat',
    #     max_out_len=4096,
    #     batch_size=8,
    #     run_cfg=dict(num_gpus=2),
    # ),
    # dict(
    #     type=HuggingFacewithChatTemplate,
    #     abbr='qwen1.5-72b-chat-hf',
    #     path='Qwen/Qwen1.5-72B-Chat',
    #     max_out_len=4096,
    #     batch_size=8,
    #     run_cfg=dict(num_gpus=8),
    # )
]

datasets = [*subjective_datasets]

# -------------Evalation Stage ----------------------------------------

## ------------- JudgeLLM Configuration
judge_models = [dict(
    abbr='GPT4-Turbo',
    type=OpenAI,
    path='gpt-4-0613', # To compare with the official leaderboard, please use gpt4-0613
    key='xxxx',  # The key will be obtained from $OPENAI_API_KEY, but you can write down your key here as well
    meta_template=api_meta_template,
    query_per_second=16,
    max_out_len=2048,
    max_seq_len=2048,
    batch_size=8,
    temperature=0,
)]

gpt4 = dict(
    abbr='gpt4-turbo',
    type=OpenAI,
    path='gpt-4-0409-preview',
    key='',  # The key will be obtained from $OPENAI_API_KEY, but you can write down your key here as well
    meta_template=api_meta_template,
    query_per_second=1,
    max_out_len=2048,
    max_seq_len=4096,
    batch_size=4,
    retry=20,
    temperature=1,
)  # Re-inference gpt4's predictions or you can choose to use the pre-commited gpt4's predictions

claude = dict(abbr='HaiKu',
        type=Claude,
        path='claude-2',
        key='YOUR_CLAUDE_KEY',
        query_per_second=1,
        max_out_len=2048, max_seq_len=2048, batch_size=2,
    )
## single evaluation
# eval = dict(
#     partitioner=dict(type=SubjectiveSizePartitioner, strategy='split', max_task_size=10000, mode='singlescore', models=models, judge_models=judge_models),
#     runner=dict(type=LocalRunner, max_num_workers=32, task=dict(type=SubjectiveEvalTask)),
# )
infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=1000, strategy='split'),
    runner=dict(
        type=SlurmSequentialRunner,
        max_num_workers=64,
        quotatype='reserved',
        partition='llmeval',
        task=dict(type=OpenICLInferTask)),
)

eval = dict(
    partitioner=dict(
        type=SubjectiveNaivePartitioner,
        mode='m2n', # m个模型 与 n个模型进行对战
        infer_order='random',
        #  在m2n模式下，需要指定base_models和compare_models，将会对base_models和compare_models生成对应的两两pair（去重且不会与自身进行比较）
        base_models = [*llama2_models, gpt4, claude], # 用于对比的基线模型
        compare_models = models, # 待评测模型
        judge_models=judge_models
    ),
    runner=dict(
        type=LocalRunner,
        # partition='llmeval',
        # quotatype='auto',
        max_num_workers=3,
        task=dict(
            type=SubjectiveEvalTask
        )),
    given_pred = [{'abbr':'gpt4-turbo', 'path':'./data/WildBench/gpt4'},
                  {'abbr': 'llama-2-70b-chat-hf', 'path':'./data/WildBench/llama2-70b'},
                  {'abbr': 'HaiKu', 'path':'./data/WildBench/claude'},
                  {'abbr': 'llama-2-70b-chat-turbomind', 'path':'./data/WildBench/llama2-70b'},
                  {'abbr': 'llama-2-70b-chat-vllm', 'path':'./data/WildBench/llama2-70b'}]
)

summarizer = dict(type=WildBenchPairSummarizer)

work_dir = 'outputs/wildbench/'
