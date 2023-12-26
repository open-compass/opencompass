from os import getenv as gv

from mmengine.config import read_base
with read_base():
    from .models.qwen.hf_qwen_7b_chat import models as hf_qwen_7b_chat
    from .models.qwen.hf_qwen_14b_chat import models as hf_qwen_14b_chat
    from .models.chatglm.hf_chatglm3_6b import models as hf_chatglm3_6b
    from .models.baichuan.hf_baichuan2_7b_chat import models as hf_baichuan2_7b
    from .models.hf_internlm.hf_internlm_chat_20b import models as hf_internlm_chat_20b
    from .models.judge_llm.auto_j.hf_autoj_eng_13b import models as hf_autoj
    from .models.judge_llm.judgelm.hf_judgelm_33b_v1 import models as hf_judgelm
    from .models.judge_llm.pandalm.hf_pandalm_7b_v1 import models as hf_pandalm
    from .datasets.subjective_alignbench.alignbench_judgeby_critiquellm import subjective_datasets

datasets = [*subjective_datasets]

from opencompass.models import HuggingFaceCausalLM, HuggingFace, OpenAIAllesAPIN, HuggingFaceChatGLM3
from opencompass.partitioners import NaivePartitioner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.runners import SlurmSequentialRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask
from opencompass.summarizers import AlignmentBenchSummarizer


# -------------Inferen Stage ----------------------------------------

models = [*hf_baichuan2_7b]#, *hf_chatglm3_6b, *hf_internlm_chat_20b, *hf_qwen_7b_chat, *hf_qwen_14b_chat]

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=SlurmSequentialRunner,
        partition='llmeval',
        quotatype='auto',
        max_num_workers=256,
        task=dict(type=OpenICLInferTask)),
)


# -------------Evalation Stage ----------------------------------------


## ------------- JudgeLLM Configuration
api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ]
)

judge_model = dict(
        abbr='GPT4-Turbo',
        type=OpenAIAllesAPIN, path='gpt-4-1106-preview',
        key='xxxx',  # The key will be obtained from $OPENAI_API_KEY, but you can write down your key here as well
        url='xxxx',
        meta_template=api_meta_template,
        query_per_second=16,
        max_out_len=2048,
        max_seq_len=2048,
        batch_size=8
)

## ------------- Evaluation Configuration
eval = dict(
    partitioner=dict(
        type=SubjectiveNaivePartitioner,
        mode='singlescore',
        models = [*hf_baichuan2_7b]
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=2,
        task=dict(
            type=SubjectiveEvalTask,
            judge_cfg=judge_model
        )),
)

summarizer = dict(
    type=AlignmentBenchSummarizer,
)

work_dir = 'outputs/alignment_bench/'
