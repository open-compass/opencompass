from opencompass.models import HuggingFaceCausalLM
from opencompass.models.turbomind import TurboMindModel
from opencompass.runners import SlurmSequentialRunner
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

from mmengine.config import read_base
with read_base():
    # eval needlebench_4k
    from .datasets.needlebench.needlebench_4k.needlebench import needlebench_datasets
    from .summarizers.needlebench import needlebench_4k_summarizer as summarizer

    # only eval original "needle in a haystack test" in needlebench_4k
    # from .datasets.needlebench.needlebench_4k.needlebench_single import needlebench_datasets_zh, needlebench_datasets_en
    # from .summarizers.needlebench import needlebench_4k_summarizer as summarizer

    # eval Ancestral Tracing Challenge(ATC)
    # from .datasets.needlebench.atc.atc import needlebench_atc_datasets_zh, needlebench_atc_datasets_en
    # from .summarizers.needlebench import needlebench_atc_summarizer as summarizer

datasets = sum([v for k, v in locals().items() if ('datasets' in k)], [])

hf_internlm2_chat_7b_model_meta_template = dict(
    round=[
        dict(role='HUMAN',
             begin='<|im_start|>user\n', end='<|im_end|>\n'),
        dict(role='BOT', begin='<|im_start|>assistant\n',
             end='<|im_end|>\n', generate=True),
    ],
)
hf_internlm2_chat_7b = dict(
        type=HuggingFaceCausalLM,
        abbr='internlm2-chat-7b-hf',
        path="internlm/internlm2-chat-7b",
        tokenizer_path='internlm/internlm2-chat-7b',
        model_kwargs=dict(
            trust_remote_code=True,
            device_map='auto',
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            use_fast=False,
            trust_remote_code=True,
        ),
        max_out_len=2000,
        max_seq_len=32768,
        batch_size=8,
        meta_template=hf_internlm2_chat_7b_model_meta_template,
        run_cfg=dict(num_gpus=1, num_procs=1),
        end_str='<|im_end|>',
        )

internlm2_chat_7b_200k = dict(
        type=TurboMindModel,
        abbr='internlm2-chat-7b-200k',
        path="internlm/internlm2-chat-7b",
        meta_template=hf_internlm2_chat_7b_model_meta_template,
        engine_config=dict(session_len=210000,
                           max_batch_size=8,
                           rope_scaling_factor=2.0,
                           model_name="internlm2-chat-7b"),
        gen_config=dict(top_k=1, top_p=0.8,
                        temperature=1.0,
                        max_new_tokens=2000),
        max_out_len=2000,
        max_seq_len=210000,
        batch_size=8,
        concurrency=8,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )

models = [
    # hf_internlm2_chat_7b,
    internlm2_chat_7b_200k,
]

work_dir = './outputs/needlebench'
