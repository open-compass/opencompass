from opencompass.partitioners import SizePartitioner, NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import SlurmRunner, LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from mmengine.config import read_base
from opencompass.models import VLLMwithChatTemplate
from opencompass.models import TurboMindModelwithChatTemplate

with read_base():
    from .datasets.RULER.ruler_4k.ruler_4k import ruler_datasets as ruler_4k_datasets
    from .datasets.RULER.ruler_8k.ruler_8k import ruler_datasets as ruler_8k_datasets
    from .datasets.RULER.ruler_16k.ruler_16k import ruler_datasets as ruler_16k_datasets
    from .datasets.RULER.ruler_32k.ruler_32k import ruler_datasets as ruler_32k_datasets
    from .datasets.RULER.ruler_128k.ruler_128k import ruler_datasets as ruler_128k_datasets


models = [
    # qwen2
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='qwen2-7b-instruct-turbomind',
        path='Qwen/Qwen2-7B-Instruct',
        engine_config=dict(session_len=33792, max_batch_size=16, tp=1),
        gen_config=dict(top_k=1, temperature=1e-6, top_p=0.9, max_new_tokens=1024),
        max_seq_len=33792,
        max_out_len=1024,
        batch_size=16,
        run_cfg=dict(num_gpus=2)
        ),
    # llama3
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='llama-3-8b-instruct-turbomind',
        path='meta-llama/Meta-Llama-3-8B-Instruct',
        engine_config=dict(max_batch_size=16, tp=1),
        gen_config=dict(top_k=1, temperature=1e-6, top_p=0.9, max_new_tokens=1024),
        max_seq_len=33792,
        max_out_len=1024,
        batch_size=16,
        run_cfg=dict(num_gpus=2),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
    ),
    # internlm
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='internlm2_5-7b-chat-1m-turbomind',
        path='internlm/internlm2_5-7b-chat-1m',
        engine_config=dict(rope_scaling_factor=2.5, session_len=33792, max_batch_size=16, tp=4), # 1M context length
        gen_config=dict(top_k=1, temperature=1e-6, top_p=0.9, max_new_tokens=2048),
        max_seq_len=33792,
        max_out_len=2048,
        batch_size=1,
        run_cfg=dict(num_gpus=4),
    )
]



datasets = [
    *ruler_4k_datasets,
    *ruler_8k_datasets,
    *ruler_16k_datasets, 
    # *ruler_32k_datasets,
    # *ruler_128k_datasets,
    ]        
models = [*models]

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=16,
        task=dict(type=OpenICLInferTask),
        retry=5),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=32,
        task=dict(type=OpenICLEvalTask)),
)