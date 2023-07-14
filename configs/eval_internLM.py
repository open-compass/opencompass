from opencompass.models.intern import internLM
from mmengine.config import read_base

with read_base():
    from .datasets.piqa.piqa_ppl import piqa_datasets
    from .datasets.nq.nq_gen import nq_datasets
datasets = []
# datasets += piqa_datasets
datasets += nq_datasets
#
# models = [
#     dict(
#         type=intern_model,
#         path="model/",
#         tokenizer_path='V7.model',
#         model_config = "model_config.py",
#         max_out_len=100,
#         max_seq_len=2048,
#         batch_size=16,
#         run_cfg=dict(num_gpus=1, num_procs=1))
# ]
#
#
models = [
    dict(
        type=internLM,
        # root_path = ""
        path="./internData/",
        tokenizer_path='./internData/V7.model',
        model_config = "./internData/model_config.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1))
]
work_dir = './outputs/2023_07_10_02/'
# infer = dict(
#     partitioner=dict(type='SizePartitioner', max_task_size=3000, gen_task_coef=10),
#     # partitioner=dict(type='NaivePartitioner'),
#     runner=dict(
#         type='SlurmRunner',
#         max_num_workers=32,
#         task=dict(type='OpenICLInferTask'),
#         retry=2),
# )
#
# eval = dict(
#     partitioner=dict(type='NaivePartitioner'),
#     runner=dict(
#         type='SlurmRunner',
#         max_num_workers=64,
#         task=dict(type='OpenICLEvalTask'),
#         retry=2),
# )