from opencompass.models.intern import intern_model
from mmengine.config import read_base

with read_base():
    from .datasets.piqa.piqa_ppl import piqa_datasets
    from .datasets.nq.nq_gen import nq_datasets
datasets = []
datasets += piqa_datasets
# datasets += nq_datasets

models = [
    dict(
        type=intern_model,
        path="/mnt/petrelfs/chenkeyu1/program/model/final_maibao",
        tokenizer_path='/mnt/petrelfs/share_data/llm_data/tokenizers/V7.model',
        tokenizer_type='v7',
        model_config = "/mnt/petrelfs/chenkeyu1/program/openAGIEval/InternLM/configs/model_config.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1))
]
work_dir = './outputs/2023_07_10/'
infer = dict(
    partitioner=dict(type='SizePartitioner', max_task_size=3000, gen_task_coef=10),
    # partitioner=dict(type='NaivePartitioner'),
    runner=dict(
        type='SlurmRunner',
        max_num_workers=32,
        task=dict(type='OpenICLInferTask'),
        retry=2),
)

eval = dict(
    partitioner=dict(type='NaivePartitioner'),
    runner=dict(
        type='SlurmRunner',
        max_num_workers=64,
        task=dict(type='OpenICLEvalTask'),
        retry=2),
)




