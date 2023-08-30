from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import HFDataset

z_bench_reader_cfg = dict(
    input_columns=['text'], output_column='category', train_split='test')

z_bench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='{text}',
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

z_bench_datasets = dict(
    type=HFDataset,
    path=
    '/mnt/petrelfs/gaotong/llm_eval/openagieval_dataset/eval_datasets/z_bench',
    data_dir=
    '/mnt/petrelfs/gaotong/llm_eval/openagieval_dataset/eval_datasets/z_bench',
    name='question',
    reader_cfg=z_bench_reader_cfg,
    infer_cfg=z_bench_infer_cfg)
