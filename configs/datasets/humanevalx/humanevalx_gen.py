from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import HFDataset, HumanEvaluator, humaneval_postprocess

humanevalx_reader_cfg = dict(
    input_columns=['prompt'], output_column='task_id', train_split='test')

# TODO: allow empty output-column
humanevalx_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='{prompt}'),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512))

humanevalx_eval_cfg = None

humanevalx_datasets = [
    dict(
        type=HFDataset,
        name=lang,
        path='THUDM/humaneval-x',
        reader_cfg=humanevalx_reader_cfg,
        infer_cfg=humanevalx_infer_cfg,
        eval_cfg=humanevalx_eval_cfg)
    for lang in ['python', 'cpp', 'go', 'java', 'js']
]