from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import EMEvaluator
from opencompass.datasets import ReCoRDDataset, ReCoRD_postprocess

ReCoRD_reader_cfg = dict(
    input_columns=['question', 'text'], output_column='answers')

ReCoRD_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=
        'Passage:{text}\nResult:{question}\nQuestion: What entity does ____ refer to in the result?Give me the entity name:'),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

ReCoRD_eval_cfg = dict(
    evaluator=dict(type=EMEvaluator), pred_postprocessor=dict(type=ReCoRD_postprocess))

ReCoRD_datasets = [
    dict(
        type=ReCoRDDataset,
        abbr='ReCoRD',
        path='./data/SuperGLUE/ReCoRD/val.jsonl',
        reader_cfg=ReCoRD_reader_cfg,
        infer_cfg=ReCoRD_infer_cfg,
        eval_cfg=ReCoRD_eval_cfg)
]
