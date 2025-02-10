from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import xIFEvalDataset, xIFEvaluator

xifeval_reader_cfg = dict(
    input_columns=['prompt'], output_column='reference')

xifeval_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt='{prompt}'),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=2048))

xifeval_eval_cfg = dict(
    evaluator=dict(type=xIFEvaluator),
    pred_role='BOT',
)
LANG = 'de'

xifeval_datasets = [
    dict(
        abbr='xIFEval',
        type=xIFEvalDataset,
        path=f'data/ifeval/input_data_google_{LANG}.jsonl',
        reader_cfg=xifeval_reader_cfg,
        infer_cfg=xifeval_infer_cfg,
        eval_cfg=xifeval_eval_cfg)
]
