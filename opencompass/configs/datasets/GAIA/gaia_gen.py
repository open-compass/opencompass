from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import GAIADataset
from opencompass.utils.text_postprocessors import first_capital_postprocess

gaia_reader_cfg = dict(
    input_columns='question',
    output_column='answerKey',
    test_split='test')

gaia_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt=
                '请根据问题：{question}\n给出答案。答：'
            ),
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)
gaia_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_capital_postprocess),
)

gaia_datasets = [
    dict(
        abbr='gaia-validation',
        type=GAIADataset,
        path='opencompass/gaia',
        local_mode=False,
        reader_cfg=gaia_reader_cfg,
        infer_cfg=gaia_infer_cfg,
        eval_cfg=gaia_eval_cfg,
    )
]
