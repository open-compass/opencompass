from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import CluewscDatasetV2
from opencompass.utils.text_postprocessors import first_capital_postprocess

cluewsc_reader_cfg = dict(
    input_columns=['span1', 'span2', 'text', 'new_text'],
    output_column='label',
)

cluewsc_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt=
                '{text}\n此处，“{span2}”是否指代“{span1}“？\nA. 是\nB. 否\n请从”A“，”B“中进行选择。\n答：',
            ),
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

cluewsc_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_capital_postprocess),
)

cluewsc_datasets = [
    dict(
        abbr='cluewsc-dev',
        type=CluewscDatasetV2,
        path='./data/FewCLUE/cluewsc/dev_few_all.json',
        reader_cfg=cluewsc_reader_cfg,
        infer_cfg=cluewsc_infer_cfg,
        eval_cfg=cluewsc_eval_cfg,
    ),
    dict(
        abbr='cluewsc-test',
        type=CluewscDatasetV2,
        path='./data/FewCLUE/cluewsc/test_public.json',
        reader_cfg=cluewsc_reader_cfg,
        infer_cfg=cluewsc_infer_cfg,
        eval_cfg=cluewsc_eval_cfg,
    ),
]
