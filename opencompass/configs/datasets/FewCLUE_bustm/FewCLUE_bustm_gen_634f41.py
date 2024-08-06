from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import AFQMCDatasetV2
from opencompass.utils.text_postprocessors import first_capital_postprocess

bustm_reader_cfg = dict(
    input_columns=['sentence1', 'sentence2'],
    output_column='label',
    test_split='train')

bustm_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt=
                '语句一：“{sentence1}”\n语句二：“{sentence2}”\n请判断语句一和语句二说的是否是一个意思？\nA. 无关\nB. 相关\n请从“A”，“B”中进行选择。\n答：',
            ),
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

bustm_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_capital_postprocess),
)

bustm_datasets = [
    dict(
        abbr='bustm-dev',
        type=AFQMCDatasetV2,  # bustm share the same format with AFQMC
        path='./data/FewCLUE/bustm/dev_few_all.json',
        local_mode=True,
        reader_cfg=bustm_reader_cfg,
        infer_cfg=bustm_infer_cfg,
        eval_cfg=bustm_eval_cfg,
    ),
    dict(
        abbr='bustm-test',
        type=AFQMCDatasetV2,  # bustm share the same format with AFQMC
        path='./data/FewCLUE/bustm/test_public.json',
        local_mode=True,
        reader_cfg=bustm_reader_cfg,
        infer_cfg=bustm_infer_cfg,
        eval_cfg=bustm_eval_cfg,
    ),
]
