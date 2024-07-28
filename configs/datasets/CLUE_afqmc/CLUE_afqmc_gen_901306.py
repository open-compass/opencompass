from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import AFQMCDatasetV2
from opencompass.utils.text_postprocessors import first_capital_postprocess

afqmc_reader_cfg = dict(
    input_columns=['sentence1', 'sentence2'],
    output_column='label',
    test_split='train')

afqmc_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt=
                '语句一：“{sentence1}”\n语句二：“{sentence2}”\n语句一与语句二是关于蚂蚁金融产品的疑问，两者所询问的内容是否完全一致？\nA. 不完全一致\nB. 完全一致\n请从“A”，“B”中进行选择。\n答：',
            ),
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

afqmc_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_capital_postprocess),
)

afqmc_datasets = [
    dict(
        abbr='afqmc-dev',
        type=AFQMCDatasetV2,
        path='opencompass/afqmc-dev',
        reader_cfg=afqmc_reader_cfg,
        infer_cfg=afqmc_infer_cfg,
        eval_cfg=afqmc_eval_cfg,
    ),
]
