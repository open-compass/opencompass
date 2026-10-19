from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import CMNLIDatasetV2
from opencompass.utils.text_postprocessors import first_capital_postprocess

ocnli_fc_reader_cfg = dict(
    input_columns=['sentence1', 'sentence2'],
    output_column='label',
    test_split='train')

ocnli_fc_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt=
                '阅读文章：{sentence1}\n根据上文，回答如下问题：{sentence2}\nA. 对\nB. 错\nC. 可能\n请从“A”，“B”，“C”中进行选择。\n答：'
            ),
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)
ocnli_fc_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_capital_postprocess),
)

ocnli_fc_datasets = [
    dict(
        abbr='ocnli_fc-dev',
        type=CMNLIDatasetV2,  # ocnli_fc share the same format with cmnli
        path='./data/FewCLUE/ocnli/dev_few_all.json',
        local_mode=True,
        reader_cfg=ocnli_fc_reader_cfg,
        infer_cfg=ocnli_fc_infer_cfg,
        eval_cfg=ocnli_fc_eval_cfg,
    ),
    dict(
        abbr='ocnli_fc-test',
        type=CMNLIDatasetV2,  # ocnli_fc share the same format with cmnli
        path='./data/FewCLUE/ocnli/test_public.json',
        local_mode=True,
        reader_cfg=ocnli_fc_reader_cfg,
        infer_cfg=ocnli_fc_infer_cfg,
        eval_cfg=ocnli_fc_eval_cfg,
    ),
]
