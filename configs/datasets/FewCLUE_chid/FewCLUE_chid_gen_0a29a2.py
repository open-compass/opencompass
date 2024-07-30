from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import CHIDDatasetV2
from opencompass.utils.text_postprocessors import first_capital_postprocess

chid_reader_cfg = dict(
    input_columns=['content','A','B','C','D','E','F','G'],
    output_column='answer',
)

chid_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt=
                    '{content}\n请选择______处所填的词\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nE. {E}\nF. {F}\nG. {G}\n请从”A“，”B“，”C“，”D“，”E“，”F“，”G“中进行选择。答：',
                ),
            ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

chid_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_capital_postprocess),
)

chid_datasets = [
    dict(
        abbr='chid-dev',
        type=CHIDDatasetV2,
        path='./data/FewCLUE/chid/dev_few_all.json',
        reader_cfg=chid_reader_cfg,
        infer_cfg=chid_infer_cfg,
        eval_cfg=chid_eval_cfg,
    ),
    dict(
        abbr='chid-test',
        type=CHIDDatasetV2,
        path='./data/FewCLUE/chid/test_public.json',
        reader_cfg=chid_reader_cfg,
        infer_cfg=chid_infer_cfg,
        eval_cfg=chid_eval_cfg,
    ),
]
