from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import KnowledgeRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import CHIDDataset_V2
from opencompass.utils.text_postprocessors import first_capital_postprocess

chid_knowledge_reader_cfg = dict(
    input_columns=["content", "A", "B", "C", "D", "E", "F", "G"],
    output_column="answer",
)

chid_knowledge_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template='以下是参考内容：{knowledge}，结合上述参考内容，考虑接下来的问题：'
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role="HUMAN",
                    prompt=
                    "</E>{content}\n请选择______处所填的词\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nE. {E}\nF. {F}\nG. {G}\n请从“A”，“B”，“C”，“D”，“E”，“F”，“G”中进行选择。答：",
                ),
            ]
        ),
        ice_token='</E>'
    ),
    retriever=dict(
        type=KnowledgeRetriever,
        knowledge_docs=[
            './data/knowledge/chengyu-01-of-02.txt',
            './data/knowledge/chengyu-02-of-02.txt',
            ],
        retrieve_keys=['A', 'B', 'C', 'D', 'E', 'F', 'G'],
        ice_eos_token='\n'
        ),
    inferencer=dict(type=GenInferencer),
)

chid_knowledge_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role="BOT",
    pred_postprocessor=dict(type=first_capital_postprocess),
)

chid_knowledge_datasets = [
    dict(
        abbr="chid-dev",
        type=CHIDDataset_V2,
        path="./data/FewCLUE/chid/dev_few_all.json",
        reader_cfg=chid_knowledge_reader_cfg,
        infer_cfg=chid_knowledge_infer_cfg,
        eval_cfg=chid_knowledge_eval_cfg,
    ),
    dict(
        abbr="chid-test",
        type=CHIDDataset_V2,
        path="./data/FewCLUE/chid/test_public.json",
        reader_cfg=chid_knowledge_reader_cfg,
        infer_cfg=chid_knowledge_infer_cfg,
        eval_cfg=chid_knowledge_eval_cfg,
    ),
]
