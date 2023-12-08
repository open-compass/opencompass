from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.datasets.subjective_cmp import SubjectiveCmpDataset
from mmengine.config import read_base

with read_base():
    from .prompt_template import build_prompt

subjective_reader_cfg = dict(
    input_columns=['question', 'index', 'reference_answer', 'evaluating_guidance', 'capability', 'prompt'],
    output_column='judge',
    train_split='test')

subjective_all_sets = [
    "creation_v0.1",
]

meta_prompt = build_prompt()

subjective_datasets = []

for _name in subjective_all_sets:
    subjective_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(
                        role='HUMAN',
                        prompt="{question}"
                    ),
                ]),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer),
        )

    subjective_eval_cfg = dict(
        evaluator=dict(
            type=LMEvaluator,
            cmp_order='both',
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(
                        role='HUMAN',
                        prompt = meta_prompt+"问题: <问题开始> {question} <问题结束>\n\n回答 1: <回答 1 开始> {prediction} <回答 1 结束>\n\n回答 2: <回答 2 开始> {prediction2} <回答 2 结束>\n\n参考答案: <参考答案开始> {reference_answer} <参考答案结束>\n\n题目评分指引: <题目评分指引开始> {evaluating_guidance} <题目评分指引结束>\n\n"
                    ),
                ]),
            ),
        ),
        pred_role="BOT",
    )

    subjective_datasets.append(
        dict(
            abbr=f"{_name}",
            type=SubjectiveCmpDataset,
            path="./data/subjective/",
            name=_name,
            reader_cfg=subjective_reader_cfg,
            infer_cfg=subjective_infer_cfg,
            eval_cfg=subjective_eval_cfg
        ))
