from mmengine.config import read_base
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.evaluator import GenericLLMEvaluator
from opencompass.datasets import generic_llmjudge_postprocess
from opencompass.datasets.PubMedQA import PubMedQADataset


QUERY_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of Options(e.g. one of ABCDEFGHIJKLMNOP). Think step by step before answering.
Question:\n
{question}
Options:\n
{choices}
""".strip()

GRADER_TEMPLATE = """
Please judge whether the candidate answer is consistent with the gold answer.
The gold answer is correct. Do not answer the original multiple-choice question.

Return exactly one letter and no other text:
X: CORRECT
Y: INCORRECT

<Original Question Begin>:
{question}
{choices}
<Original Question End>

<Gold Target Begin>:
{label}
<Gold Target End>

<Predicted Answer Begin>:
{prediction}
<Predicted Answer End>

Verdict:
""".strip()

PubMedQA_datasets = []

PubMedQA_reader_cfg = dict(
    input_columns=['question', 'choices'],
    output_column='label',
)

PubMedQA_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt=QUERY_TEMPLATE),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

PubMedQA_eval_cfg = dict(
    evaluator=dict(
        type=GenericLLMEvaluator,
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin=[
                    dict(
                        role='SYSTEM',
                        fallback_role='HUMAN',
                        prompt="You are a helpful assistant who evaluates the correctness and quality of models' outputs.",
                    )
                ],
                round=[
                    dict(role='HUMAN', prompt=GRADER_TEMPLATE),
                ],
            ),
        ),
        dataset_cfg=dict(
            type=PubMedQADataset,
            path='qiaojin/PubMedQA',
            reader_cfg=PubMedQA_reader_cfg,
        ),
        judge_cfg=dict(),
        dict_postprocessor=dict(
            type=generic_llmjudge_postprocess,
            true_tag='X',
            false_tag='Y',
        ),
    ),
)

PubMedQA_datasets.append(
    dict(
        abbr=f'PubMedQA',
        type=PubMedQADataset,
        path='qiaojin/PubMedQA',
        reader_cfg=PubMedQA_reader_cfg,
        infer_cfg=PubMedQA_infer_cfg,
        eval_cfg=PubMedQA_eval_cfg,
    )
)
