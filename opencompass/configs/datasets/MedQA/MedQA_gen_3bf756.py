from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.utils.text_postprocessors import first_option_postprocess
from opencompass.datasets.MedQA import MedQADataset


QUERY_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of Options(e.g. one of ABCDEFGHIJKLMNOP). Think step by step before answering.

Question:\n
{question}

Options:\n
{choices}

""".strip()


MedQA_datasets = []

MedQA_reader_cfg = dict(
    input_columns=['question', 'choices'],
    output_column='label',
)

MedQA_infer_cfg = dict(
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

MedQA_subsets = {
    'US': 'xuxuxuxuxu/MedQA_US_test',
    'Mainland': 'xuxuxuxuxu/MedQA_Mainland_test',
    'Taiwan': 'xuxuxuxuxu/MedQA_Taiwan_test',
}

for split in list(MedQA_subsets.keys()):

    MedQA_eval_cfg = dict(
        evaluator=dict(type=AccEvaluator),
        pred_postprocessor=dict(type=first_option_postprocess, options='ABCD')
    )

    MedQA_datasets.append(
        dict(
            abbr=f'MedQA_{split}',
            type=MedQADataset,
            path=MedQA_subsets[split],
            reader_cfg=MedQA_reader_cfg,
            infer_cfg=MedQA_infer_cfg,
            eval_cfg=MedQA_eval_cfg,
        )
    )
