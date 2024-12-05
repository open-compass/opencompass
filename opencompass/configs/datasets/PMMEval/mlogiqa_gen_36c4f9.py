from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.PMMEval import PMMEvalMLogiQADataset, PMMEvalMLogiQAEvaluator, pmmeval_mlogiqa_postprocess

NATURAL_LANGUAGE_CODES = ['en', 'zh', 'ar', 'es', 'fr', 'ja', 'ko', 'pt', 'th', 'vi']

PMMEVAL_MLOGIQA_TEMPLATE = "Passage: {context}\nQuestion: {question}\nChoices:\nA.{option_1}\nB.{option_2}\nC.{option_3}\nD.{option_4}\nPlease choose the most suitable one among A, B, C and D as the answer to this question, and return it in the following JSON format:\n{'answer': '[choice]'}\nwhere [choice] must be one of A, B, C and D."

PMMEval_MLogiQA_datasets = []


PMMEval_MLogiQA_reader_cfg = dict(
    input_columns=['context', 'question', 'option_1', 'option_2', 'option_3', 'option_4'],
    output_column='answer',
    train_split='test')

PMMEval_MLogiQA_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt=PMMEVAL_MLOGIQA_TEMPLATE
                )
            ]
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)


for lang_code in NATURAL_LANGUAGE_CODES:
    PMMEval_MLogiQA_eval_cfg = dict(
        evaluator=dict(type=PMMEvalMLogiQAEvaluator),
        pred_role='BOT',
        pred_postprocessor=dict(type=pmmeval_mlogiqa_postprocess, lang_code=lang_code))

    PMMEval_MLogiQA_datasets.append(
        dict(
            abbr=f'mlogiqa-{lang_code}',
            type=PMMEvalMLogiQADataset,
            path='P-MMEval',
            lang=lang_code,
            reader_cfg=PMMEval_MLogiQA_reader_cfg,
            infer_cfg=PMMEval_MLogiQA_infer_cfg,
            eval_cfg=PMMEval_MLogiQA_eval_cfg)
    )
