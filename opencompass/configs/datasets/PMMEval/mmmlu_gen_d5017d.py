from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.PMMEval import PMMEvalMMMLUDataset, PMMEvalMMMLUEvaluator, pmmeval_mmmlu_postprocess

NATURAL_LANGUAGE_CODES_MMMLU = ['EN-US', 'ZH-CN', 'AR-XY', 'ES-LA', 'FR-FR', 'JA-JP', 'KO-KR', 'PT-BR', 'TH-TL', 'VI-VT']

PMMEVAL_MMMLU_TEMPLATE = "The following is a multiple-choice question. Please choose the most suitable one among A, B, C and D as the answer to this question, and return it in the following JSON format:\n{\"answer\": \"[choice]\"}\nwhere [choice] must be one of A, B, C and D.\n\n{Question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}"

PMMEval_MMMLU_datasets = []


PMMEval_MMMLU_reader_cfg = dict(
    input_columns=['Question', 'A', 'B', 'C', 'D'],
    output_column='Answer',
    train_split='test')


PMMEval_MMMLU_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt=PMMEVAL_MMMLU_TEMPLATE
                )
            ]
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)


for lang_code in NATURAL_LANGUAGE_CODES_MMMLU:
    PMMEval_MMMLU_eval_cfg = dict(
        evaluator=dict(type=PMMEvalMMMLUEvaluator),
        pred_role='BOT',
        pred_postprocessor=dict(type=pmmeval_mmmlu_postprocess, lang_code=lang_code))

    PMMEval_MMMLU_datasets.append(
        dict(
            abbr=f'mmmlu-{lang_code}',
            type=PMMEvalMMMLUDataset,
            path='P-MMEval',
            lang=lang_code,
            difficulty='all',
            reader_cfg=PMMEval_MMMLU_reader_cfg,
            infer_cfg=PMMEval_MMMLU_infer_cfg,
            eval_cfg=PMMEval_MMMLU_eval_cfg)
    )
