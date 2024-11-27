from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.PMMEval import PMMEvalMHellaswagDataset, PMMEvalMHellaswagEvaluator, pmmeval_mhellaswag_postprocess

NATURAL_LANGUAGE_CODES = ['en', 'zh', 'ar', 'es', 'fr', 'ja', 'ko', 'pt', 'th', 'vi']

PMMEVAL_MHELLASWAG_TEMPLATE = "Input: {ctx}\nOptions: \nA. {option_1}\nB. {option_2}\nC. {option_3}\nD. {option_4}\nPick the correct ending for the sentence from A, B, C, and D, and return it in the following JSON format:\n{\"answer\": \"[choice]\"}\nwhere [choice] must be one of A, B, C or D."

PMMEval_MHellaswag_datasets = list()

PMMEval_MHellaswag_reader_cfg = dict(
    input_columns=['ctx', 'option_1', 'option_2', 'option_3', 'option_4'],
    output_column='label',
    test_split='test'
)

PMMEval_MHellaswag_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt=PMMEVAL_MHELLASWAG_TEMPLATE
                )
            ]
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)


PMMEval_MHellaswag_datasets = list()


for lang_code in NATURAL_LANGUAGE_CODES:
    PMMEval_MHellaswag_eval_cfg = dict(
        evaluator=dict(type=PMMEvalMHellaswagEvaluator),
        pred_role='BOT',
        pred_postprocessor=dict(type=pmmeval_mhellaswag_postprocess, lang_code=lang_code)
    )

    PMMEval_MHellaswag_datasets.append(
        dict(
            abbr=f'mhellaswag-{lang_code}',
            type=PMMEvalMHellaswagDataset,
            path='P-MMEval',
            lang=lang_code,
            reader_cfg=PMMEval_MHellaswag_reader_cfg,
            infer_cfg=PMMEval_MHellaswag_infer_cfg,
            eval_cfg=PMMEval_MHellaswag_eval_cfg)
    )
