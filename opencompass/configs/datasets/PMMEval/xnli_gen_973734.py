from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.PMMEval import PMMEvalXNLIDataset, PMMEvalXNLIEvaluator, pmmeval_xnli_postprocess

NATURAL_LANGUAGE_CODES = ['en', 'zh', 'ar', 'es', 'fr', 'ja', 'ko', 'pt', 'th', 'vi']

PMMEVAL_XNLI_TEMPLATE = """Take the following as truth: {premise}
Then the following statement: \"{statement}\" is
Options: 
A. true
B. inconclusive
C. false
Select the correct option from A, B, and C, and return it in the following JSON format:
{"answer": "[choice]"}
where [choice] must be one of A, B, and C."""

PMMEval_XNLI_datasets = list()

# Add flores_200

PMMEval_XNLI_reader_cfg = dict(
    input_columns=['premise', 'statement'],
    output_column='answer',
    test_split='test'
)


PMMEval_XNLI_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(
                        role='HUMAN',
                        prompt=PMMEVAL_XNLI_TEMPLATE
                    )
                ]
            )
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )

for lang_code in NATURAL_LANGUAGE_CODES:
    PMMEval_XNLI_eval_cfg = dict(
        evaluator=dict(type=PMMEvalXNLIEvaluator),
        pred_role='BOT',
        pred_postprocessor=dict(type=pmmeval_xnli_postprocess, lang_code=lang_code))

    PMMEval_XNLI_datasets.append(
        dict(
            abbr=f'xnli-{lang_code}',
            type=PMMEvalXNLIDataset,
            path='P-MMEval',
            lang=lang_code,
            reader_cfg=PMMEval_XNLI_reader_cfg,
            infer_cfg=PMMEval_XNLI_infer_cfg,
            eval_cfg=PMMEval_XNLI_eval_cfg)
    )
