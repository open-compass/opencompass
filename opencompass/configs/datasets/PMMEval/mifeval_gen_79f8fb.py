from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.PMMEval import PMMEvalMIFEvalDataset, PMMEvalMIFEvalEvaluator, pmmeval_mifeval_postprocess

NATURAL_LANGUAGE_CODES = ['en', 'zh', 'ar', 'es', 'fr', 'ja', 'ko', 'pt', 'th', 'vi']

PMMEVAL_MIFEVAL_TEMPLATE = '{prompt}'

PMMEval_MIFEval_datasets = list()

PMMEval_MIFEval_reader_cfg = dict(
    input_columns=['prompt', 'instruction_id_list', 'kwargs'],
    output_column=None,
    test_split='test'
)


PMMEval_MIFEval_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt=PMMEVAL_MIFEVAL_TEMPLATE
                )
            ]
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

for lang_code in NATURAL_LANGUAGE_CODES:
    PMMEval_MIFEval_eval_cfg = dict(
        evaluator=dict(type=PMMEvalMIFEvalEvaluator),
        pred_role='BOT',
        pred_postprocessor=dict(type=pmmeval_mifeval_postprocess, lang_code=lang_code)
    )

    PMMEval_MIFEval_datasets.append(
        dict(
            abbr=f'mifeval-{lang_code}',
            type=PMMEvalMIFEvalDataset,
            path='P-MMEval',
            lang=lang_code,
            reader_cfg=PMMEval_MIFEval_reader_cfg,
            infer_cfg=PMMEval_MIFEval_infer_cfg,
            eval_cfg=PMMEval_MIFEval_eval_cfg)
    )
