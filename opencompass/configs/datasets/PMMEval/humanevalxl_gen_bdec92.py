from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.PMMEval import PMMEvalHumanEvalXLDataset, PMMEvalHumanEvalXLEvaluator

NATURAL_LANGUAGE_FULLNAMES = ['English', 'Chinese', 'Arabic', 'Spanish', 'French', 'Japanese', 'Korean', 'Portuguese', 'Thai', 'Vietnamese']

PMMEval_HumanEvalXL_datasets = list()

PMMEval_HumanEvalXL_reader_cfg = dict(
    input_columns=['task_id', 'prompt', 'entry_point', 'test', 'language', 'description', 'natural_language'],
    output_column='declaration',
    test_split='test'
)

PMMEval_HumanEvalXL_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='{prompt}'),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)


PMMEval_HumanEvalXL_datasets = list()

for lang_fullname in NATURAL_LANGUAGE_FULLNAMES:
    for program_lang in ['python', 'java', 'javascript']:

        PMMEval_HumanEvalXL_eval_cfg = dict(
            evaluator=dict(
                type=PMMEvalHumanEvalXLEvaluator,
                language=program_lang,
                text_language=lang_fullname,
                ip_address='localhost',
                port=5001),
            pred_role='BOT')

        PMMEval_HumanEvalXL_datasets.append(
            dict(
                abbr=f'humanevalxl-{program_lang}-{lang_fullname}',
                type=PMMEvalHumanEvalXLDataset,
                path='P-MMEval',
                lang=lang_fullname,
                program_lang=program_lang,
                reader_cfg=PMMEval_HumanEvalXL_reader_cfg,
                infer_cfg=PMMEval_HumanEvalXL_infer_cfg,
                eval_cfg=PMMEval_HumanEvalXL_eval_cfg)
        )
