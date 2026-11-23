from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import HardMTBenchDataset, HardMTBenchEvaluator

hardmtbench_reader_cfg = dict(
    input_columns=['target_language_zh', 'source_text'],
    output_column='reference',
)

HARDMTBENCH_INFER_TEMPLATE = (
    '将以下文本翻译为{target_language_zh}，注意只需要输出翻译后的结果，不要额外解释：'
    '\n\n{source_text}'
)

hardmtbench_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            {'role': 'user', 'content': HARDMTBENCH_INFER_TEMPLATE},
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

GEMBA_DA_TEMPLATE = (
    'Score the following translation from {source_language} to '
    '{target_language} on a continuous scale from 0 to 100, where a score of '
    'zero means "no meaning preserved" and score of one hundred means '
    '"perfect meaning and grammar".\n\n'
    '{source_language} source: "{source_text}"\n'
    '{target_language} translation: "{prediction}"\n'
    'Score: '
)

GEMBA_DA_REF_TEMPLATE = (
    'Score the following translation from {source_language} to '
    '{target_language} with respect to human reference on a continuous scale '
    '0 to 100 where score of zero means "no meaning preserved" and score of '
    'one hundred means "perfect meaning and grammar".\n\n'
    '{source_language} source: "{source_text}"\n'
    '{target_language} human reference: {reference}\n'
    '{target_language} machine translation: "{prediction}"\n'
    'Score: '
)

hardmtbench_eval_cfg = dict(
    evaluator=dict(
        type=HardMTBenchEvaluator,
        prompt_template=dict(
            type=RawPromptTemplate,
            messages=[
                {'role': 'user', 'content': GEMBA_DA_TEMPLATE},
            ],
        ),
        judge_cfg=dict(),
        metrics=['gemba_da', 'xcomet_xxl', 'term_acc'],
        xcomet_api_base=None,
        pred_postprocessor=dict(type='extract-non-reasoning-content'),
    ),
)

hardmtbench_datasets = [
    dict(
        type=HardMTBenchDataset,
        abbr=f'hardmtbench_{direction}',
        path='opencompass/hardmtbench',
        direction=direction,
        reader_cfg=hardmtbench_reader_cfg,
        infer_cfg=hardmtbench_infer_cfg,
        eval_cfg=hardmtbench_eval_cfg,
    ) for direction in ['zh2en', 'en2zh']
]
