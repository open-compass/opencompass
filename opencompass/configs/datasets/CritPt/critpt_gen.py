from opencompass.datasets.critpt import (CritPtDataset, CritPtEvaluator,
                                         CritPtInferencer)
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever


critpt_reader_cfg = dict(
    input_columns=['messages'],
    output_column='target',
    train_split='test',
    test_split='test',
)

critpt_prompt_template = dict(
    type=RawPromptTemplate,
    messages=[{
        'expand_column': 'messages'
    }],
    format_variables=False,
)

critpt_eval_cfg = dict(
    evaluator=dict(
        type=CritPtEvaluator,
        submit=False,
    ),
    pred_role='BOT',
)

critpt_infer_cfg = dict(
    prompt_template=critpt_prompt_template,
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=CritPtInferencer, max_out_len=32768),
)

critpt_common = dict(
    path='CritPt-Benchmark/CritPt',
    parsing=False,
    use_python=False,
    use_web_search=False,
)

critpt_datasets = [
    dict(
        type=CritPtDataset,
        abbr='CritPt_main',
        reader_cfg=critpt_reader_cfg,
        infer_cfg=critpt_infer_cfg,
        eval_cfg=critpt_eval_cfg,
        **critpt_common,
    ),
]
