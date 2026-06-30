from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import (ELBenchChoiceEvaluator,
                                  ELBenchHighLevelOmniDataset)

# 高阶育人-omni: multiple-choice cultivation questions (objective).
elbench_omni_reader_cfg = dict(
    input_columns=['question'],
    output_column='answer',
)

elbench_omni_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt='{question}\n\n请在回答的最后单独一行用“ANSWER: X”的格式给出'
                '正确选项（若有多个正确选项，用逗号分隔，例如 ANSWER: A, C）。',
            ),
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=2048),
)

elbench_omni_eval_cfg = dict(evaluator=dict(type=ELBenchChoiceEvaluator))

elbench_highlevel_omni_datasets = [
    dict(
        abbr='elbench_highlevel_omni',
        type=ELBenchHighLevelOmniDataset,
        path='高阶育人/omni',
        name='高阶育人-omni',
        reader_cfg=elbench_omni_reader_cfg,
        infer_cfg=elbench_omni_infer_cfg,
        eval_cfg=elbench_omni_eval_cfg,
    )
]
