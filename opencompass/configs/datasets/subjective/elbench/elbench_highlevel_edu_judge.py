from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.evaluator import GenericLLMEvaluator
from opencompass.datasets import (ELBenchHighLevelEduDataset,
                                  elbench_highlevel_edu_postprocess)

# 高阶育人-edu: open-ended educational responses scored by an LLM judge.
subjective_reader_cfg = dict(
    input_columns=['question'],
    output_column='judge',
)

base_prompt = """你是一名资深的教育专家评审。下面给出一道面向学生的教育任务，以及“被评估模型”的回答。请从教育性、准确性、针对性与建设性等角度，对该回答的整体质量进行评分。

[任务]
{question}

[被评估模型的回答]
{prediction}

请先简要分析回答的优缺点，然后在最后单独一行用 [[分数]] 的格式给出 1 到 10 的整数评分（10 表示非常优秀，1 表示很差）。最后一行必须且只能是形如 [[7]] 的内容。
"""

subjective_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            {'role': 'user', 'content': '{question}'},
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

subjective_eval_cfg = dict(
    evaluator=dict(
        type=GenericLLMEvaluator,
        prompt_template=dict(
            type=RawPromptTemplate,
            messages=[
                {'role': 'user', 'content': base_prompt},
            ],
        ),
        dataset_cfg=dict(
            type=ELBenchHighLevelEduDataset,
            path='高阶育人/edu',
            name='高阶育人-edu',
            reader_cfg=subjective_reader_cfg,
        ),
        judge_cfg=dict(),
        dict_postprocessor=dict(type=elbench_highlevel_edu_postprocess),
    ),
)

elbench_highlevel_edu_datasets = [
    dict(
        abbr='elbench_highlevel_edu',
        type=ELBenchHighLevelEduDataset,
        path='高阶育人/edu',
        name='高阶育人-edu',
        reader_cfg=subjective_reader_cfg,
        infer_cfg=subjective_infer_cfg,
        eval_cfg=subjective_eval_cfg,
        mode='singlescore',
    )
]
