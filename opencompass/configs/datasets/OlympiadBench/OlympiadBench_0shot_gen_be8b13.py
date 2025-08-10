from mmengine.config import read_base
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import OlympiadBenchDataset, OlympiadBenchEvaluator, olympiadbench_postprocess_v2


with read_base():
    from .OlympiadBench_categories import categories

# Create prompter instance for problems
olympiadbench_prompter_cfg = dict(
    type='OlympiadBenchPrompter'
)

olympiadbench_reader_cfg = dict(
    input_columns=[
        'problem', 'language', 'subject', 'question_type', 
        'answer_type', 'is_multiple_answer', 'unit', 'questions'
    ], 
    output_column='solution'
)

olympiadbench_datasets = []
for _name in categories:
    olympiadbench_infer_cfg = dict(
        prompt_template=dict(
            type='OlympiadBenchTemplate'
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )

    olympiadbench_eval_cfg = dict(
        evaluator=dict(type=OlympiadBenchEvaluator, version='v2'), 
        pred_postprocessor=dict(type=olympiadbench_postprocess_v2),
    )

    olympiadbench_datasets.append(
        dict(
            type=OlympiadBenchDataset,
            abbr=f'OlympiadBench_{_name}',
            path='opencompass/OlympiadBench',
            name=_name,
            reader_cfg=olympiadbench_reader_cfg,
            infer_cfg=olympiadbench_infer_cfg,
            eval_cfg=olympiadbench_eval_cfg,
        )
    )

del _name
