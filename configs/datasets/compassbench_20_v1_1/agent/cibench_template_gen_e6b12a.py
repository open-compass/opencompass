from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import AgentInferencer
from opencompass.datasets import CIBenchDataset, CIBenchEvaluator

libs = [
    '/lightgbm',
    '/matplotlib',
    '/nltk',
    '/opencv',
    '/pandas',
    '/pytorch',
    '/scipy',
    '/seaborn',
    '/sklearn',
    '/tensorflow',
    '_chinese/lightgbm',
    '_chinese/matplotlib',
    '_chinese/nltk',
    '_chinese/opencv',
    '_chinese/pandas',
    '_chinese/pytorch',
    '_chinese/scipy',
    '_chinese/seaborn',
    '_chinese/sklearn',
    '_chinese/tensorflow',
]


cibench_datasets = []
for lib in libs:
    cibench_reader_cfg = dict(
        input_columns=['questions'], output_column='references', train_split='test', test_split='test'
    )

    cibench_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template='{questions}',
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=AgentInferencer, infer_mode='every'),
    )

    cibench_eval_cfg = dict(evaluator=dict(type=CIBenchEvaluator), pred_role='BOT')

    cibench_datasets.append(
        dict(
            abbr=f'cibench_template{lib}',
            type=CIBenchDataset,
            path=f'data/compassbench_v1.1/agent-cibench/cibench_template{lib}',
            internet_check=False,
            reader_cfg=cibench_reader_cfg,
            infer_cfg=cibench_infer_cfg,
            eval_cfg=cibench_eval_cfg,
        )
    )
