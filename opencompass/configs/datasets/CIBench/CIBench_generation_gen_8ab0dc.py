from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import AgentInferencer

from opencompass.datasets import CIBenchDataset, CIBenchEvaluator

cibench_reader_cfg = dict(
    input_columns=['questions'],
    output_column='references',
    train_split='test',
    test_split='test')

cibench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="""{questions}""",
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=AgentInferencer, infer_mode='every'),
)

libs = ['matplotlib', 'opencv', 'pandas', 'pytorch', 'scipy', 'seaborn']
cibench_eval_cfg = dict(evaluator=dict(type=CIBenchEvaluator), pred_role='BOT')

cibench_datasets = [
    dict(
        abbr=f'cibench_generation/{lib}',
        type=CIBenchDataset,
        path=f'./data/cibench_dataset/cibench_generation/{lib}',
        internet_check=False,
        reader_cfg=cibench_reader_cfg,
        infer_cfg=cibench_infer_cfg,
        eval_cfg=cibench_eval_cfg,
    ) for lib in libs
]
