from opencompass.datasets.babilong.babilong import BabiLongDataset, BabiLongEvaluator
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer


babiLong_1m_datasets = []
split_name='1m'
tasks = ['qa1', 'qa2', 'qa3', 'qa4', 'qa5', 'qa6', 'qa7', 'qa8', 'qa9', 'qa10']


for task in tasks:
    tmp_dataset =  {
        'abbr': f'babilong_{task}_{split_name}',
        'type': BabiLongDataset,
        'path': 'opencompass/babilong',
        'task': task,
        'split_name': split_name,
        'reader_cfg': dict(input_columns=['prompt'], output_column='answer'),
        'infer_cfg': dict(
            prompt_template=dict(
                type=RawPromptTemplate,
                messages=[
                    {'role': 'user', 'content': '{prompt}'},
                ],
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer),
        ),
        'eval_cfg': dict(
            evaluator=dict(type=BabiLongEvaluator),
        ),
    }
    babiLong_1m_datasets.append(tmp_dataset)
