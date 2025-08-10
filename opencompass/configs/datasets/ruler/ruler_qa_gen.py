from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.ruler.ruler_qa import RulerQaDataset
from opencompass.datasets.ruler.ruler_qa import RulerQaEvaluator

qa_configurations = [
    {'dataset': 'squad', 'path': './data/ruler/dev-v2.0.json'},
    {'dataset': 'hotpotqa', 'path': './data/ruler/hotpotqa.json'},
]

qa_datasets = []
for index, config in enumerate(qa_configurations):
    dataset_dict = {
        'abbr': f'ruler_qa_{config["dataset"]}',
        'dataset': config['dataset'],
        'path': config['path'],
        'type': RulerQaDataset,
        'tokens_to_generate': 50,
        'reader_cfg': dict(input_columns=['prompt'], output_column='answer'),
        'infer_cfg': dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    round=[
                        dict(role='HUMAN', prompt='{prompt}'),
                        dict(role='BOT', prompt='{answer}\n'),
                    ]
                ),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer),
        ),
        'eval_cfg': dict(
            evaluator=dict(type=RulerQaEvaluator),
        ),
    }
    qa_datasets.append(dataset_dict)
