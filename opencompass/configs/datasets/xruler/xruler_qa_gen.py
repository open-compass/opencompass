from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.xruler.ruler_qa import xRulerQaDataset
from opencompass.datasets.xruler.ruler_qa import xRulerQaEvaluator
LANGS = ['ar', 'bn', 'cs']
qa_configurations = [
    {'lang': lang, 'dataset': 'squad', 'path': f'./data/xruler/xquad/xquad_{lang}.json'} for lang in LANGS
    # {'dataset': 'hotpotqa', 'path': './data/ruler/hotpotqa.json'},
]

qa_datasets = []
for index, config in enumerate(qa_configurations):
    dataset_dict = {
        'abbr': f'xruler_qa_{config["dataset"]}_{config["lang"]}',
        'dataset': config['dataset'],
        'path': config['path'],
        'lang': config['lang'],
        'type': xRulerQaDataset,
        'tokens_to_generate': 50,
        'max_seq_length': 1024 * 8,
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
            evaluator=dict(type=xRulerQaEvaluator),
        ),
    }
    qa_datasets.append(dataset_dict)
