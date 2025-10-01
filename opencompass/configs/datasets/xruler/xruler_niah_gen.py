from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.xruler.xruler_niah import xRulerNiahDataset
from opencompass.datasets.xruler.xruler_niah import xRulerNiahEvaluator


# Ruler Dataset settings
niah_configurations = [
    {
        'abbr': 'single_1',
        'type_haystack': 'repeat',
        'type_needle_k': 'words',
        'type_needle_v': 'numbers',
        'num_needle_k': 1,
        'num_needle_v': 1,
        'num_needle_q': 1,
    },
    {
        'abbr': 'single_2',
        'type_haystack': 'essay',
        'type_needle_k': 'words',
        'type_needle_v': 'numbers',
        'num_needle_k': 1,
        'num_needle_v': 1,
        'num_needle_q': 1,
    },
    {
        'abbr': 'single_3',
        'type_haystack': 'essay',
        'type_needle_k': 'words',
        'type_needle_v': 'uuids',
        'num_needle_k': 1,
        'num_needle_v': 1,
        'num_needle_q': 1,
    },
    {
        'abbr': 'multikey_1',
        'type_haystack': 'essay',
        'type_needle_k': 'words',
        'type_needle_v': 'numbers',
        'num_needle_k': 4,
        'num_needle_v': 1,
        'num_needle_q': 1,
    },
    {
        'abbr': 'multikey_2',
        'type_haystack': 'needle',
        'type_needle_k': 'words',
        'type_needle_v': 'numbers',
        'num_needle_k': 1,
        'num_needle_v': 1,
        'num_needle_q': 1,
    },
    {
        'abbr': 'multikey_3',
        'type_haystack': 'needle',
        'type_needle_k': 'uuids',
        'type_needle_v': 'uuids',
        'num_needle_k': 1,
        'num_needle_v': 1,
        'num_needle_q': 1,
    },
    {
        'abbr': 'multivalue',
        'type_haystack': 'essay',
        'type_needle_k': 'words',
        'type_needle_v': 'numbers',
        'num_needle_k': 1,
        'num_needle_v': 4,
        'num_needle_q': 1,
    },
    {
        'abbr': 'multiquery',
        'type_haystack': 'essay',
        'type_needle_k': 'words',
        'type_needle_v': 'numbers',
        'num_needle_k': 1,
        'num_needle_v': 1,
        'num_needle_q': 4,
    },
]

niah_datasets = []
LANGS = ['en', 'zh', 'ar', 'bn', 'cs', 'de', 'es', 'fr', 'hu']
# NIAH Dataset
base_path = './data/xruler/un_test'
for lang in LANGS:
    file_path = f'un_test.{lang}.json'
    for index, config in enumerate(niah_configurations):
        dataset_dict = {
            'abbr': f'ruler_niah_{config["abbr"]}_{lang}',
            'type': xRulerNiahDataset,
            'base_path': base_path,
            'file_path': file_path,
            # 'tokenizer_model': model_path,
            'tokens_to_generate': 128,
            'type_haystack': config['type_haystack'],
            'type_needle_k': config['type_needle_k'],
            'type_needle_v': config['type_needle_v'],
            'num_needle_k': config['num_needle_k'],
            'num_needle_v': config['num_needle_v'],
            'num_needle_q': config['num_needle_q'],
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
                evaluator=dict(type=xRulerNiahEvaluator),
            ),
        }
        niah_datasets.append(dataset_dict)
