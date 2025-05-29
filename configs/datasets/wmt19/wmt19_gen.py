from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, BM25Retriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import BleuEvaluator
from opencompass.datasets.wmt19 import WMT19TranslationDataset

LANG_CODE_TO_NAME = {
    'cs': 'Czech',
    'de': 'German',
    'en': 'English',
    'fi': 'Finnish',
    'fr': 'French',
    'gu': 'Gujarati',
    'kk': 'Kazakh',
    'lt': 'Lithuanian',
    'ru': 'Russian',
    'zh': 'Chinese'
}

wmt19_reader_cfg = dict(
    input_columns=['input'],
    output_column='target',
    train_split='validation',
    test_split='validation')

wmt19_infer_cfg_0shot = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='Translate the following {src_lang_name} text to {tgt_lang_name}:\n{{input}}\n'),
                dict(role='BOT', prompt='Translation:\n')
            ]
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)

wmt19_infer_cfg_5shot = dict(
    ice_template=dict(
        type=PromptTemplate,
        template='Example:\n{src_lang_name}: {{input}}\n{tgt_lang_name}: {{target}}'
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template='</E>\nTranslate the following {src_lang_name} text to {tgt_lang_name}:\n{{input}}\nTranslation:\n',
        ice_token='</E>',
    ),
    retriever=dict(type=BM25Retriever, ice_num=5),
    inferencer=dict(type=GenInferencer),
)

wmt19_eval_cfg = dict(
    evaluator=dict(
        type=BleuEvaluator
    ),
    pred_role='BOT',
)

language_pairs = [
    ('cs', 'en'), ('de', 'en'), ('fi', 'en'), ('fr', 'de'), 
    ('gu', 'en'), ('kk', 'en'), ('lt', 'en'), ('ru', 'en'), ('zh', 'en')
]

wmt19_datasets = []

for src_lang, tgt_lang in language_pairs:
    src_lang_name = LANG_CODE_TO_NAME[src_lang]
    tgt_lang_name = LANG_CODE_TO_NAME[tgt_lang]
    
    wmt19_datasets.extend([
        dict(
            abbr=f'wmt19_{src_lang}-{tgt_lang}_0shot',
            type=WMT19TranslationDataset,
            path='/path/to/wmt19',
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            reader_cfg=wmt19_reader_cfg,
            infer_cfg={
                **wmt19_infer_cfg_0shot,
                'prompt_template': {
                    **wmt19_infer_cfg_0shot['prompt_template'],
                    'template': {
                        **wmt19_infer_cfg_0shot['prompt_template']['template'],
                        'round': [
                            {
                                **wmt19_infer_cfg_0shot['prompt_template']['template']['round'][0],
                                'prompt': wmt19_infer_cfg_0shot['prompt_template']['template']['round'][0]['prompt'].format(
                                    src_lang_name=src_lang_name, tgt_lang_name=tgt_lang_name
                                )
                            },
                            wmt19_infer_cfg_0shot['prompt_template']['template']['round'][1]
                        ]
                    }
                }
            },
            eval_cfg=wmt19_eval_cfg),
        dict(
            abbr=f'wmt19_{src_lang}-{tgt_lang}_5shot',
            type=WMT19TranslationDataset,
            path='/path/to/wmt19',
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            reader_cfg=wmt19_reader_cfg,
            infer_cfg={
                **wmt19_infer_cfg_5shot,
                'ice_template': {
                    **wmt19_infer_cfg_5shot['ice_template'],
                    'template': wmt19_infer_cfg_5shot['ice_template']['template'].format(
                        src_lang_name=src_lang_name, tgt_lang_name=tgt_lang_name
                    )
                },
                'prompt_template': {
                    **wmt19_infer_cfg_5shot['prompt_template'],
                    'template': wmt19_infer_cfg_5shot['prompt_template']['template'].format(
                        src_lang_name=src_lang_name, tgt_lang_name=tgt_lang_name
                    )
                }
            },
            eval_cfg=wmt19_eval_cfg),
    ])