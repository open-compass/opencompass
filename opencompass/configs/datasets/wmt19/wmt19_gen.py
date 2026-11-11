from opencompass.datasets.wmt19 import WMT19TranslationDataset
from opencompass.openicl.icl_evaluator import BleuEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import BM25Retriever, ZeroRetriever

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

wmt19_reader_cfg_0shot = dict(input_columns=['input'],
                              output_column='target',
                              train_split='validation',
                              test_split='validation')

wmt19_reader_cfg_5shot = dict(input_columns=['input'],
                              output_column='target',
                              train_split='train',
                              test_split='validation')

wmt19_prompt_messages = [
    dict(
        role='user',
        content=('Translate the following {src_lang_name} text '
                 'to {tgt_lang_name}:\n{{input}}\nTranslation:\n'),
    ),
]

wmt19_infer_cfg_0shot = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=wmt19_prompt_messages,
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

wmt19_infer_cfg_5shot = dict(
    ice_template=dict(
        type=RawPromptTemplate,
        messages=[
            dict(
                role='user',
                content=('Example:\n{src_lang_name}: {{input}}\n'
                         '{tgt_lang_name}: '),
            ),
            dict(role='assistant', content='{{target}}'),
        ],
    ),
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=['</E>', *wmt19_prompt_messages],
        ice_token='</E>',
    ),
    retriever=dict(type=BM25Retriever, ice_num=5),
    inferencer=dict(type=GenInferencer),
)

wmt19_eval_cfg = dict(
    evaluator=dict(type=BleuEvaluator),
    pred_role='BOT',
)

language_pairs = [('cs', 'en'), ('de', 'en'), ('fi', 'en'), ('fr', 'de'),
                  ('gu', 'en'), ('kk', 'en'), ('lt', 'en'), ('ru', 'en'),
                  ('zh', 'en')]

wmt19_datasets = []

for src_lang, tgt_lang in language_pairs:
    src_lang_name = LANG_CODE_TO_NAME[src_lang]
    tgt_lang_name = LANG_CODE_TO_NAME[tgt_lang]
    prompt_messages = [
        {
            **message,
            'content': message['content'].format(
                src_lang_name=src_lang_name,
                tgt_lang_name=tgt_lang_name,
            ),
        }
        for message in wmt19_prompt_messages
    ]
    ice_messages = [
        {
            **message,
            'content': message['content'].format(
                src_lang_name=src_lang_name,
                tgt_lang_name=tgt_lang_name,
            ),
        }
        for message in wmt19_infer_cfg_5shot['ice_template']['messages']
    ]

    wmt19_datasets.extend([
        dict(abbr=f'wmt19_{src_lang}-{tgt_lang}_0shot',
             type=WMT19TranslationDataset,
             path='wmt/wmt19',
             src_lang=src_lang,
             tgt_lang=tgt_lang,
             reader_cfg=wmt19_reader_cfg_0shot,
             infer_cfg={
                 **wmt19_infer_cfg_0shot, 'prompt_template': {
                     **wmt19_infer_cfg_0shot['prompt_template'],
                     'messages': prompt_messages,
                 }
             },
             eval_cfg=wmt19_eval_cfg),
        dict(abbr=f'wmt19_{src_lang}-{tgt_lang}_5shot',
             type=WMT19TranslationDataset,
             path='wmt/wmt19',
             src_lang=src_lang,
             tgt_lang=tgt_lang,
             train_samples=512,
             reader_cfg=wmt19_reader_cfg_5shot,
             infer_cfg={
                 **wmt19_infer_cfg_5shot, 'ice_template': {
                     **wmt19_infer_cfg_5shot['ice_template'],
                     'messages': ice_messages,
                 },
                 'prompt_template': {
                     **wmt19_infer_cfg_5shot['prompt_template'],
                     'messages': ['</E>', *prompt_messages],
                 }
             },
             eval_cfg=wmt19_eval_cfg),
    ])
