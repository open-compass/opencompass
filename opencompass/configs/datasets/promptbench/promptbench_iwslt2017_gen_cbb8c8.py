from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import BM25Retriever
from opencompass.openicl.icl_inferencer import AttackInferencer
from opencompass.openicl.icl_evaluator import BleuEvaluator
from opencompass.datasets import IWSLT2017Dataset
from opencompass.utils.text_postprocessors import general_cn_postprocess

iwslt2017_reader_cfg = dict(
    input_columns='en', output_column='de', train_split='validation')

original_prompt_list = [
        'Translate the provided sentence from English to German while maintaining the original meaning and context:',
        'Convert the following sentence from its original English language to the target language German:',
        'Given the sentence below, perform a machine translation from English to German:',
        'Translate the subsequent sentence from its source language English into the desired language German:',
        'Accurately translate the sentence from English to German, ensuring the meaning remains intact:',
        'Please perform a translation of the given sentence, converting it from English to German:',
        'Translate the following text from the source language English to the target language German:',
        'Using machine translation, convert the given sentence from English into the German language:',
        'Translate the subsequent text passage from its original English language to the German language:',
        'Perform a machine translation for the provided sentence, changing it from English to German:',
]

iwslt2017_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(
            begin='</E>',
            round=[
                dict(role='HUMAN', prompt='{adv_prompt}\n{en}'),
                dict(role='BOT', prompt='{de}'),
            ]
        ),
        ice_token='</E>'),
    retriever=dict(type=BM25Retriever, ice_num=1),
    inferencer=dict(
        type=AttackInferencer,
        original_prompt_list=original_prompt_list,
        adv_key='adv_prompt',
        metric_key='score'))

iwslt2017_eval_cfg = dict(
    evaluator=dict(type=BleuEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=general_cn_postprocess),
    dataset_postprocessor=dict(type=general_cn_postprocess))

iwslt2017_datasets = [
    dict(
        abbr='iwslt',
        type=IWSLT2017Dataset,
        path='iwslt2017',
        name='iwslt2017-en-de',
        reader_cfg=iwslt2017_reader_cfg,
        infer_cfg=iwslt2017_infer_cfg,
        eval_cfg=iwslt2017_eval_cfg)
]
