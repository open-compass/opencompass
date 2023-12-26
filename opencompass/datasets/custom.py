import csv
import json
import os

from datasets import Dataset

from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer, PPLInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.registry import LOAD_DATASET
from opencompass.utils.text_postprocessors import first_option_postprocess

from .base import BaseDataset


@LOAD_DATASET.register_module()
class CustomDataset(BaseDataset):

    @staticmethod
    def load(path):
        if path.endswith('.jsonl'):
            with open(path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]
        elif path.endswith('.csv'):
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                data = [dict(zip(header, row)) for row in reader]
        else:
            raise ValueError(f'Unsupported file format: {path}')

        return Dataset.from_list(data)


def stringfy_types(obj):
    for k, v in obj.items():
        if k == 'type':
            obj[k] = f'{v.__module__}.{v.__name__}'
        elif isinstance(v, dict):
            stringfy_types(v)
    return obj


def make_mcq_gen_config(meta):
    if meta.get('template', None) is None:
        _human_prompt = 'Question: {question}' + ''.join(
            [f'\n{item}. {{{item}}}' for item in meta['options']])
        human_prompt = meta.get('human_prompt', _human_prompt)
        _bot_prompt = f'Answer: {{{meta["output_column"]}}}'
        bot_prompt = meta.get('bot_prompt', _bot_prompt)
        template = dict(round=[
            dict(role='HUMAN', prompt=human_prompt),
            dict(role='BOT', prompt=bot_prompt),
        ])
    else:
        template = meta['template']

    reader_cfg = dict(
        input_columns=meta['input_columns'],
        output_column=meta['output_column'],
    )
    infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=template,
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )

    eval_cfg = dict(evaluator=dict(type=AccEvaluator),
                    pred_role='BOT',
                    pred_postprocessor=dict(
                        type=first_option_postprocess,
                        options=''.join(meta['options']),
                    ))

    dataset = dict(
        abbr=meta['abbr'],
        type=CustomDataset,
        path=meta['path'],
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
    )
    return dataset


def make_qa_gen_config(meta):
    if meta.get('template', None) is None:
        human_prompt = meta.get('human_prompt', '{question}')
        if meta['output_column'] is None:
            template = dict(round=[
                dict(role='HUMAN', prompt=human_prompt),
            ])
        else:
            bot_prompt = meta.get('bot_prompt', f'{{{meta["output_column"]}}}')
            template = dict(round=[
                dict(role='HUMAN', prompt=human_prompt),
                dict(role='BOT', prompt=bot_prompt),
            ])
    else:
        template = meta['template']

    reader_cfg = dict(
        input_columns=meta['input_columns'],
        output_column=meta['output_column'],
    )
    infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=template,
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )

    eval_cfg = dict(
        evaluator=dict(type=AccEvaluator),
        pred_role='BOT',
    )

    dataset = dict(
        abbr=meta['abbr'],
        type=CustomDataset,
        path=meta['path'],
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
    )
    return dataset


def make_mcq_ppl_config(meta):
    if meta.get('template', None) is None:
        _human_prompt = 'Question: {question}' + ''.join(
            [f'\n{item}. {{{item}}}' for item in meta['options']])
        human_prompt = meta.get('human_prompt', _human_prompt)
        _bot_prompt = f'Answer: {{{meta["output_column"]}}}'
        bot_prompt = meta.get('bot_prompt', _bot_prompt)
        template = {
            answer: dict(round=[
                dict(role='HUMAN', prompt=human_prompt),
                dict(role='BOT',
                     prompt=bot_prompt.format(
                         **{meta['output_column']: answer})),
            ], )
            for answer in meta['options']
        }
    else:
        template = meta['template']

    reader_cfg = dict(
        input_columns=meta['input_columns'],
        output_column=meta['output_column'],
    )
    infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=template,
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=PPLInferencer),
    )

    eval_cfg = dict(evaluator=dict(type=AccEvaluator))

    dataset = dict(
        abbr=meta['abbr'],
        type=CustomDataset,
        path=meta['path'],
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
    )
    return dataset


def parse_example_dataset(config):
    # try to read meta json
    path = config['path']
    meta_path = config.get('meta_path', path + '.meta.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
    else:
        meta = {}

    # load sample
    if path.endswith('.jsonl'):
        with open(path, 'r', encoding='utf-8') as f:
            data_item = json.loads(f.readline())
    elif path.endswith('.csv'):
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            row = next(reader)
            data_item = dict(zip(header, row))
    else:
        raise ValueError(f'Unsupported ext: {path}, .jsonl or .csv required')

    meta['path'] = path
    input_columns = [i for i in data_item.keys() if i != 'answer']
    meta.setdefault('input_columns', input_columns)
    output_column = 'answer' if 'answer' in data_item else None
    meta.setdefault('output_column', output_column)
    options = []
    for i in range(26):
        i = chr(ord('A') + i)
        if i in data_item:
            options.append(i)
        else:
            break
    meta.setdefault('options', options)
    abbr = os.path.basename(path).split('.')[0]
    meta.setdefault('abbr', abbr)

    if 'data_type' in config:
        meta.setdefault('data_type', config['data_type'])
    else:
        data_type = 'mcq' if len(options) > 1 else 'qa'
        meta.setdefault('data_type', data_type)
    if 'infer_method' in config:
        meta.setdefault('infer_method', config['infer_method'])
    else:
        meta.setdefault('infer_method', 'gen')

    return meta


def make_custom_dataset_config(config):
    # considered as a custom dataset
    meta = parse_example_dataset(config)
    make_config_func = {
        ('mcq', 'gen'): make_mcq_gen_config,
        ('mcq', 'ppl'): make_mcq_ppl_config,
        ('qa', 'gen'): make_qa_gen_config,
    }.get((meta['data_type'], meta['infer_method']), None)
    if make_config_func is None:
        raise ValueError(f'Unsupported dataset data_type: {meta["data_type"]}'
                         f' and infer_method: {meta["infer_method"]}')
    dataset = make_config_func(meta)
    dataset = stringfy_types(dataset)
    return dataset
