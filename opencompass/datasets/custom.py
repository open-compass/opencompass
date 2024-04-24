import copy
import csv
import json
import os
from typing import List

from datasets import Dataset

from opencompass.datasets.circular import (CircularDatasetMeta,
                                           CircularEvaluator)
from opencompass.openicl.icl_evaluator import AccEvaluator, BaseEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer, PPLInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


class OptionSimAccEvaluator(BaseEvaluator):

    def __init__(self, options) -> None:
        super().__init__()
        if not all((isinstance(i, str) and i.isupper() and len(i) == 1)
                   for i in options):
            raise ValueError(
                f'Each options should be single upper letter, got {options}')

        self.options = options

    def match_any_label(self, pred, test_item):
        from rapidfuzz.distance import Levenshtein as L

        from opencompass.utils.text_postprocessors import \
            first_option_postprocess

        pred = pred.strip()
        if any([pred == i for i in self.options]):
            parsed = pred
        else:
            parsed = ''
        if parsed == '':
            parsed = first_option_postprocess(pred,
                                              ''.join(self.options),
                                              cushion=False)
        if parsed == '':
            possible_options = []
            for opt in self.options:
                opt_str = test_item[opt]
                if opt_str is not None and opt_str.lower() in pred.lower():
                    possible_options.append(opt)
            if len(possible_options) == 1:
                parsed = possible_options[0]
        if parsed == '':
            dists = []
            for opt in self.options:
                opt_str = test_item[opt]
                if opt_str is None:
                    continue
                cands = [opt, opt_str, opt + '. ' + opt_str]
                d = min(L.distance(pred, cand) for cand in cands)
                dists.append((d, opt))
            if len(dists) > 0:
                parsed = min(dists)[1]
        return parsed

    def score(self, predictions: List, references: List, test_set) -> dict:
        assert len(predictions) == len(references)

        num_correct, num_total = 0, 0
        details = {}
        for index in range(len(predictions)):
            pred = predictions[index]
            refr = references[index]
            parsed = self.match_any_label(pred, test_set[index])
            num_correct += 1 if parsed == refr else 0
            num_total += 1
            details[str(index)] = {}
            details[str(index)]['pred'] = pred
            details[str(index)]['parsed'] = parsed
            details[str(index)]['refr'] = refr
            details[str(index)]['correct'] = parsed == refr
        return {'accuracy': num_correct / num_total * 100, 'details': details}


# TODO: DO NOT COPY YOURSELF!!!
class CircularOptionSimAccEvaluator(OptionSimAccEvaluator):

    def __init__(self, options, circular_pattern='circular'):
        super().__init__(options)
        self.circular_pattern = circular_pattern

    def score(self, predictions, references, test_set):
        from opencompass.datasets.circular import (get_all_possible_patterns,
                                                   get_circular_patterns,
                                                   get_origin_patterns)

        circular_patterns = {}
        circular_patterns['origin'] = get_origin_patterns(
            test_set[0]['circular_pattern'])
        circular_patterns['circular'] = get_circular_patterns(
            test_set[0]['circular_pattern'])
        if self.circular_pattern == 'all_possible':
            circular_patterns['all_possible'] = get_all_possible_patterns(
                test_set[0]['circular_pattern'])

        metrics = {}
        tmp_metrics = {}
        tmp_metrics.update({f'correct_{k}': 0 for k in circular_patterns})
        tmp_metrics.update({f'count_{k}': 0 for k in circular_patterns})
        # calculate the original accuracy
        for pred, refr, origin_item in zip(predictions, references, test_set):
            parsed = self.match_any_label(pred, origin_item)
            circular_pattern = origin_item['circular_pattern']
            for k in circular_patterns:
                if tuple(circular_pattern) in circular_patterns[k]:
                    tmp_metrics[f'correct_{k}'] += (1 if parsed == refr else 0)
                    tmp_metrics[f'count_{k}'] += 1

        for k in circular_patterns:
            metrics[f'acc_{k}'] = (tmp_metrics[f'correct_{k}'] /
                                   tmp_metrics[f'count_{k}'] * 100)

        # calculate the circular accuracy
        _details = {k: {} for k in circular_patterns}
        for pred, refr, origin_item in zip(predictions, references, test_set):
            index = origin_item['qid']
            parsed = self.match_any_label(pred, origin_item)
            circular_pattern = origin_item['circular_pattern']
            for k in circular_patterns:
                if tuple(circular_pattern) in circular_patterns[k]:
                    _details[k].setdefault(
                        index, []).append(True if parsed == refr else False)
        for k in _details:
            _details[k] = {
                index: sum(_details[k][index])
                for index in _details[k]
            }
        for k in _details:
            for j in range(1, len(circular_patterns[k]) + 1):
                count = sum([_details[k][index] >= j for index in _details[k]])
                total = len(_details[k])
                if j != len(circular_patterns[k]):
                    metrics[f'more_{j}_{k}'] = count / total * 100
                else:
                    metrics[f'perf_{k}'] = count / total * 100

        # make details
        details = {}
        for index in range(len(predictions)):
            parsed = self.match_any_label(predictions[index], test_set[index])
            details[str(index)] = {}
            if 'question' in test_set[index]:
                details[str(index)]['question'] = test_set[index]['question']
            details[str(index)]['pred'] = predictions[index]
            details[str(index)]['parsed'] = parsed
            details[str(index)]['refr'] = references[index]
            details[str(index)]['correct'] = parsed == references[index]
        metrics['details'] = details
        return metrics


@LOAD_DATASET.register_module()
class CustomDataset(BaseDataset):

    @staticmethod
    def load(path):
        if path.endswith('.jsonl'):
            with open(path, 'r', encoding='utf-8-sig') as f:
                data = [json.loads(line) for line in f]
        elif path.endswith('.csv'):
            with open(path, 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                header = next(reader)
                data = [dict(zip(header, row)) for row in reader]
        else:
            raise ValueError(f'Unsupported file format: {path}')

        return Dataset.from_list(data)


class CircularCustomDataset(CustomDataset, metaclass=CircularDatasetMeta):
    dataset_class = CustomDataset


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
    if 'test_range' in meta:
        reader_cfg['test_range'] = meta['test_range']
    infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=template,
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )

    eval_cfg = dict(
        evaluator=dict(type=meta.get('evaluator', OptionSimAccEvaluator),
                       **meta.get('evaluator_kwargs',
                                  {'options': meta['options']})),
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


def make_circular_mcq_gen_config(meta):
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
    if 'test_range' in meta:
        reader_cfg['test_range'] = meta['test_range']
    infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=template,
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )

    eval_cfg = dict(
        evaluator=dict(type=meta.get('evaluator',
                                     CircularOptionSimAccEvaluator),
                       **meta.get('evaluator_kwargs',
                                  {'options': meta['options']})),
        pred_role='BOT',
    )

    dataset = dict(
        abbr=meta['abbr'],
        type=CircularCustomDataset,
        option_keys=meta['options'],
        answer_key=meta['output_column'],
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
    if 'test_range' in meta:
        reader_cfg['test_range'] = meta['test_range']
    infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=template,
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )

    eval_cfg = dict(
        evaluator=dict(type=meta.get('evaluator', AccEvaluator),
                       **meta.get('evaluator_kwargs', {})),
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
    if 'test_range' in meta:
        reader_cfg['test_range'] = meta['test_range']
    infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=template,
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=PPLInferencer),
    )

    eval_cfg = dict(evaluator=dict(type=meta.get('evaluator', AccEvaluator),
                                   **meta.get('evaluator_kwargs', {})))

    dataset = dict(
        abbr=meta['abbr'],
        type=CustomDataset,
        path=meta['path'],
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
    )
    return dataset


def make_circular_mcq_ppl_config(meta):
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
    if 'test_range' in meta:
        reader_cfg['test_range'] = meta['test_range']
    infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=template,
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=PPLInferencer),
    )

    eval_cfg = dict(
        evaluator=dict(type=meta.get('evaluator', CircularEvaluator),
                       **meta.get('evaluator_kwargs', {})))

    dataset = dict(
        abbr=meta['abbr'],
        type=CircularCustomDataset,
        option_keys=meta['options'],
        answer_key=meta['output_column'],
        path=meta['path'],
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
    )
    return dataset


def parse_example_dataset(config):
    # config -> .meta.jsonl -> parsed_results
    path = config['path']

    # load sample and get parsed_meta
    parsed_meta = {}
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

    parsed_meta['path'] = path
    input_columns = [i for i in data_item.keys() if i != 'answer']
    parsed_meta['input_columns'] = input_columns
    output_column = 'answer' if 'answer' in data_item else None
    parsed_meta['output_column'] = output_column
    options = []
    for i in range(26):
        i = chr(ord('A') + i)
        if i in data_item:
            options.append(i)
        else:
            break
    parsed_meta['options'] = options
    abbr = os.path.basename(path).split('.')[0]
    parsed_meta['abbr'] = abbr
    parsed_meta['data_type'] = 'mcq' if len(options) > 1 else 'qa'
    parsed_meta['infer_method'] = 'gen'

    # try to read meta json
    meta_path = config.get('meta_path', path + '.meta.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            read_from_file_meta = json.load(f)
    else:
        read_from_file_meta = {}

    # get config meta
    config_meta = copy.deepcopy(config)

    # merge meta
    meta = {}
    meta.update(parsed_meta)
    meta.update(read_from_file_meta)
    meta.update(config_meta)

    return meta


def make_custom_dataset_config(config):
    # considered as a custom dataset
    meta = parse_example_dataset(config)
    make_config_func = {
        ('mcq', 'gen'): make_mcq_gen_config,
        ('mcq', 'ppl'): make_mcq_ppl_config,
        ('qa', 'gen'): make_qa_gen_config,
        ('circular-mcq', 'gen'): make_circular_mcq_gen_config,
        ('circular-mcq', 'ppl'): make_circular_mcq_ppl_config,
    }.get((meta['data_type'], meta['infer_method']), None)
    if make_config_func is None:
        raise ValueError(f'Unsupported dataset data_type: {meta["data_type"]}'
                         f' and infer_method: {meta["infer_method"]}')
    dataset = make_config_func(meta)
    dataset = stringfy_types(dataset)
    return dataset
