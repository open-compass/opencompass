import argparse
import json
import os
import os.path as osp
import re

import mmengine
from mmengine import Config
from mmengine.utils import mkdir_or_exist

from opencompass.datasets.humanevalx import _clean_up_code
from opencompass.utils import (dataset_abbr_from_cfg, get_infer_output_path,
                               get_logger, model_abbr_from_cfg)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Collect Humanevalx dataset predictions.')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('-r',
                        '--reuse',
                        nargs='?',
                        type=str,
                        const='latest',
                        help='Reuse previous outputs & results, and run any '
                        'missing jobs presented in the config. If its '
                        'argument is not specified, the latest results in '
                        'the work_dir will be reused. The argument should '
                        'also be a specific timestamp, e.g. 20230516_144254'),
    args = parser.parse_args()
    return args


_LANGUAGE_NAME_DICT = {
    'cpp': 'CPP',
    'go': 'Go',
    'java': 'Java',
    'js': 'JavaScript',
    'python': 'Python',
    'rust': 'Rust',
}
FAILED = 0
SUCCEED = 1


def gpt_python_postprocess(ori_prompt: str, text: str) -> str:
    """Better answer postprocessor for better instruction-aligned models like
    GPT."""
    if '```' in text:
        blocks = re.findall(r'```(.*?)```', text, re.DOTALL)
        if len(blocks) == 0:
            text = text.split('```')[1]  # fall back to default strategy
        else:
            text = blocks[0]  # fetch the first code block
            if not text.startswith('\n'):  # in case starting with ```python
                text = text[max(text.find('\n') + 1, 0):]

    match_ori = re.search(r'def(.*?)\(', ori_prompt)
    match = re.search(r'def(.*?)\(', text)
    if match:
        if match.group() == match_ori.group():
            text = re.sub('def(.*?)\n', '', text, count=1)

    for c_index, c in enumerate(text[:5]):
        if c != ' ':
            text = ' ' * (4 - c_index) + text
            break

    text = text.split('\n\n\n')[0]
    return text


def wizardcoder_postprocess(text: str) -> str:
    """Postprocess for WizardCoder Models."""
    if '```' in text:
        blocks = re.findall(r'```(.*?)```', text, re.DOTALL)
        if len(blocks) == 0:
            text = text.split('```')[1]  # fall back to default strategy
        else:
            text = blocks[0]  # fetch the first code block
            if not text.startswith('\n'):  # in case starting with ```python
                text = text[max(text.find('\n') + 1, 0):]
    else:
        match = re.search(r'Here(.*?)\n', text)
        if match:
            text = re.sub('Here(.*?)\n', '', text, count=1)

    return text


def collect_preds(filename: str):
    # in case the prediction is partial
    root, ext = osp.splitext(filename)
    partial_filename = root + '_0' + ext
    # collect all the prediction results
    if not osp.exists(osp.realpath(filename)) and not osp.exists(
            osp.realpath(partial_filename)):
        print(f'No predictions found for {filename}')
        return FAILED, None, None
    else:
        if osp.exists(osp.realpath(filename)):
            preds = mmengine.load(filename)
            pred_strs = [
                preds[str(i)]['prediction'] for i in range(len(preds))
            ]
            ori_prompt_strs = [
                preds[str(i)]['origin_prompt'] for i in range(len(preds))
            ]
        else:
            filename = partial_filename
            pred_strs = []
            ori_prompt_strs = []
            i = 1
            while osp.exists(osp.realpath(filename)):
                preds = mmengine.load(filename)
                filename = root + f'_{i}' + ext
                i += 1
                pred_strs += [
                    preds[str(i)]['prediction'] for i in range(len(preds))
                ]
                ori_prompt_strs += [
                    preds[str(i)]['origin_prompt'] for i in range(len(preds))
                ]
        return SUCCEED, ori_prompt_strs, pred_strs


def main():
    args = parse_args()
    # initialize logger
    logger = get_logger(log_level='INFO')
    cfg = Config.fromfile(args.config)
    cfg.setdefault('work_dir', './outputs/default/')

    assert args.reuse, 'Please provide the experienment work dir.'
    if args.reuse:
        if args.reuse == 'latest':
            if not os.path.exists(cfg.work_dir) or not os.listdir(
                    cfg.work_dir):
                logger.warning('No previous results to reuse!')
            else:
                dirs = os.listdir(cfg.work_dir)
                dir_time_str = sorted(dirs)[-1]
        else:
            dir_time_str = args.reuse
        logger.info(f'Reusing experiements from {dir_time_str}')
    # update "actual" work_dir
    cfg['work_dir'] = osp.join(cfg.work_dir, dir_time_str)

    for model in cfg.models:
        model_abbr = model_abbr_from_cfg(model)
        for dataset in cfg.datasets:
            dataset_abbr = dataset_abbr_from_cfg(dataset)
            filename = get_infer_output_path(
                model, dataset, osp.join(cfg.work_dir, 'predictions'))

            succeed, ori_prompt_strs, pred_strs = collect_preds(filename)
            if not succeed:
                continue

            # infer the language type
            for k, v in _LANGUAGE_NAME_DICT.items():
                if k in dataset_abbr:
                    lang = k
                    task = v
                    break

            # special postprocess for GPT
            if model_abbr in [
                    'WizardCoder-1B-V1.0',
                    'WizardCoder-3B-V1.0',
                    'WizardCoder-15B-V1.0',
                    'WizardCoder-Python-13B-V1.0',
                    'WizardCoder-Python-34B-V1.0',
            ]:
                predictions = [{
                    'task_id': f'{task}/{i}',
                    'generation': wizardcoder_postprocess(pred),
                } for i, pred in enumerate(pred_strs)]
            elif 'CodeLlama' not in model_abbr and lang == 'python':
                predictions = [{
                    'task_id':
                    f'{task}/{i}',
                    'generation':
                    gpt_python_postprocess(ori_prompt, pred),
                } for i, (ori_prompt,
                          pred) in enumerate(zip(ori_prompt_strs, pred_strs))]
            else:
                predictions = [{
                    'task_id': f'{task}/{i}',
                    'generation': _clean_up_code(pred, lang),
                } for i, pred in enumerate(pred_strs)]

            # save processed results if not exists
            result_file_path = os.path.join(cfg['work_dir'], 'humanevalx',
                                            model_abbr,
                                            f'humanevalx_{lang}.json')
            if osp.exists(result_file_path):
                logger.info(
                    f'File exists for {model_abbr}, skip copy from predictions.'  # noqa
                )
            else:
                mkdir_or_exist(osp.split(result_file_path)[0])
                with open(result_file_path, 'w') as f:
                    for pred in predictions:
                        f.write(json.dumps(pred) + '\n')


if __name__ == '__main__':
    main()
