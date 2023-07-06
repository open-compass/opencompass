import argparse
import fnmatch
from typing import Dict

from mmengine.config import Config, ConfigDict

from opencompass.openicl.icl_inferencer import (CLPInferencer, GenInferencer,
                                                PPLInferencer)
from opencompass.registry import ICL_PROMPT_TEMPLATES, ICL_RETRIEVERS
from opencompass.utils import (Menu, build_dataset_from_cfg,
                               build_model_from_cfg, dataset_abbr_from_cfg,
                               model_abbr_from_cfg)


def parse_args():
    parser = argparse.ArgumentParser(description='Run an evaluation task')
    parser.add_argument('config', help='Train config file path')
    parser.add_argument('-n', '--non-interactive', action='store_true')
    parser.add_argument('-a', '--all', action='store_true')
    parser.add_argument('-p',
                        '--pattern',
                        type=str,
                        help='To match the dataset abbr.')
    args = parser.parse_args()
    return args


def parse_model_cfg(model_cfg: ConfigDict) -> Dict[str, ConfigDict]:
    model2cfg = {}
    for model in model_cfg:
        model2cfg[model_abbr_from_cfg(model)] = model
    return model2cfg


def parse_dataset_cfg(dataset_cfg: ConfigDict) -> Dict[str, ConfigDict]:
    dataset2cfg = {}
    for dataset in dataset_cfg:
        dataset2cfg[dataset_abbr_from_cfg(dataset)] = dataset
    return dataset2cfg


def print_prompts(model_cfg, dataset_cfg):
    # TODO: A really dirty method that copies code from PPLInferencer and
    # GenInferencer. In the future, the prompt extraction code should be
    # extracted and generalized as a static method in these Inferencers
    # and reused here.
    if model_cfg:
        max_seq_len = model_cfg.max_seq_len
        if not model_cfg['type'].is_api:
            model_cfg['tokenizer_only'] = True
        model = build_model_from_cfg(model_cfg)
    else:
        max_seq_len = None
        model = None

    infer_cfg = dataset_cfg.get('infer_cfg')

    fix_id_list = infer_cfg.inferencer.get('fix_id_list', [])
    dataset = build_dataset_from_cfg(dataset_cfg)

    ice_template = None
    if hasattr(infer_cfg, 'ice_template'):
        ice_template = ICL_PROMPT_TEMPLATES.build(infer_cfg['ice_template'])

    prompt_template = None
    if hasattr(infer_cfg, 'prompt_template'):
        prompt_template = ICL_PROMPT_TEMPLATES.build(
            infer_cfg['prompt_template'])

    infer_cfg['retriever']['dataset'] = dataset
    retriever = ICL_RETRIEVERS.build(infer_cfg['retriever'])

    if fix_id_list:
        ice_idx_list = retriever.retrieve(fix_id_list)
    else:
        ice_idx_list = retriever.retrieve()

    assert infer_cfg.inferencer.type in [PPLInferencer, GenInferencer], \
        'Only PPLInferencer and GenInferencer are supported'

    if infer_cfg.inferencer.type == PPLInferencer:
        labels = retriever.get_labels(ice_template=ice_template,
                                      prompt_template=prompt_template)
        ice = [
            retriever.generate_ice(ice_idx_list[idx],
                                   ice_template=ice_template)
            for idx in range(len(ice_idx_list))
        ]
        print('-' * 100)
        print('ICE Template:')
        print('-' * 100)
        print(ice[0])
        print('-' * 100)
        for label in labels:
            idx = 0
            prompt = retriever.generate_label_prompt(
                idx,
                ice[idx],
                label,
                ice_template=ice_template,
                prompt_template=prompt_template,
                remain_sep=None)
            if max_seq_len is not None:
                prompt_token_num = model.get_token_len_from_template(prompt)
                while len(ice_idx_list[idx]
                          ) > 0 and prompt_token_num > max_seq_len:
                    num_ice = len(ice_idx_list[idx])
                    print(f'Truncating ice {num_ice} -> {num_ice - 1}',
                          f'Number of tokens: {prompt_token_num} -> ...')
                    ice_idx_list[idx] = ice_idx_list[idx][:-1]
                    ice[idx] = retriever.generate_ice(
                        ice_idx_list[idx], ice_template=ice_template)
                    prompt = retriever.generate_label_prompt(
                        idx,
                        ice[idx],
                        label,
                        ice_template=ice_template,
                        prompt_template=prompt_template)
                    prompt_token_num = model.get_token_len_from_template(
                        prompt)
                print(f'Number of tokens: {prompt_token_num}')
            if model is not None:
                prompt = model.parse_template(prompt, mode='ppl')
            print('-' * 100)
            print(f'Label: {label}')
            print('Sample prompt:')
            print('-' * 100)
            print(prompt)
            print('-' * 100)
    elif infer_cfg.inferencer.type in [GenInferencer, CLPInferencer]:
        idx, ice_idx = 0, ice_idx_list[0]
        ice = retriever.generate_ice(ice_idx, ice_template=ice_template)
        prompt = retriever.generate_prompt_for_generate_task(
            idx,
            ice,
            gen_field_replace_token=infer_cfg.inferencer.get(
                'gen_field_replace_token', ''),
            ice_template=ice_template,
            prompt_template=prompt_template)
        if max_seq_len is not None:
            prompt_token_num = model.get_token_len_from_template(prompt)
            while len(ice_idx) > 0 and prompt_token_num > max_seq_len:
                num_ice = len(ice_idx)
                print(f'Truncating ice {num_ice} -> {num_ice - 1}',
                      f'Number of tokens: {prompt_token_num} -> ...')
                ice_idx = ice_idx[:-1]
                ice = retriever.generate_ice(ice_idx,
                                             ice_template=ice_template)
                prompt = retriever.generate_prompt_for_generate_task(
                    idx,
                    ice,
                    gen_field_replace_token=infer_cfg.inferencer.get(
                        'gen_field_replace_token', ''),
                    ice_template=ice_template,
                    prompt_template=prompt_template)
                prompt_token_num = model.get_token_len_from_template(prompt)
            print(f'Number of tokens:  {prompt_token_num}')
        if model is not None:
            prompt = model.parse_template(prompt, mode='gen')
        print('-' * 100)
        print('Sample prompt:')
        print('-' * 100)
        print(prompt)
        print('-' * 100)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    # cfg.models =
    model2cfg = parse_model_cfg(cfg.models) if 'models' in cfg else {
        'None': None
    }
    if 'datasets' in cfg:
        dataset2cfg = parse_dataset_cfg(cfg.datasets)
    else:
        dataset2cfg = {}
        for key in cfg.keys():
            if key.endswith('_datasets'):
                dataset2cfg.update(parse_dataset_cfg(cfg[key]))

    if args.pattern is not None:
        matches = fnmatch.filter(dataset2cfg, args.pattern)
        if len(matches) == 0:
            raise ValueError(
                'No dataset match the pattern. Please select from: \n' +
                '\n'.join(dataset2cfg.keys()))
        dataset2cfg = {k: dataset2cfg[k] for k in matches}

    if not args.all:
        if not args.non_interactive:
            model, dataset = Menu(
                [list(model2cfg.keys()),
                 list(dataset2cfg.keys())], [
                     f'Please make a selection of {s}:'
                     for s in ['model', 'dataset']
                 ]).run()
        else:
            model = list(model2cfg.keys())[0]
            dataset = list(dataset2cfg.keys())[0]
        model_cfg = model2cfg[model]
        dataset_cfg = dataset2cfg[dataset]
        print_prompts(model_cfg, dataset_cfg)
    else:
        for model_abbr, model_cfg in model2cfg.items():
            for dataset_abbr, dataset_cfg in dataset2cfg.items():
                print('=' * 64, '[BEGIN]', '=' * 64)
                print(f'[MODEL]: {model_abbr}')
                print(f'[DATASET]: {dataset_abbr}')
                print('---')
                print_prompts(model_cfg, dataset_cfg)
                print('=' * 65, '[END]', '=' * 65)
                print()


if __name__ == '__main__':
    main()
