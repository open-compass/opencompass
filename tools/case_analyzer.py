import argparse
import copy
import json
import os.path as osp

import mmengine
from mmengine.config import Config, ConfigDict
from mmengine.utils import mkdir_or_exist
from tqdm import tqdm

from opencompass.registry import TEXT_POSTPROCESSORS
from opencompass.utils import build_dataset_from_cfg, get_infer_output_path


def parse_args():
    parser = argparse.ArgumentParser(description='Run case analyzer')
    parser.add_argument('config', help='Train config file path')
    parser.add_argument(
        '-f',
        '--force',
        help='Force to run the task even if the results already exist',
        action='store_true',
        default=False)
    parser.add_argument('-w',
                        '--work-dir',
                        help='Work path, all the outputs will be '
                        'saved in this path, including the slurm logs, '
                        'the evaluation results, the summary results, etc.'
                        'If not specified, the work_dir will be set to '
                        './outputs/default.',
                        default=None,
                        type=str)
    args = parser.parse_args()
    return args


class BadcaseShower:
    """"""

    def __init__(self, cfg: ConfigDict) -> None:

        self.cfg = cfg
        self.model_cfg = copy.deepcopy(self.cfg['model'])
        self.dataset_cfg = copy.deepcopy(self.cfg['dataset'])
        self.work_dir = self.cfg.get('work_dir')
        # Load Dataset
        self.eval_cfg = self.dataset_cfg.get('eval_cfg')
        self.ds_split = self.eval_cfg.get('ds_split', None)
        self.ds_column = self.eval_cfg.get('ds_column')

    def run(self):
        filename = get_infer_output_path(
            self.model_cfg, self.dataset_cfg,
            osp.join(self.work_dir, 'predictions'))
        root, ext = osp.splitext(filename)
        partial_filename = root + '_0' + ext

        if not osp.exists(osp.realpath(filename)) and not osp.exists(
                osp.realpath(partial_filename)):
            print(f'{filename} not found')
            return

        dataset = build_dataset_from_cfg(self.dataset_cfg)
        # Postprocess dataset if necessary
        if 'dataset_postprocessor' in self.eval_cfg:

            def postprocess(sample):
                s = sample[self.ds_column]
                proc = TEXT_POSTPROCESSORS.get(
                    self.eval_cfg['dataset_postprocessor']['type'])
                sample[self.ds_column] = proc(s)
                return sample

            dataset = dataset.map(postprocess)

        # Load predictions
        if osp.exists(osp.realpath(filename)):
            preds = mmengine.load(filename)
        else:
            filename = partial_filename
            preds, offset = {}, 0
            i = 1
            while osp.exists(osp.realpath(filename)):
                _preds = mmengine.load(filename)
                filename = root + f'_{i}' + ext
                i += 1
                for _o in range(len(_preds)):
                    preds[str(offset)] = _preds[str(_o)]
                    offset += 1
        pred_strs = [preds[str(i)]['prediction'] for i in range(len(preds))]

        # Postprocess predictions if necessary
        if 'pred_postprocessor' in self.eval_cfg:
            proc = TEXT_POSTPROCESSORS.get(
                self.eval_cfg['pred_postprocessor']['type'])
            pred_strs = [proc(s) for s in pred_strs]

        if self.ds_split:
            references = dataset[self.ds_split][self.ds_column]
        else:
            references = dataset[self.ds_column]

        if len(pred_strs) != len(references):
            print('length mismatch')
            return

        # combine cases
        allcase, badcase = [], []
        if 'in-context examples' in preds['0']:
            # ppl eval
            for i, (pred_str,
                    reference) in enumerate(zip(tqdm(pred_strs), references)):
                ref_str = str(reference)
                try:
                    pred_prompt = preds[str(i)]['label: ' +
                                                pred_str]['testing input']
                    pred_PPL = preds[str(i)]['label: ' + pred_str]['PPL']
                    ref_prompt = preds[str(i)]['label: ' +
                                               ref_str]['testing input']
                    ref_PPL = preds[str(i)]['label: ' + ref_str]['PPL']
                except KeyError:
                    continue
                item = {
                    'prediction_prompt': pred_prompt,
                    'prediction': pred_str,
                    'prediction_PPL': pred_PPL,
                    'reference_prompt': ref_prompt,
                    'reference': ref_str,
                    'reference_PPL': ref_PPL
                }
                if pred_str != ref_str:
                    badcase.append(item)
                    allcase.append(item)
                else:
                    allcase.append(item)

        else:
            # gen eval
            for i, (pred_str,
                    reference) in enumerate(zip(tqdm(pred_strs), references)):
                ref_str = str(reference)
                origin_prompt = preds[str(i)]['origin_prompt']
                item = {
                    'origin_prompt': origin_prompt,
                    'prediction': pred_str,
                    'reference': ref_str
                }
                # FIXME: we now consider all cases as bad cases
                badcase.append(item)
                allcase.append(item)

        # Save result
        out_path = get_infer_output_path(
            self.cfg['model'], self.cfg['dataset'],
            osp.join(self.work_dir, 'case_analysis/bad'))
        mkdir_or_exist(osp.split(out_path)[0])
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(badcase, f, indent=4, ensure_ascii=False)

        out_path = get_infer_output_path(
            self.cfg['model'], self.cfg['dataset'],
            osp.join(self.work_dir, 'case_analysis/all'))
        mkdir_or_exist(osp.split(out_path)[0])
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(allcase, f, indent=4, ensure_ascii=False)


def dispatch_tasks(cfg, force=False):
    for model in cfg['models']:
        for dataset in cfg['datasets']:
            if force or not osp.exists(
                    get_infer_output_path(
                        model, dataset,
                        osp.join(cfg['work_dir'], 'case_analysis/all'))):
                BadcaseShower({
                    'model': model,
                    'dataset': dataset,
                    'work_dir': cfg['work_dir']
                }).run()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    # set work_dir
    if args.work_dir is not None:
        cfg['work_dir'] = args.work_dir
    else:
        cfg.setdefault('work_dir', './outputs/default')
    dispatch_tasks(cfg, force=args.force)


if __name__ == '__main__':
    main()
