import argparse
import copy
import json
import os

import mmengine
from mmengine.config import Config, ConfigDict

from opencompass.utils import build_dataset_from_cfg, get_infer_output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Merge patitioned predictions')
    parser.add_argument('config', help='Train config file path')
    parser.add_argument('-w', '--work-dir', default=None, type=str)
    parser.add_argument('-r', '--reuse', default='latest', type=str)
    parser.add_argument('-c', '--clean', action='store_true')
    parser.add_argument('-f', '--force', action='store_true')
    args = parser.parse_args()
    return args


class PredictionMerger:

    def __init__(self, cfg: ConfigDict) -> None:
        self.cfg = cfg
        self.model_cfg = copy.deepcopy(self.cfg['model'])
        self.dataset_cfg = copy.deepcopy(self.cfg['dataset'])
        self.work_dir = self.cfg.get('work_dir')

    def run(self):
        filename = get_infer_output_path(
            self.model_cfg, self.dataset_cfg,
            os.path.join(self.work_dir, 'predictions'))
        root, ext = os.path.splitext(filename)
        partial_filename = root + '_0' + ext

        if os.path.exists(
                os.path.realpath(filename)) and not self.cfg['force']:
            return

        if not os.path.exists(os.path.realpath(partial_filename)):
            print(f'{filename} not found')
            return

        # Load predictions
        partial_filenames = []
        preds, offset = {}, 0
        i = 1
        while os.path.exists(os.path.realpath(partial_filename)):
            partial_filenames.append(os.path.realpath(partial_filename))
            _preds = mmengine.load(partial_filename)
            partial_filename = root + f'_{i}' + ext
            i += 1
            for _o in range(len(_preds)):
                preds[str(offset)] = _preds[str(_o)]
                offset += 1

        dataset = build_dataset_from_cfg(self.dataset_cfg)
        if len(preds) != len(dataset.test):
            print('length mismatch')
            return

        print(f'Merge {partial_filenames} to {filename}')
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(preds, f, indent=4, ensure_ascii=False)

        if self.cfg['clean']:
            for partial_filename in partial_filenames:
                print(f'Remove {partial_filename}')
                os.remove(partial_filename)


def dispatch_tasks(cfg):
    for model in cfg['models']:
        for dataset in cfg['datasets']:
            PredictionMerger({
                'model': model,
                'dataset': dataset,
                'work_dir': cfg['work_dir'],
                'clean': cfg['clean'],
                'force': cfg['force'],
            }).run()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    # set work_dir
    if args.work_dir is not None:
        cfg['work_dir'] = args.work_dir
    else:
        cfg.setdefault('work_dir', './outputs/default')

    if args.reuse:
        if args.reuse == 'latest':
            if not os.path.exists(cfg.work_dir) or not os.listdir(
                    cfg.work_dir):
                print('No previous results to reuse!')
                return
            else:
                dirs = os.listdir(cfg.work_dir)
                dir_time_str = sorted(dirs)[-1]
        else:
            dir_time_str = args.reuse
    cfg['work_dir'] = os.path.join(cfg.work_dir, dir_time_str)

    cfg['clean'] = args.clean
    cfg['force'] = args.force

    dispatch_tasks(cfg)


if __name__ == '__main__':
    main()
