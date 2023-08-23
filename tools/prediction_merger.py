import argparse
import copy
import json
import os.path as osp

import mmengine
from mmengine.config import Config, ConfigDict

from opencompass.utils import build_dataset_from_cfg, get_infer_output_path


def parse_args():
    parser = argparse.ArgumentParser(description='Run an evaluation task')
    parser.add_argument('config', help='Train config file path')
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


class PredictionMerger:
    """"""

    def __init__(self, cfg: ConfigDict) -> None:

        self.cfg = cfg
        self.model_cfg = copy.deepcopy(self.cfg['model'])
        self.dataset_cfg = copy.deepcopy(self.cfg['dataset'])
        self.work_dir = self.cfg.get('work_dir')

    def run(self):
        filename = get_infer_output_path(
            self.model_cfg, self.dataset_cfg,
            osp.join(self.work_dir, 'predictions'))
        root, ext = osp.splitext(filename)
        partial_filename = root + '_0' + ext

        if osp.exists(osp.realpath(filename)):
            return

        if not osp.exists(osp.realpath(partial_filename)):
            print(f'{filename} not found')
            return

        # Load predictions
        partial_filenames = []
        if osp.exists(osp.realpath(filename)):
            preds = mmengine.load(filename)
        else:
            preds, offset = {}, 0
            i = 1
            while osp.exists(osp.realpath(partial_filename)):
                partial_filenames.append(osp.realpath(partial_filename))
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


def dispatch_tasks(cfg):
    for model in cfg['models']:
        for dataset in cfg['datasets']:
            PredictionMerger({
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
    dispatch_tasks(cfg)


if __name__ == '__main__':
    main()
