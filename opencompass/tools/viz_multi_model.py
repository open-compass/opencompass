import argparse
from pathlib import Path
from typing import List

from mmengine.config import Config

from opencompass.registry import build_from_cfg
from opencompass.summarizers.multi_model import MultiModelSummarizer


def parse_args(parser):
    parser.add_argument(
        'cfg_paths',
        type=List[Path],
        nargs='+',
        help='The path(s) to the config file(s) of the task',
        default=[],
    )
    parser.add_argument(
        'work_dirs',
        type=List[Path],
        nargs='+',
        help='The work dir(s) for the task (named by timestamp)\
            , need to ensure the order is the same as cfg_paths.',
        default=[],
    )
    parser.add_argument(
        '--group',
        type=str,
        default=None,
        help='If not None, show the accuracy in the group.',
    )
    return parser


def main(args):
    cfg_paths = args.cfg_paths
    work_dirs = args.work_dirs
    group = args.group

    cfgs = [Config.fromfile(it, format_python_code=False) for it in cfg_paths]

    multi_models_summarizer = None
    for cfg, work_dir in zip(cfgs, work_dirs):
        cfg['work_dir'] = work_dir
        summarizer_cfg = cfg.get('summarizer', {})
        summarizer_cfg['type'] = MultiModelSummarizer
        summarizer_cfg['config'] = cfg
        summarizer = build_from_cfg(summarizer_cfg)
        if multi_models_summarizer is None:
            multi_models_summarizer = summarizer
        else:
            multi_models_summarizer.merge(summarizer)
    multi_models_summarizer.summarize()
    if group:
        multi_models_summarizer.show_group(group)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize the results of multiple models')
    parser = parse_args(parser)
    args = parser.parse_args()
    main(args)
