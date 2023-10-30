from pathlib import Path
from typing import List

import typer
from mmengine.config import Config
from typer import Option

from opencompass.registry import build_from_cfg
from opencompass.summarizers.multi_model import MultiModelSummarizer

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command(help='Visualize the results of multiple models')
def main(
    cfg_paths: List[Path] = Option(
        ...,
        help='The path to the config file of the task',
        exists=True,
    ),
    work_dirs: List[Path] = Option(
        ...,
        help='The work dirs for the task(named by timestamp), '
        'need to ensure the order is the same as cfg_paths.',
        exists=True,
    ),
    group: str = Option(None,
                        help='If not None, show the accuracy in the group.'),
):
    assert len(cfg_paths) == len(work_dirs)
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
    app()
