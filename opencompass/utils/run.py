import os
from typing import List, Union

import tabulate
from mmengine.config import Config

from opencompass.partitioners import NaivePartitioner, SizePartitioner
from opencompass.runners import DLCRunner, LocalRunner, SlurmRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask
from opencompass.utils import get_logger, match_files


def match_cfg_file(workdir: str, pattern: Union[str, List[str]]) -> List[str]:
    """Match the config file in workdir recursively given the pattern.

    Additionally, if the pattern itself points to an existing file, it will be
    directly returned.
    """
    if isinstance(pattern, str):
        pattern = [pattern]
    pattern = [p + '.py' if not p.endswith('.py') else p for p in pattern]
    files = match_files(workdir, pattern, fuzzy=False)
    if len(files) != len(pattern):
        nomatched = []
        ambiguous = []
        err_msg = ('The provided pattern matches 0 or more than one '
                   'config. Please verify your pattern and try again. '
                   'You may use tools/list_configs.py to list or '
                   'locate the configurations.\n')
        for p in pattern:
            files = match_files(workdir, p, fuzzy=False)
            if len(files) == 0:
                nomatched.append([p[:-3]])
            elif len(files) > 1:
                ambiguous.append([p[:-3], '\n'.join(f[1] for f in files)])
        if nomatched:
            table = [['Not matched patterns'], *nomatched]
            err_msg += tabulate.tabulate(table,
                                         headers='firstrow',
                                         tablefmt='psql')
        if ambiguous:
            table = [['Ambiguous patterns', 'Matched files'], *ambiguous]
            err_msg += tabulate.tabulate(table,
                                         headers='firstrow',
                                         tablefmt='psql')
        raise ValueError(err_msg)
    return files


def get_config_from_arg(args) -> Config:
    """Get the config object given args.

    Only a few argument combinations are accepted (priority from high to low)
    1. args.config
    2. args.models and args.datasets
    3. Huggingface parameter groups and args.datasets
    """
    if args.config:
        return Config.fromfile(args.config, format_python_code=False)
    if args.datasets is None:
        raise ValueError('You must specify "--datasets" if you do not specify '
                         'a config file path.')
    datasets = []
    datasets_dir = os.path.join(args.config_dir, 'datasets')
    for dataset in match_cfg_file(datasets_dir, args.datasets):
        get_logger().info(f'Loading {dataset[0]}: {dataset[1]}')
        cfg = Config.fromfile(dataset[1])
        for k in cfg.keys():
            if k.endswith('_datasets'):
                datasets += cfg[k]
    if not args.models and not args.hf_path:
        raise ValueError('You must specify a config file path, '
                         'or specify --models and --datasets, or '
                         'specify HuggingFace model parameters and '
                         '--datasets.')
    models = []
    if args.models:
        model_dir = os.path.join(args.config_dir, 'models')
        for model in match_cfg_file(model_dir, args.models):
            get_logger().info(f'Loading {model[0]}: {model[1]}')
            cfg = Config.fromfile(model[1])
            if 'models' not in cfg:
                raise ValueError(
                    f'Config file {model[1]} does not contain "models" field')
            models += cfg['models']
    else:
        from opencompass.models import HuggingFace
        model = dict(type=f'{HuggingFace.__module__}.{HuggingFace.__name__}',
                     path=args.hf_path,
                     peft_path=args.peft_path,
                     tokenizer_path=args.tokenizer_path,
                     model_kwargs=args.model_kwargs,
                     tokenizer_kwargs=args.tokenizer_kwargs,
                     max_seq_len=args.max_seq_len,
                     max_out_len=args.max_out_len,
                     batch_padding=not args.no_batch_padding,
                     batch_size=args.batch_size,
                     pad_token_id=args.pad_token_id,
                     run_cfg=dict(num_gpus=args.num_gpus))
        models.append(model)

    summarizer = None
    if args.summarizer:
        summarizers_dir = os.path.join(args.config_dir, 'summarizers')
        s = match_cfg_file(summarizers_dir, [args.summarizer])[0]
        get_logger().info(f'Loading {s[0]}: {s[1]}')
        cfg = Config.fromfile(s[1])
        summarizer = cfg['summarizer']

    return Config(dict(models=models, datasets=datasets,
                       summarizer=summarizer),
                  format_python_code=False)


def exec_mm_infer_runner(tasks, args, cfg):
    """execute multimodal infer runner according to args."""
    if args.slurm:
        runner = SlurmRunner(dict(type='MultimodalInferTask'),
                             max_num_workers=args.max_num_workers,
                             partition=args.partition,
                             quotatype=args.quotatype,
                             retry=args.retry,
                             debug=args.debug,
                             lark_bot_url=cfg['lark_bot_url'])
    elif args.dlc:
        raise NotImplementedError('Currently, we do not support evaluating \
                             multimodal models on dlc.')
    else:
        runner = LocalRunner(task=dict(type='MultimodalInferTask'),
                             max_num_workers=args.max_num_workers,
                             debug=args.debug,
                             lark_bot_url=cfg['lark_bot_url'])
    runner(tasks)


def get_config_type(obj) -> str:
    return f'{obj.__module__}.{obj.__name__}'


def fill_infer_cfg(cfg, args):
    new_cfg = dict(infer=dict(
        partitioner=dict(type=get_config_type(SizePartitioner),
                         max_task_size=args.max_partition_size,
                         gen_task_coef=args.gen_task_coef),
        runner=dict(
            max_num_workers=args.max_num_workers,
            debug=args.debug,
            task=dict(type=get_config_type(OpenICLInferTask)),
            lark_bot_url=cfg['lark_bot_url'],
        )), )
    if args.slurm:
        new_cfg['infer']['runner']['type'] = get_config_type(SlurmRunner)
        new_cfg['infer']['runner']['partition'] = args.partition
        new_cfg['infer']['runner']['quotatype'] = args.quotatype
        new_cfg['infer']['runner']['qos'] = args.qos
        new_cfg['infer']['runner']['retry'] = args.retry
    elif args.dlc:
        new_cfg['infer']['runner']['type'] = get_config_type(DLCRunner)
        new_cfg['infer']['runner']['aliyun_cfg'] = Config.fromfile(
            args.aliyun_cfg)
        new_cfg['infer']['runner']['retry'] = args.retry
    else:
        new_cfg['infer']['runner']['type'] = get_config_type(LocalRunner)
        new_cfg['infer']['runner'][
            'max_workers_per_gpu'] = args.max_workers_per_gpu
    cfg.merge_from_dict(new_cfg)


def fill_eval_cfg(cfg, args):
    new_cfg = dict(
        eval=dict(partitioner=dict(type=get_config_type(NaivePartitioner)),
                  runner=dict(
                      max_num_workers=args.max_num_workers,
                      debug=args.debug,
                      task=dict(type=get_config_type(OpenICLEvalTask)),
                      lark_bot_url=cfg['lark_bot_url'],
                  )))
    if args.slurm:
        new_cfg['eval']['runner']['type'] = get_config_type(SlurmRunner)
        new_cfg['eval']['runner']['partition'] = args.partition
        new_cfg['eval']['runner']['quotatype'] = args.quotatype
        new_cfg['eval']['runner']['qos'] = args.qos
        new_cfg['eval']['runner']['retry'] = args.retry
    elif args.dlc:
        new_cfg['eval']['runner']['type'] = get_config_type(DLCRunner)
        new_cfg['eval']['runner']['aliyun_cfg'] = Config.fromfile(
            args.aliyun_cfg)
        new_cfg['eval']['runner']['retry'] = args.retry
    else:
        new_cfg['eval']['runner']['type'] = get_config_type(LocalRunner)
        new_cfg['eval']['runner'][
            'max_workers_per_gpu'] = args.max_workers_per_gpu
    cfg.merge_from_dict(new_cfg)
