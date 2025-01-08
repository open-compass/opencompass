# flake8: noqa
# yapf: disable
import argparse
import copy
import getpass
import os
import os.path as osp
from datetime import datetime

from mmengine.config import Config, DictAction

from opencompass.registry import PARTITIONERS, RUNNERS, build_from_cfg
from opencompass.runners import SlurmRunner
from opencompass.summarizers import DefaultSummarizer
from opencompass.utils import LarkReporter, get_logger
from opencompass.utils.run import (fill_eval_cfg, fill_infer_cfg,
                                   get_config_from_arg)


def parse_args():
    parser = argparse.ArgumentParser(description='Run an evaluation task')
    parser.add_argument('config', nargs='?', help='Train config file path')

    # add mutually exclusive args `--slurm` `--dlc`, defaults to local runner
    # if "infer" or "eval" not specified
    launch_method = parser.add_mutually_exclusive_group()
    launch_method.add_argument('--slurm',
                               action='store_true',
                               default=False,
                               help='Whether to force tasks to run with srun. '
                               'If True, `--partition(-p)` must be set. '
                               'Defaults to False')
    launch_method.add_argument('--dlc',
                               action='store_true',
                               default=False,
                               help='Whether to force tasks to run on dlc. If '
                               'True, `--aliyun-cfg` must be set. Defaults'
                               ' to False')
    # Add shortcut parameters (models, datasets and summarizer)
    parser.add_argument('--models', nargs='+', help='', default=None)
    parser.add_argument('--datasets', nargs='+', help='', default=None)
    parser.add_argument('--summarizer', help='', default=None)
    # add general args
    parser.add_argument('--debug',
                        help='Debug mode, in which scheduler will run tasks '
                        'in the single process, and output will not be '
                        'redirected to files',
                        action='store_true',
                        default=False)
    parser.add_argument('--dry-run',
                        help='Dry run mode, in which the scheduler will not '
                        'actually run the tasks, but only print the commands '
                        'to run',
                        action='store_true',
                        default=False)
    parser.add_argument(
        '-a', '--accelerator',
        help='Infer accelerator, support vllm and lmdeploy now.',
        choices=['vllm', 'lmdeploy', None],
        default=None,
        type=str)
    parser.add_argument('-m',
                        '--mode',
                        help='Running mode. You can choose "infer" if you '
                        'only want the inference results, or "eval" if you '
                        'already have the results and want to evaluate them, '
                        'or "viz" if you want to visualize the results.',
                        choices=['all', 'infer', 'eval', 'viz'],
                        default='all',
                        type=str)
    parser.add_argument('-r',
                        '--reuse',
                        nargs='?',
                        type=str,
                        const='latest',
                        help='Reuse previous outputs & results, and run any '
                        'missing jobs presented in the config. If its '
                        'argument is not specified, the latest results in '
                        'the work_dir will be reused. The argument should '
                        'also be a specific timestamp, e.g. 20230516_144254')
    parser.add_argument('-w',
                        '--work-dir',
                        help='Work path, all the outputs will be '
                        'saved in this path, including the slurm logs, '
                        'the evaluation results, the summary results, etc.'
                        'If not specified, the work_dir will be set to '
                        'outputs/default.',
                        default=None,
                        type=str)
    parser.add_argument(
        '--config-dir',
        default='configs',
        help='Use the custom config directory instead of config/ to '
        'search the configs for datasets, models and summarizers',
        type=str)
    parser.add_argument('-l',
                        '--lark',
                        help='Report the running status to lark bot',
                        action='store_true',
                        default=False)
    parser.add_argument('--max-num-workers',
                        help='Max number of workers to run in parallel. '
                        'Will be overrideen by the "max_num_workers" argument '
                        'in the config.',
                        type=int,
                        default=1)
    parser.add_argument('--max-workers-per-gpu',
                        help='Max task to run in parallel on one GPU. '
                        'It will only be used in the local runner.',
                        type=int,
                        default=1)
    parser.add_argument(
        '--retry',
        help='Number of retries if the job failed when using slurm or dlc. '
        'Will be overrideen by the "retry" argument in the config.',
        type=int,
        default=2)
    parser.add_argument(
        '--dump-eval-details',
        help='Whether to dump the evaluation details, including the '
        'correctness of each sample, bpb, etc.',
        action='store_true',
    )
    parser.add_argument(
        '--dump-extract-rate',
        help='Whether to dump the evaluation details, including the '
        'correctness of each sample, bpb, etc.',
        action='store_true',
    )
    # set srun args
    slurm_parser = parser.add_argument_group('slurm_args')
    parse_slurm_args(slurm_parser)
    # set dlc args
    dlc_parser = parser.add_argument_group('dlc_args')
    parse_dlc_args(dlc_parser)
    # set hf args
    hf_parser = parser.add_argument_group('hf_args')
    parse_hf_args(hf_parser)
    # set custom dataset args
    custom_dataset_parser = parser.add_argument_group('custom_dataset_args')
    parse_custom_dataset_args(custom_dataset_parser)
    args = parser.parse_args()
    if args.slurm:
        assert args.partition is not None, (
            '--partition(-p) must be set if you want to use slurm')
    if args.dlc:
        assert os.path.exists(args.aliyun_cfg), (
            'When launching tasks using dlc, it needs to be configured '
            'in "~/.aliyun.cfg", or use "--aliyun-cfg $ALiYun-CFG_Path"'
            ' to specify a new path.')
    return args


def parse_slurm_args(slurm_parser):
    """These args are all for slurm launch."""
    slurm_parser.add_argument('-p',
                              '--partition',
                              help='Slurm partition name',
                              default=None,
                              type=str)
    slurm_parser.add_argument('-q',
                              '--quotatype',
                              help='Slurm quota type',
                              default=None,
                              type=str)
    slurm_parser.add_argument('--qos',
                              help='Slurm quality of service',
                              default=None,
                              type=str)


def parse_dlc_args(dlc_parser):
    """These args are all for dlc launch."""
    dlc_parser.add_argument('--aliyun-cfg',
                            help='The config path for aliyun config',
                            default='~/.aliyun.cfg',
                            type=str)




def parse_hf_args(hf_parser):
    """These args are all for the quick construction of HuggingFace models."""
    hf_parser.add_argument('--hf-type', type=str, choices=['base', 'chat'], default='chat', help='The type of the HuggingFace model, base or chat')
    hf_parser.add_argument('--hf-path', type=str, help='The path to the HuggingFace model, e.g. "facebook/opt-125m", required')
    hf_parser.add_argument('--model-kwargs', nargs='+', action=DictAction, default={}, help='The kwargs for the HuggingFace model')
    hf_parser.add_argument('--tokenizer-path', type=str, help='The path to the HuggingFace tokenizer, same as --hf-path if not specified')
    hf_parser.add_argument('--tokenizer-kwargs', nargs='+', action=DictAction, default={}, help='The kwargs for the tokenizer')
    hf_parser.add_argument('--peft-path', type=str, help='The path to the PEFT model')
    hf_parser.add_argument('--peft-kwargs', nargs='+', action=DictAction, default={}, help='The kwargs for the PEFT model')
    hf_parser.add_argument('--generation-kwargs', nargs='+', action=DictAction, default={}, help='The kwargs for the generation')
    hf_parser.add_argument('--max-seq-len', type=int, help='The max sequence length for the HuggingFace model')
    hf_parser.add_argument('--max-out-len', type=int, default=256, help='The max output length for the HuggingFace model')
    hf_parser.add_argument('--min-out-len', type=int, default=1, help='The min output length for the HuggingFace model')
    hf_parser.add_argument('--batch-size', type=int, default=8, help='The batch size for the HuggingFace model')
    hf_parser.add_argument('--num-gpus', type=int, default=None, help='Deprecated, please use --hf-num-gpus instead')
    hf_parser.add_argument('--hf-num-gpus', type=int, default=1, help='The number of GPUs for the HuggingFace model passed via cli')
    hf_parser.add_argument('--pad-token-id', type=int, help='The pad token id for the HuggingFace model')
    hf_parser.add_argument('--stop-words', nargs='+', default=[], help='The stop words for the HuggingFace model')


def parse_custom_dataset_args(custom_dataset_parser):
    """These args are all for the quick construction of custom datasets."""
    custom_dataset_parser.add_argument('--custom-dataset-path', type=str)
    custom_dataset_parser.add_argument('--custom-dataset-meta-path', type=str)
    custom_dataset_parser.add_argument('--custom-dataset-data-type',
                                       type=str,
                                       choices=['mcq', 'qa'])
    custom_dataset_parser.add_argument('--custom-dataset-infer-method',
                                       type=str,
                                       choices=['gen', 'ppl'])


def main():
    args = parse_args()

    if args.num_gpus is not None:
        raise ValueError('The `--num-gpus` argument is deprecated, please use '
                         '`--hf-num-gpus` to describe number of gpus used for '
                         'the HuggingFace model instead.')

    if args.dry_run:
        args.debug = True
    # initialize logger
    logger = get_logger(log_level='DEBUG' if args.debug else 'INFO')

    cfg = get_config_from_arg(args)
    if args.work_dir is not None:
        cfg['work_dir'] = args.work_dir
    else:
        cfg.setdefault('work_dir', os.path.join('outputs', 'default'))

    # cfg_time_str defaults to the current time
    cfg_time_str = dir_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
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
    elif args.mode in ['eval', 'viz']:
        raise ValueError('You must specify -r or --reuse when running in eval '
                         'or viz mode!')

    # update "actual" work_dir
    cfg['work_dir'] = osp.join(cfg.work_dir, dir_time_str)
    current_workdir = cfg['work_dir']
    logger.info(f'Current exp folder: {current_workdir}')

    os.makedirs(osp.join(cfg.work_dir, 'configs'), exist_ok=True)

    # dump config
    output_config_path = osp.join(cfg.work_dir, 'configs',
                                  f'{cfg_time_str}_{os.getpid()}.py')
    cfg.dump(output_config_path)
    # Config is intentally reloaded here to avoid initialized
    # types cannot be serialized
    cfg = Config.fromfile(output_config_path, format_python_code=False)

    # report to lark bot if specify --lark
    if not args.lark:
        cfg['lark_bot_url'] = None
    elif cfg.get('lark_bot_url', None):
        content = f'{getpass.getuser()}\'s task has been launched!'
        LarkReporter(cfg['lark_bot_url']).post(content)

    if args.mode in ['all', 'infer']:
        # When user have specified --slurm or --dlc, or have not set
        # "infer" in config, we will provide a default configuration
        # for infer
        if (args.dlc or args.slurm) and cfg.get('infer', None):
            logger.warning('You have set "infer" in the config, but '
                           'also specified --slurm or --dlc. '
                           'The "infer" configuration will be overridden by '
                           'your runtime arguments.')

        if args.dlc or args.slurm or cfg.get('infer', None) is None:
            fill_infer_cfg(cfg, args)

        if args.partition is not None:
            if RUNNERS.get(cfg.infer.runner.type) == SlurmRunner:
                cfg.infer.runner.partition = args.partition
                cfg.infer.runner.quotatype = args.quotatype
        else:
            logger.warning('SlurmRunner is not used, so the partition '
                           'argument is ignored.')
        if args.debug:
            cfg.infer.runner.debug = True
        if args.lark:
            cfg.infer.runner.lark_bot_url = cfg['lark_bot_url']
        cfg.infer.partitioner['out_dir'] = osp.join(cfg['work_dir'],
                                                    'predictions/')
        partitioner = PARTITIONERS.build(cfg.infer.partitioner)
        tasks = partitioner(cfg)
        if args.dry_run:
            return
        runner = RUNNERS.build(cfg.infer.runner)
        # Add extra attack config if exists
        if hasattr(cfg, 'attack'):
            for task in tasks:
                cfg.attack.dataset = task.datasets[0][0].abbr
                task.attack = cfg.attack
        runner(tasks)

    # evaluate
    if args.mode in ['all', 'eval']:
        # When user have specified --slurm or --dlc, or have not set
        # "eval" in config, we will provide a default configuration
        # for eval
        if (args.dlc or args.slurm) and cfg.get('eval', None):
            logger.warning('You have set "eval" in the config, but '
                           'also specified --slurm or --dlc. '
                           'The "eval" configuration will be overridden by '
                           'your runtime arguments.')

        if args.dlc or args.slurm or cfg.get('eval', None) is None:
            fill_eval_cfg(cfg, args)
        if args.dump_eval_details:
            cfg.eval.runner.task.dump_details = True
        if args.dump_extract_rate:
            cfg.eval.runner.task.cal_extract_rate = True
        if args.partition is not None:
            if RUNNERS.get(cfg.eval.runner.type) == SlurmRunner:
                cfg.eval.runner.partition = args.partition
                cfg.eval.runner.quotatype = args.quotatype
            else:
                logger.warning('SlurmRunner is not used, so the partition '
                               'argument is ignored.')
        if args.debug:
            cfg.eval.runner.debug = True
        if args.lark:
            cfg.eval.runner.lark_bot_url = cfg['lark_bot_url']
        cfg.eval.partitioner['out_dir'] = osp.join(cfg['work_dir'], 'results/')
        partitioner = PARTITIONERS.build(cfg.eval.partitioner)
        tasks = partitioner(cfg)
        if args.dry_run:
            return
        runner = RUNNERS.build(cfg.eval.runner)

        # For meta-review-judge in subjective evaluation
        if isinstance(tasks, list) and len(tasks) != 0 and isinstance(
                tasks[0], list):
            for task_part in tasks:
                runner(task_part)
        else:
            runner(tasks)

    # visualize
    if args.mode in ['all', 'eval', 'viz']:
        summarizer_cfg = cfg.get('summarizer', {})

        # For subjective summarizer
        if summarizer_cfg.get('function', None):
            main_summarizer_cfg = copy.deepcopy(summarizer_cfg)
            grouped_datasets = {}
            for dataset in cfg.datasets:
                prefix = dataset['abbr'].split('_')[0]
                if prefix not in grouped_datasets:
                    grouped_datasets[prefix] = []
                grouped_datasets[prefix].append(dataset)
            all_grouped_lists = []
            for prefix in grouped_datasets:
                all_grouped_lists.append(grouped_datasets[prefix])
            dataset_score_container = []
            for dataset in all_grouped_lists:
                temp_cfg = copy.deepcopy(cfg)
                temp_cfg.datasets = dataset
                summarizer_cfg = dict(type=dataset[0]['summarizer']['type'], config=temp_cfg)
                summarizer = build_from_cfg(summarizer_cfg)
                dataset_score = summarizer.summarize(time_str=cfg_time_str)
                if dataset_score:
                    dataset_score_container.append(dataset_score)
            main_summarizer_cfg['config'] = cfg
            main_summarizer = build_from_cfg(main_summarizer_cfg)
            main_summarizer.summarize(time_str=cfg_time_str, subjective_scores=dataset_score_container)
        else:
            if not summarizer_cfg or summarizer_cfg.get('type', None) is None:
                summarizer_cfg['type'] = DefaultSummarizer
            summarizer_cfg['config'] = cfg
            summarizer = build_from_cfg(summarizer_cfg)
            summarizer.summarize(time_str=cfg_time_str)



if __name__ == '__main__':
    main()
