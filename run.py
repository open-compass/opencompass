import argparse
import getpass
import os
import os.path as osp
from datetime import datetime

from mmengine.config import Config

from opencompass.partitioners import NaivePartitioner, SizePartitioner
from opencompass.runners import DLCRunner, LocalRunner, SlurmRunner
from opencompass.utils import LarkReporter, Summarizer, get_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Run an evaluation task')
    parser.add_argument('config', help='Train config file path')
    # add mutually exclusive args `--slurm` `--dlc`, default to local runner
    luach_method = parser.add_mutually_exclusive_group()
    luach_method.add_argument('--slurm',
                              action='store_true',
                              default=False,
                              help='Whether to use srun to launch tasks, if '
                              'True, `--partition(-p)` must be set. Defaults'
                              ' to False')
    luach_method.add_argument('--dlc',
                              action='store_true',
                              default=False,
                              help='Whether to use dlc to launch tasks, if '
                              'True, `--aliyun-cfg` must be set. Defaults'
                              ' to False')
    # add general args
    parser.add_argument('--debug',
                        help='Debug mode, in which scheduler will run tasks '
                        'in the single process, and output will not be '
                        'redirected to files',
                        action='store_true',
                        default=False)
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
                        'also be a specific timestamp, e.g. 20230516_144254'),
    parser.add_argument('-w',
                        '--work-dir',
                        help='Work path, all the outputs will be saved in '
                        'this path, including the slurm logs, the evaluation'
                        ' results, the summary results, etc. If not specified,'
                        ' the work_dir will be set to None',
                        default=None,
                        type=str)
    parser.add_argument('-l',
                        '--lark',
                        help='Report the running status to lark bot',
                        action='store_true',
                        default=False)
    parser.add_argument('--max-partition-size',
                        help='The maximum size of a task.',
                        type=int,
                        default=2000),
    parser.add_argument(
        '--gen-task-coef',
        help='The dataset cost measurement coefficient for generation tasks',
        type=int,
        default=20)
    parser.add_argument('--max-num-workers',
                        help='Max number of workers to run in parallel.',
                        type=int,
                        default=32)
    parser.add_argument(
        '--retry',
        help='Number of retries if the job failed when using slurm or dlc.',
        type=int,
        default=2)
    # set srun args
    slurm_parser = parser.add_argument_group('slurm_args')
    parse_slurm_args(slurm_parser)
    # set dlc args
    dlc_parser = parser.add_argument_group('dlc_args')
    parse_dlc_args(dlc_parser)
    args = parser.parse_args()
    if args.slurm:
        assert args.partition is not None, (
            '--partition(-p) must be set if you want to use slurm')
    if args.dlc:
        assert os.path.exists(args.aliyun_cfg), (
            'When luaching tasks using dlc, it needs to be configured'
            'in "~/.aliyun.cfg", or use "--aliyun-cfg $ALiYun-CFG_Path"'
            ' to specify a new path.')
    return args


def parse_slurm_args(slurm_parser):
    """these args are all for slurm launch."""
    slurm_parser.add_argument('-p',
                              '--partition',
                              help='Slurm partition name',
                              default=None,
                              type=str)
    slurm_parser.add_argument('-q',
                              '--quotatype',
                              help='Slurm quota type',
                              default='auto',
                              type=str)


def parse_dlc_args(dlc_parser):
    """these args are all for dlc launch."""
    dlc_parser.add_argument('--aliyun-cfg',
                            help='The config path for aliyun config',
                            default='~/.aliyun.cfg',
                            type=str)


def main():
    args = parse_args()

    # initialize logger
    logger = get_logger(log_level='DEBUG' if args.debug else 'INFO')

    cfg = Config.fromfile(args.config)
    if args.work_dir is not None:
        cfg['work_dir'] = args.work_dir
    else:
        cfg.setdefault('work_dir', './outputs/default/')

    # cfg_time_str defaults to the current time
    cfg_time_str = dir_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.reuse:
        if args.reuse == 'latest':
            dirs = os.listdir(cfg.work_dir)
            assert len(dirs) > 0, 'No previous results to reuse!'
            dir_time_str = sorted(dirs)[-1]
        else:
            dir_time_str = args.reuse
        logger.info(f'Reusing experiements from {dir_time_str}')
    elif args.mode in ['eval', 'viz']:
        raise ValueError('You must specify -r or --reuse when running in eval '
                         'or viz mode!')

    # update "actual" work_dir
    cfg['work_dir'] = osp.join(cfg.work_dir, dir_time_str)
    os.makedirs(osp.join(cfg.work_dir, 'configs'), exist_ok=True)

    # dump config
    output_config_path = osp.join(cfg.work_dir, 'configs',
                                  f'{cfg_time_str}.py')
    cfg.dump(output_config_path)
    # Config is intentally reloaded here to avoid initialized
    # types cannot be serialized
    cfg = Config.fromfile(output_config_path)

    # report to lark bot if specify --lark
    if not args.lark:
        cfg['lark_bot_url'] = None
    elif cfg.get('lark_bot_url', None):
        content = f'{getpass.getuser()}\'s task has been launched!'
        LarkReporter(cfg['lark_bot_url']).post(content)

    if args.mode in ['all', 'infer']:
        # Use SizePartitioner to split into subtasks
        partitioner = SizePartitioner(osp.join(cfg['work_dir'],
                                               'predictions/'),
                                      max_task_size=args.max_partition_size,
                                      gen_task_coef=args.gen_task_coef)
        tasks = partitioner(cfg)
        # execute the infer subtasks
        exec_infer_runner(tasks, args, cfg)

    # evaluate
    if args.mode in ['all', 'eval']:
        # Use NaivePartitioner，not split
        partitioner = NaivePartitioner(osp.join(cfg['work_dir'], 'results/'))
        tasks = partitioner(cfg)
        # execute the eval tasks
        exec_eval_runner(tasks, args, cfg)

    # visualize
    if args.mode in ['all', 'eval', 'viz']:
        summarizer = Summarizer(cfg)
        summarizer.summarize(time_str=cfg_time_str)


def exec_infer_runner(tasks, args, cfg):
    """execute infer runner according to args."""
    if args.slurm:
        runner = SlurmRunner(dict(type='OpenICLInferTask'),
                             max_num_workers=args.max_num_workers,
                             partition=args.partition,
                             quotatype=args.quotatype,
                             retry=args.retry,
                             debug=args.debug,
                             lark_bot_url=cfg['lark_bot_url'])
    elif args.dlc:
        runner = DLCRunner(dict(type='OpenICLInferTask'),
                           max_num_workers=args.max_num_workers,
                           aliyun_cfg=Config.fromfile(args.aliyun_cfg),
                           retry=args.retry,
                           debug=args.debug,
                           lark_bot_url=cfg['lark_bot_url'])
    else:
        runner = LocalRunner(
            task=dict(type='OpenICLInferTask'),
            # max_num_workers = args.max_num_workers,
            debug=args.debug,
            lark_bot_url=cfg['lark_bot_url'])
    runner(tasks)


def exec_eval_runner(tasks, args, cfg):
    """execute infer runner according to args."""
    if args.slurm:
        runner = SlurmRunner(dict(type='OpenICLEvalTask'),
                             max_num_workers=args.max_num_workers,
                             partition=args.partition,
                             quotatype=args.quotatype,
                             retry=args.retry,
                             debug=args.debug,
                             lark_bot_url=cfg['lark_bot_url'])
    elif args.dlc:
        runner = DLCRunner(dict(type='OpenICLEvalTask'),
                           max_num_workers=args.max_num_workers,
                           aliyun_cfg=Config.fromfile(args.aliyun_cfg),
                           retry=args.retry,
                           debug=args.debug,
                           lark_bot_url=cfg['lark_bot_url'])
    else:
        runner = LocalRunner(
            task=dict(type='OpenICLEvalTask'),
            # max_num_workers = args.max_num_workers,
            debug=args.debug,
            lark_bot_url=cfg['lark_bot_url'])
    runner(tasks)


if __name__ == '__main__':
    main()
