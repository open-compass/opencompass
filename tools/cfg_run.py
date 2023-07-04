import argparse
import getpass
import os
import os.path as osp
from datetime import datetime

from mmengine.config import Config

from opencompass.registry import PARTITIONERS, RUNNERS
from opencompass.runners import SlurmRunner
from opencompass.utils import LarkReporter, Summarizer, get_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Run an evaluation task')
    parser.add_argument('config', help='Train config file path')
    parser.add_argument('-p',
                        '--partition',
                        help='Slurm partition name',
                        default=None,
                        type=str)
    parser.add_argument('-q',
                        '--quotatype',
                        help='Slurm quota type',
                        default='auto',
                        type=str)
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
                        help='Work path, all the outputs will be '
                        'saved in this path, including the slurm logs, '
                        'the evaluation results, the summary results, etc.'
                        'If not specified, the work_dir will be set to '
                        './outputs/default.',
                        default=None,
                        type=str)
    parser.add_argument('-l',
                        '--lark',
                        help='Report the running status to lark bot',
                        action='store_true',
                        default=False)
    args = parser.parse_args()
    return args


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

    # infer
    if not args.lark:
        cfg['lark_bot_url'] = None
    elif cfg.get('lark_bot_url', None):
        content = f'{getpass.getuser()} 的新任务已启动！'
        LarkReporter(cfg['lark_bot_url']).post(content)

    if cfg.get('infer', None) is not None and args.mode in ['all', 'infer']:
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
        runner = RUNNERS.build(cfg.infer.runner)
        runner(tasks)

    # evaluate
    if cfg.get('eval', None) is not None and args.mode in ['all', 'eval']:
        if args.partition is not None:
            if RUNNERS.get(cfg.infer.runner.type) == SlurmRunner:
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
        runner = RUNNERS.build(cfg.eval.runner)
        runner(tasks)

    # visualize
    if args.mode in ['all', 'eval', 'viz']:
        summarizer = Summarizer(cfg)
        summarizer.summarize(time_str=cfg_time_str)


if __name__ == '__main__':
    main()
