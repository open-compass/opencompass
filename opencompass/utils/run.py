# flake8: noqa
# yapf: disable
import os
from typing import List, Tuple, Union

import tabulate
from mmengine.config import Config

from opencompass.datasets.custom import make_custom_dataset_config
from opencompass.models import (VLLM, HuggingFace, HuggingFaceBaseModel,
                                HuggingFaceCausalLM, HuggingFaceChatGLM3,
                                HuggingFacewithChatTemplate, TurboMindModel,
                                TurboMindModelwithChatTemplate,
                                VLLMwithChatTemplate)
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import DLCRunner, LocalRunner, SlurmRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask
from opencompass.utils import get_logger, match_files


def match_cfg_file(workdir: Union[str, List[str]],
                   pattern: Union[str, List[str]]) -> List[Tuple[str, str]]:
    """Match the config file in workdir recursively given the pattern.

    Additionally, if the pattern itself points to an existing file, it will be
    directly returned.
    """
    def _mf_with_multi_workdirs(workdir, pattern, fuzzy=False):
        if isinstance(workdir, str):
            workdir = [workdir]
        files = []
        for wd in workdir:
            files += match_files(wd, pattern, fuzzy=fuzzy)
        return files

    if isinstance(pattern, str):
        pattern = [pattern]
    pattern = [p + '.py' if not p.endswith('.py') else p for p in pattern]
    files = _mf_with_multi_workdirs(workdir, pattern, fuzzy=False)
    if len(files) != len(pattern):
        nomatched = []
        ambiguous = []
        err_msg = ('The provided pattern matches 0 or more than one '
                   'config. Please verify your pattern and try again. '
                   'You may use tools/list_configs.py to list or '
                   'locate the configurations.\n')
        for p in pattern:
            files = _mf_with_multi_workdirs(workdir, p, fuzzy=False)
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


def try_fill_in_custom_cfgs(config):
    for i, dataset in enumerate(config['datasets']):
        if 'type' not in dataset:
            config['datasets'][i] = make_custom_dataset_config(dataset)
    if 'model_dataset_combinations' not in config:
        return config
    for mdc in config['model_dataset_combinations']:
        for i, dataset in enumerate(mdc['datasets']):
            if 'type' not in dataset:
                mdc['datasets'][i] = make_custom_dataset_config(dataset)
    return config


def get_config_from_arg(args) -> Config:
    """Get the config object given args.

    Only a few argument combinations are accepted (priority from high to low)
    1. args.config
    2. args.models and args.datasets
    3. Huggingface parameter groups and args.datasets
    """
    logger = get_logger()
    if args.config:
        config = Config.fromfile(args.config, format_python_code=False)
        config = try_fill_in_custom_cfgs(config)
        # set infer accelerator if needed
        if args.accelerator in ['vllm', 'lmdeploy']:
            config['models'] = change_accelerator(config['models'], args.accelerator)
            if config.get('eval', {}).get('partitioner', {}).get('models') is not None:
                config['eval']['partitioner']['models'] = change_accelerator(config['eval']['partitioner']['models'], args.accelerator)
            if config.get('eval', {}).get('partitioner', {}).get('base_models') is not None:
                config['eval']['partitioner']['base_models'] = change_accelerator(config['eval']['partitioner']['base_models'], args.accelerator)
            if config.get('eval', {}).get('partitioner', {}).get('compare_models') is not None:
                config['eval']['partitioner']['compare_models'] = change_accelerator(config['eval']['partitioner']['compare_models'], args.accelerator)
            if config.get('eval', {}).get('partitioner', {}).get('judge_models') is not None:
                config['eval']['partitioner']['judge_models'] = change_accelerator(config['eval']['partitioner']['judge_models'], args.accelerator)
            if config.get('judge_models') is not None:
                config['judge_models'] = change_accelerator(config['judge_models'], args.accelerator)
        return config

    # parse dataset args
    if not args.datasets and not args.custom_dataset_path:
        raise ValueError('You must specify "--datasets" or "--custom-dataset-path" if you do not specify a config file path.')
    datasets = []
    if args.datasets:
        datasets_dir = [
            os.path.join(args.config_dir, 'datasets'),
            os.path.join(args.config_dir, 'dataset_collections')
        ]
        for dataset_arg in args.datasets:
            if '/' in dataset_arg:
                dataset_name, dataset_suffix = dataset_arg.split('/', 1)
                dataset_key_suffix = dataset_suffix
            else:
                dataset_name = dataset_arg
                dataset_key_suffix = '_datasets'

            for dataset in match_cfg_file(datasets_dir, [dataset_name]):
                logger.info(f'Loading {dataset[0]}: {dataset[1]}')
                cfg = Config.fromfile(dataset[1])
                for k in cfg.keys():
                    if k.endswith(dataset_key_suffix):
                        datasets += cfg[k]
    else:
        dataset = {'path': args.custom_dataset_path}
        if args.custom_dataset_infer_method is not None:
            dataset['infer_method'] = args.custom_dataset_infer_method
        if args.custom_dataset_data_type is not None:
            dataset['data_type'] = args.custom_dataset_data_type
        if args.custom_dataset_meta_path is not None:
            dataset['meta_path'] = args.custom_dataset_meta_path
        dataset = make_custom_dataset_config(dataset)
        datasets.append(dataset)

    # parse model args
    if not args.models and not args.hf_path:
        raise ValueError('You must specify a config file path, or specify --models and --datasets, or specify HuggingFace model parameters and --datasets.')
    models = []
    if args.models:
        model_dir = os.path.join(args.config_dir, 'models')
        for model in match_cfg_file(model_dir, args.models):
            logger.info(f'Loading {model[0]}: {model[1]}')
            cfg = Config.fromfile(model[1])
            if 'models' not in cfg:
                raise ValueError(f'Config file {model[1]} does not contain "models" field')
            models += cfg['models']
    else:
        if args.hf_type == 'chat':
            mod = HuggingFacewithChatTemplate
        else:
            mod = HuggingFaceBaseModel
        model = dict(type=f'{mod.__module__}.{mod.__name__}',
                     abbr=args.hf_path.split('/')[-1] + '_hf',
                     path=args.hf_path,
                     model_kwargs=args.model_kwargs,
                     tokenizer_path=args.tokenizer_path,
                     tokenizer_kwargs=args.tokenizer_kwargs,
                     generation_kwargs=args.generation_kwargs,
                     peft_path=args.peft_path,
                     peft_kwargs=args.peft_kwargs,
                     max_seq_len=args.max_seq_len,
                     max_out_len=args.max_out_len,
                     batch_size=args.batch_size,
                     pad_token_id=args.pad_token_id,
                     stop_words=args.stop_words,
                     run_cfg=dict(num_gpus=args.hf_num_gpus))
        logger.debug(f'Using model: {model}')
        models.append(model)
    # set infer accelerator if needed
    if args.accelerator in ['vllm', 'lmdeploy']:
        models = change_accelerator(models, args.accelerator)
    # parse summarizer args
    summarizer_arg = args.summarizer if args.summarizer is not None else 'example'
    summarizers_dir = os.path.join(args.config_dir, 'summarizers')

    # Check if summarizer_arg contains '/'
    if '/' in summarizer_arg:
        # If it contains '/', split the string by '/'
        # and use the second part as the configuration key
        summarizer_file, summarizer_key = summarizer_arg.split('/', 1)
    else:
        # If it does not contain '/', keep the original logic unchanged
        summarizer_key = 'summarizer'
        summarizer_file = summarizer_arg

    s = match_cfg_file(summarizers_dir, [summarizer_file])[0]
    logger.info(f'Loading {s[0]}: {s[1]}')
    cfg = Config.fromfile(s[1])
    # Use summarizer_key to retrieve the summarizer definition
    # from the configuration file
    summarizer = cfg[summarizer_key]

    return Config(dict(models=models, datasets=datasets, summarizer=summarizer), format_python_code=False)


def change_accelerator(models, accelerator):
    models = models.copy()
    logger = get_logger()
    model_accels = []
    for model in models:
        logger.info(f'Transforming {model["abbr"]} to {accelerator}')
        # change HuggingFace model to VLLM or TurboMindModel
        if model['type'] in [HuggingFace, HuggingFaceCausalLM, HuggingFaceChatGLM3]:
            gen_args = dict()
            if model.get('generation_kwargs') is not None:
                generation_kwargs = model['generation_kwargs'].copy()
                gen_args['temperature'] = generation_kwargs.get('temperature', 0.001)
                gen_args['top_k'] = generation_kwargs.get('top_k', 1)
                gen_args['top_p'] = generation_kwargs.get('top_p', 0.9)
                gen_args['stop_token_ids'] = generation_kwargs.get('eos_token_id', None)
                generation_kwargs['stop_token_ids'] = generation_kwargs.get('eos_token_id', None)
                generation_kwargs.pop('eos_token_id')
            else:
                # if generation_kwargs is not provided, set default values
                generation_kwargs = dict()
                gen_args['temperature'] = 0.0
                gen_args['top_k'] = 1
                gen_args['top_p'] = 0.9
                gen_args['stop_token_ids'] = None

            if accelerator == 'lmdeploy':
                logger.info(f'Transforming {model["abbr"]} to {accelerator}')
                mod = TurboMindModel
                acc_model = dict(
                    type=f'{mod.__module__}.{mod.__name__}',
                    abbr=model['abbr'].replace('hf', 'turbomind') if '-hf' in model['abbr'] else model['abbr'] + '-turbomind',
                    path=model['path'],
                    engine_config=dict(session_len=model['max_seq_len'],
                                       max_batch_size=model['batch_size'],
                                       tp=model['run_cfg']['num_gpus']),
                    gen_config=dict(top_k=gen_args['top_k'],
                                    temperature=gen_args['temperature'],
                                    top_p=gen_args['top_p'],
                                    max_new_tokens=model['max_out_len'],
                                    stop_words=gen_args['stop_token_ids']),
                    max_out_len=model['max_out_len'],
                    max_seq_len=model['max_seq_len'],
                    batch_size=model['batch_size'],
                    concurrency=model['batch_size'],
                    run_cfg=model['run_cfg'],
                )
                for item in ['meta_template']:
                    if model.get(item) is not None:
                        acc_model[item] = model[item]
            elif accelerator == 'vllm':
                logger.info(f'Transforming {model["abbr"]} to {accelerator}')

                acc_model = dict(
                    type=f'{VLLM.__module__}.{VLLM.__name__}',
                    abbr=model['abbr'].replace('hf', 'vllm') if '-hf' in model['abbr'] else model['abbr'] + '-vllm',
                    path=model['path'],
                    model_kwargs=dict(tensor_parallel_size=model['run_cfg']['num_gpus']),
                    max_out_len=model['max_out_len'],
                    max_seq_len=model['max_seq_len'],
                    batch_size=model['batch_size'],
                    generation_kwargs=generation_kwargs,
                    run_cfg=model['run_cfg'],
                )
                for item in ['meta_template', 'end_str']:
                    if model.get(item) is not None:
                        acc_model[item] = model[item]
            else:
                raise ValueError(f'Unsupported accelerator {accelerator} for model type {model["type"]}')
        elif model['type'] in [HuggingFacewithChatTemplate]:
            if accelerator == 'vllm':
                mod = VLLMwithChatTemplate
                acc_model = dict(
                    type=f'{mod.__module__}.{mod.__name__}',
                    abbr=model['abbr'].replace('hf', 'vllm') if '-hf' in model['abbr'] else model['abbr'] + '-vllm',
                    path=model['path'],
                    model_kwargs=dict(tensor_parallel_size=model['run_cfg']['num_gpus']),
                    max_out_len=model['max_out_len'],
                    batch_size=16,
                    run_cfg=model['run_cfg'],
                    stop_words=model.get('stop_words', []),
                )
            elif accelerator == 'lmdeploy':
                mod = TurboMindModelwithChatTemplate
                acc_model = dict(
                    type=f'{mod.__module__}.{mod.__name__}',
                    abbr=model['abbr'].replace('hf', 'turbomind') if '-hf' in model['abbr'] else model['abbr'] + '-turbomind',
                    path=model['path'],
                    engine_config=dict(max_batch_size=model.get('batch_size', 16), tp=model['run_cfg']['num_gpus']),
                    gen_config=dict(top_k=1, temperature=1e-6, top_p=0.9),
                    max_seq_len=model.get('max_seq_len', 2048),
                    max_out_len=model['max_out_len'],
                    batch_size=16,
                    run_cfg=model['run_cfg'],
                    stop_words=model.get('stop_words', []),
                )
            else:
                raise ValueError(f'Unsupported accelerator {accelerator} for model type {model["type"]}')
        else:
            acc_model = model
            logger.warning(f'Unsupported model type {model["type"]}, will keep the original model.')
        model_accels.append(acc_model)
    return model_accels


def get_config_type(obj) -> str:
    return f'{obj.__module__}.{obj.__name__}'


def fill_infer_cfg(cfg, args):
    new_cfg = dict(infer=dict(
        partitioner=dict(type=get_config_type(NumWorkerPartitioner),
                         num_worker=args.max_num_workers),
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
