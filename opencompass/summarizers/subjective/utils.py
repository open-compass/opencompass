# flake8: noqa: E501
import os.path as osp

import mmengine

from opencompass.utils import dataset_abbr_from_cfg


def get_outdir(cfg, time_str):
    """Get out put path.

    Args:
        cfg (ConfigDict): The running config.
        time_str (str): Current time.
    """
    work_dir = cfg['work_dir']
    output_path = osp.join(work_dir, 'summary', f'summary_{time_str}.txt')
    output_dir = osp.join(osp.split(output_path)[0], f'{time_str}')
    mmengine.mkdir_or_exist(output_dir)
    results_folder = osp.join(work_dir, 'results')
    return output_dir, results_folder


def get_judgeanswer_and_reference(dataset, subdir_path, post_process):
    """Extract judgements (scores) and references.

    Args:
        dataset (ConfigDict): Dataset config.
        subdir_path (str): Model path in results dir.
        post_process (function): The pre-defined extract function.
    """
    dataset_abbr = dataset_abbr_from_cfg(dataset)
    filename = osp.join(subdir_path, dataset_abbr + '.json')
    partial_filename = osp.join(subdir_path, dataset_abbr + '_0.json')
    if osp.exists(osp.realpath(filename)):
        result = mmengine.load(filename)
    elif osp.exists(osp.realpath(partial_filename)):
        filename = partial_filename
        result = {}
        i = 1
        partial_dict_flag = 0
        while osp.exists(osp.realpath(filename)):
            res = mmengine.load(filename)
            for k, v in res.items():
                result[partial_dict_flag] = v
                partial_dict_flag += 1
            filename = osp.join(subdir_path,
                                dataset_abbr + '_' + str(i) + '.json')
            i += 1
    else:
        result = {}

    if len(result) == 0:
        print('*' * 100)
        print('There are no results for ' + filename + ' or ' +
              partial_filename)
        print('*' * 100)

    judged_answers = []
    references = []
    for k, v in result.items():
        processed_judge = post_process(v['prediction'])
        if processed_judge is not None:
            judged_answers.append(processed_judge)
            references.append(v['gold'])
        # else:
        #     print(v['prediction'])
        #     print('-' * 128)
    if len(judged_answers) <= 0.95 * len(result):
        print('*' * 100)
        print(
            f'For your {filename} judge. Among {len(result)} judgements, successfully extracted {len(judged_answers)} judgements, please check!'
        )
        print('*' * 100)
    return judged_answers, references
