import json
import os
import os.path as osp
import re

from opencompass.utils.abbr import (dataset_abbr_from_cfg,
                                    deal_with_judge_model_abbr,
                                    model_abbr_from_cfg)


def save_to_station(cfg, args):

    if args.station_path is not None:
        station_path = args.station_path
    else:
        station_path = cfg.get('station_path')

    work_dict = cfg['work_dir']

    # objective dataset processing
    if 'judge_models' not in cfg.keys():
        model_list = [model_abbr_from_cfg(model) for model in cfg['models']]
        dataset_list = [
            dataset_abbr_from_cfg(dataset) for dataset in cfg['datasets']
        ]

        rs_exist_results = []
        if 'rs_exist_results' in cfg.keys():
            rs_exist_results = cfg['rs_exist_results']

        for dataset in dataset_list:
            result_path = osp.join(station_path, dataset)
            if not osp.exists(result_path):
                os.makedirs(result_path)

            for model in model_list:
                if ([model, dataset] in rs_exist_results
                        and not args.station_overwrite):
                    continue
                result_file_name = model + '.json'
                if osp.exists(osp.join(
                        result_path,
                        result_file_name)) and not args.station_overwrite:
                    print('result of {} with {} already exists'.format(
                        dataset, model))
                    continue
                else:
                    # get result dict
                    local_result_path = osp.join(work_dict, 'results', model)
                    local_result_json = osp.join(local_result_path,
                                                 dataset + '.json')
                    if not osp.exists(local_result_json):
                        if args.mode == 'viz':
                            continue
                        raise ValueError(
                            'invalid file: {}'.format(local_result_json))
                    with open(local_result_json, 'r') as f:
                        this_result = json.load(f)
                    f.close()

                    # get prediction list
                    local_prediction_path = osp.join(work_dict, 'predictions',
                                                     model)
                    local_prediction_regex = \
                        rf'^{re.escape(dataset)}(?:_\d+)?\.json$'
                    local_prediction_json = find_files_by_regex(
                        local_prediction_path, local_prediction_regex)
                    if not check_filenames(
                            dataset,
                            local_prediction_json) and args.mode != 'viz':
                        raise ValueError('invalid filelist: {}'.format(
                            local_prediction_json))

                    this_prediction = []
                    for prediction_json in local_prediction_json:
                        with open(
                                osp.join(local_prediction_path,
                                         prediction_json), 'r') as f:
                            this_prediction_load_json = json.load(f)
                        f.close()
                        for prekey in this_prediction_load_json.keys():
                            this_prediction.append(
                                this_prediction_load_json[prekey])

                    # get config dict
                    model_cfg = [
                        i for i in cfg['models']
                        if model_abbr_from_cfg(i) == model
                    ][0]
                    dataset_cfg = [
                        i for i in cfg['datasets']
                        if dataset_abbr_from_cfg(i) == dataset
                    ][0]
                    this_cfg = {'models': model_cfg, 'datasets': dataset_cfg}

                    # dict combine
                    data_model_results = {
                        'predictions': this_prediction,
                        'results': this_result,
                        'cfg': this_cfg
                    }
                    with open(osp.join(result_path, result_file_name),
                              'w') as f:
                        json.dump(data_model_results,
                                  f,
                                  ensure_ascii=False,
                                  indent=4)
                    f.close()
                    print(
                        'successfully save result of {} with {} to the station'
                        .format(dataset, model))
        return True

    # subjective processing
    else:
        model_list = [model for model in cfg['models']]
        judge_list = [judge_model for judge_model in cfg['judge_models']]
        model_pair_list = [[
            deal_with_judge_model_abbr(model, judge_model)
            for judge_model in judge_list
        ] for model in model_list]

        dataset_list = [[
            dataset_abbr_from_cfg(dataset),
            [dataset_abbr_from_cfg(base) for base in dataset['base_models']]
        ] if 'base_models' in dataset.keys() else
                        [dataset_abbr_from_cfg(dataset), ['']]
                        for dataset in cfg['datasets']]

        rs_exist_results = []
        if 'rs_exist_results' in cfg.keys():
            rs_exist_results = cfg['rs_exist_results']

        for pair_of_dataset_and_base in dataset_list:
            dataset, base_list = pair_of_dataset_and_base[
                0], pair_of_dataset_and_base[1]

            result_path = osp.join(station_path, dataset)
            if not osp.exists(result_path):
                os.makedirs(result_path)

            for base_model in base_list:
                base_model_name = base_model
                if base_model_name != '':
                    base_model_name += '_'
                for model_pair_sub_list in model_pair_list:
                    for model_pair in model_pair_sub_list:
                        model = model_abbr_from_cfg(model_pair[0])
                        model_result = model_abbr_from_cfg(model_pair)
                        if ([model, dataset] in rs_exist_results
                                and not args.station_overwrite):
                            continue
                        result_file_name = (base_model_name + model_result +
                                            '.json')
                        if osp.exists(osp.join(result_path, result_file_name)
                                      ) and not args.station_overwrite:
                            print('{} at {} already exists'.format(
                                result_file_name, result_path))
                            continue
                        else:
                            # get result dict
                            local_result_path = osp.join(
                                work_dict, 'results',
                                base_model_name + model_result)
                            local_result_json = osp.join(
                                local_result_path, dataset + '.json')
                            if not osp.exists(local_result_json):
                                if args.mode == 'viz':
                                    continue
                                raise ValueError('invalid file: {}'.format(
                                    local_result_json))
                            with open(local_result_json, 'r') as f:
                                this_result = json.load(f)
                            f.close()

                            # get prediction list
                            local_prediction_path = osp.join(
                                work_dict, 'predictions', model)
                            local_prediction_regex = \
                                rf'^{re.escape(dataset)}(?:_\d+)?\.json$'
                            local_prediction_json = find_files_by_regex(
                                local_prediction_path, local_prediction_regex)
                            if not check_filenames(dataset,
                                                   local_prediction_json
                                                   ) and args.mode != 'viz':
                                raise ValueError('invalid filelist: {}'.format(
                                    local_prediction_json))

                            this_prediction = []
                            for prediction_json in local_prediction_json:
                                with open(
                                        osp.join(local_prediction_path,
                                                 prediction_json), 'r') as f:
                                    this_prediction_load_json = json.load(f)
                                f.close()
                                for prekey in this_prediction_load_json.keys():
                                    this_prediction.append(
                                        this_prediction_load_json[prekey])

                            # get config dict
                            model_cfg = [
                                i for i in cfg['models']
                                if model_abbr_from_cfg(i) == model
                            ][0]
                            dataset_cfg = [
                                i for i in cfg['datasets']
                                if dataset_abbr_from_cfg(i) == dataset
                            ][0]
                            judge_model_cfg = [
                                i for i in cfg['judge_models']
                                if 'judged-by--' + model_abbr_from_cfg(i) ==
                                model_abbr_from_cfg(model_pair[1])
                            ][0]

                            this_cfg = {
                                'models': model_cfg,
                                'datasets': dataset_cfg,
                                'judge_models': judge_model_cfg
                            }

                            # dict combine
                            data_model_results = {
                                'predictions': this_prediction,
                                'results': this_result,
                                'cfg': this_cfg
                            }

                            with open(osp.join(result_path, result_file_name),
                                      'w') as f:
                                json.dump(data_model_results,
                                          f,
                                          ensure_ascii=False,
                                          indent=4)
                            f.close()
                            print('successfully save result: {} at {} to the'
                                  'station'.format(result_file_name,
                                                   result_path))
        return True


def read_from_station(cfg, args):

    assert args.station_path is not None or cfg.get('station_path') is not None
    if args.station_path is not None:
        station_path = args.station_path
    else:
        station_path = cfg.get('station_path')

    # objective check
    if 'judge_models' not in cfg.keys():
        model_list = [model_abbr_from_cfg(model) for model in cfg['models']]
        dataset_list = [
            dataset_abbr_from_cfg(dataset) for dataset in cfg['datasets']
        ]

        existing_results_list = []
        result_local_path = osp.join(cfg['work_dir'], 'results')
        if not osp.exists(result_local_path):
            os.makedirs(result_local_path)

        for dataset in dataset_list:
            for model in model_list:
                result_file_path = osp.join(station_path, dataset,
                                            model + '.json')
                if not osp.exists(result_file_path):
                    print('do not find result file: {} with {} at station'.
                          format(model, dataset))
                    continue
                else:
                    print('find result file: {} with {} at station'.format(
                        model, dataset))
                    with open(result_file_path, 'r') as f:
                        download_json = json.load(f)
                    f.close()
                    existing_results_list.append({
                        'combination': [model, dataset],
                        'file':
                        download_json
                    })

        # save results to local
        for i in existing_results_list:
            this_result = i['file']['results']
            this_result_local_path = osp.join(result_local_path,
                                              i['combination'][0])
            if not osp.exists(this_result_local_path):
                os.makedirs(this_result_local_path)
            this_result_local_file_path = osp.join(
                this_result_local_path, i['combination'][1] + '.json')
            if osp.exists(this_result_local_file_path):
                continue
            with open(this_result_local_file_path, 'w') as f:
                json.dump(this_result, f, ensure_ascii=False, indent=4)
            f.close()

        return existing_results_list

    # subjective check
    else:
        model_list = [model for model in cfg['models']]
        judge_list = [judge_model for judge_model in cfg['judge_models']]
        model_pair_list = [[
            deal_with_judge_model_abbr(model, judge_model)
            for judge_model in judge_list
        ] for model in model_list]

        dataset_list = [[
            dataset_abbr_from_cfg(dataset),
            [dataset_abbr_from_cfg(base) for base in dataset['base_models']]
        ] if 'base_models' in dataset.keys() else
                        [dataset_abbr_from_cfg(dataset), ['']]
                        for dataset in cfg['datasets']]

        existing_results_list = []
        result_local_path = osp.join(cfg['work_dir'], 'results')
        if not osp.exists(result_local_path):
            os.makedirs(result_local_path)

        for pair_of_dataset_and_base in dataset_list:
            dataset, base_list = pair_of_dataset_and_base[
                0], pair_of_dataset_and_base[1]

            for model_pair_sub_list in model_pair_list:
                result_file_path_list_origin = []
                for model_pair in model_pair_sub_list:
                    model_result = model_abbr_from_cfg(model_pair)
                    for base_model in base_list:
                        base_model_name = base_model
                        if base_model_name != '':
                            base_model_name += '_'

                        result_file_path_list_origin.append(
                            osp.join(station_path, dataset,
                                     base_model_name + model_result + '.json'))

                result_file_path_list = [
                    result_file_path
                    for result_file_path in result_file_path_list_origin
                    if osp.exists(result_file_path)
                ]
                model = model_abbr_from_cfg(model_pair_sub_list[0][0])

                # save all parts of results to local
                for result_file_path in result_file_path_list:
                    with open(result_file_path, 'r') as f:
                        this_result = json.load(f)['results']
                    f.close()
                    this_result_local_path = osp.join(
                        result_local_path,
                        osp.splitext(osp.basename(result_file_path))[0])
                    if not osp.exists(this_result_local_path):
                        os.makedirs(this_result_local_path)
                    this_result_local_file_path = osp.join(
                        this_result_local_path, dataset + '.json')
                    if osp.exists(this_result_local_file_path):
                        continue
                    with open(this_result_local_file_path, 'w') as f:
                        json.dump(this_result, f, ensure_ascii=False, indent=4)
                    f.close()

                # check whether complete
                if len(result_file_path_list) == len(
                        result_file_path_list_origin):
                    print('find complete results of {} with {} at station'.
                          format(model, dataset))
                    existing_results_list.append({
                        'combination': [model, dataset],
                        'file':
                        result_file_path_list
                    })
                else:
                    print('results of {} with {} at station is not complete'.
                          format(model, dataset))

        return existing_results_list


def find_files_by_regex(directory, pattern):

    regex = re.compile(pattern)

    matched_files = []
    for filename in os.listdir(directory):
        if regex.match(filename):
            matched_files.append(filename)

    return matched_files


def check_filenames(x, filenames):

    if not filenames:
        return False

    single_pattern = re.compile(rf'^{re.escape(x)}\.json$')
    numbered_pattern = re.compile(rf'^{re.escape(x)}_(\d+)\.json$')

    is_single = all(single_pattern.match(name) for name in filenames)
    is_numbered = all(numbered_pattern.match(name) for name in filenames)

    if not (is_single or is_numbered):
        return False

    if is_single:
        return len(filenames) == 1

    if is_numbered:
        numbers = []
        for name in filenames:
            match = numbered_pattern.match(name)
            if match:
                numbers.append(int(match.group(1)))

        if sorted(numbers) != list(range(len(numbers))):
            return False

    return True
