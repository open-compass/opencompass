import json
import os
import os.path as osp
import re


def Save_To_Station(cfg, args):

    assert args.station_path is not None or 'station_path' in cfg.keys(
    ) and cfg['station_path'] is not None
    if 'station_path' in cfg.keys() and cfg['station_path'] is not None:
        station_path = cfg['station_path']
    else:
        station_path = args.station_path

    work_dict = cfg['work_dir']
    model_list = [i['abbr'] for i in cfg['models']]
    dataset_list = [i['abbr'] for i in cfg['datasets']]

    for dataset in dataset_list:
        result_path = osp.join(station_path, dataset)
        if not osp.exists(result_path):
            os.makedirs(result_path)

        for model in model_list:
            result_file_name = model + '.json'
            if osp.exists(osp.join(result_path, result_file_name)):
                print('result of {} with {} already exists'.format(
                    dataset, model))
                continue
            else:

                # get result dict
                local_result_path = work_dict + '/results/' + model + '/'
                local_result_json = local_result_path + dataset + '.json'
                if not osp.exists(local_result_json):
                    raise ValueError(
                        'invalid file: {}'.format(local_result_json))
                with open(local_result_json, 'r') as f:
                    this_result = json.load(f)
                f.close()

                # get prediction list
                local_prediction_path = (work_dict + '/predictions/' + model +
                                         '/')
                local_prediction_regex = \
                    rf'^{re.escape(dataset)}(?:_\d+)?\.json$'
                local_prediction_json = find_files_by_regex(
                    local_prediction_path, local_prediction_regex)
                if not check_filenames(dataset, local_prediction_json):
                    raise ValueError(
                        'invalid filelist: {}'.format(local_prediction_json))

                this_prediction = []
                for prediction_json in local_prediction_json:
                    with open(local_prediction_path + prediction_json,
                              'r') as f:
                        this_prediction_load_json = json.load(f)
                    f.close()
                    for prekey in this_prediction_load_json.keys():
                        this_prediction.append(
                            this_prediction_load_json[prekey])

                # dict combine
                data_model_results = {
                    'predictions': this_prediction,
                    'results': this_result
                }
                with open(osp.join(result_path, result_file_name), 'w') as f:
                    json.dump(data_model_results,
                              f,
                              ensure_ascii=False,
                              indent=4)
                f.close()
                print('result of {} with {} already exists'.format(
                    dataset, model))

    return True


def Read_From_Station(cfg, args, dir_time_str):

    assert args.station_path is not None or 'station_path' in cfg.keys(
    ) and cfg['station_path'] is not None
    if 'station_path' in cfg.keys() and cfg['station_path'] is not None:
        station_path = cfg['station_path']
    else:
        station_path = args.station_path

    work_dict = osp.join(cfg.work_dir, dir_time_str)
    model_list = [i['abbr'] for i in cfg['models']]
    dataset_list = [i['abbr'] for i in cfg['datasets']]

    if not osp.exists(work_dict):
        os.makedirs(work_dict)
    local_prediction_path = osp.join(work_dict, 'predictions')
    if not osp.exists(local_prediction_path):
        os.makedirs(local_prediction_path)
    local_result_path = osp.join(work_dict, 'results')
    if not osp.exists(local_result_path):
        os.makedirs(local_result_path)

    for model in model_list:

        for data in dataset_list:
            result_file_path = osp.join(station_path, data, model + '.json')
            if not osp.exists(result_file_path):
                print('do not find result file: {} with {} at station'.format(
                    model, data))
                continue
            else:
                print('find result file: {} with {} at station'.format(
                    model, data))

                with open(result_file_path, 'r') as f:
                    download_json = json.load(f)
                f.close()

                this_local_prediction_path = osp.join(local_prediction_path,
                                                      model)
                if not osp.exists(this_local_prediction_path):
                    os.makedirs(this_local_prediction_path)
                this_local_result_path = osp.join(local_result_path, model)
                if not osp.exists(this_local_result_path):
                    os.makedirs(this_local_result_path)

                this_local_prediction_path = osp.join(
                    this_local_prediction_path, data + '.json')
                this_local_result_path = osp.join(this_local_result_path,
                                                  data + '.json')

                download_json_prediction = download_json['predictions']
                download_json_result = download_json['results']

                # save predictions
                local_prediction = {}
                for i in range(len(download_json_prediction)):
                    local_prediction[str(i)] = download_json_prediction[i]
                with open(this_local_prediction_path, 'w') as f:
                    json.dump(local_prediction,
                              f,
                              ensure_ascii=False,
                              indent=4)
                f.close()

                # save results
                with open(this_local_result_path, 'w') as f:
                    json.dump(download_json_result,
                              f,
                              ensure_ascii=False,
                              indent=4)
                f.close()
    return True


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
