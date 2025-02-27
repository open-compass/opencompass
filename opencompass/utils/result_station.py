import os
import os.path as osp
from typing import List, Tuple, Union
from mmengine.config import Config
import json
import re





def Save_To_Station(cfg, args):
    from dotenv import load_dotenv
    load_dotenv()
    station_path = os.getenv('RESULTS_STATION_PATH')
    assert station_path != None or args.station_path != None
    station_path = args.station_path if station_path == None else station_path


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
                print('result of {} with {} already exists'.format(dataset, model))
                continue
            else:

                # get result dict
                local_result_path = work_dict + '/results/' + model + '/'
                local_result_json = local_result_path + dataset + '.json'
                if not osp.exists(local_result_json):
                    raise ValueError('invalid file: {}'.format(local_result_json))
                with open(local_result_json, 'r') as f:
                    this_result = json.load(f)
                f.close()

                # get prediction list
                local_prediction_path = work_dict + '/predictions/' + model + '/'
                local_prediction_regex = rf"^{re.escape(dataset)}(?:_\d+)?\.json$"
                local_prediction_json = find_files_by_regex(local_prediction_path, local_prediction_regex)
                if not check_filenames(dataset, local_prediction_json):
                    raise ValueError('invalid filelist: {}'.format(local_prediction_json))

                this_prediction = []
                for prediction_json in local_prediction_json:
                    with open(local_prediction_path + prediction_json, 'r') as f:
                        this_prediction_load_json = json.load(f)
                    f.close()
                    for prekey in this_prediction_load_json.keys():
                        this_prediction.append(this_prediction_load_json[prekey])

                # dict combine
                data_model_results = {
                        'predictions': this_prediction,
                        'results': this_result
                }
                with open(osp.join(result_path, result_file_name), 'w') as f:
                    json.dump(data_model_results, f, ensure_ascii=False, indent=4)
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

    single_pattern = re.compile(rf"^{re.escape(x)}\.json$")
    numbered_pattern = re.compile(rf"^{re.escape(x)}_(\d+)\.json$")

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
