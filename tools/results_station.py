import argparse
import json
import os

import yaml
from dotenv import load_dotenv

load_dotenv()
RESULTS_STATION_PATH = os.getenv('RESULTS_STATION_PATH')

data_file_map = {
    'ifeval': 'IFEval',
}

data_prefix_map = {}

with open('dataset-index.yml', 'r') as f1:
    data_list = yaml.load(f1, Loader=yaml.FullLoader)
f1.close()
data_searchable_list = [next(iter(i.keys())) for i in data_list]


def parse_args():
    parser = argparse.ArgumentParser(description='connect to results station')

    parser.add_argument('-sp',
                        '--station-path',
                        type=str,
                        default=None,
                        help='if no env path, use this.')
    parser.add_argument('-p',
                        '--my-path',
                        type=str,
                        default=None,
                        help='your operation path.')
    parser.add_argument(
        '-op',
        '--operation',
        type=str,
        default='d',
        help='u:update, d:download, ls: show dataset and model options')
    parser.add_argument('-d',
                        '--dataset',
                        type=str,
                        default='mmlu_pro',
                        help='target dataset name')
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        default='deepseek-v2_5-turbomind',
                        help='target model name')
    # parser.add_argument('-all',
    #                     '--all-transfer',
    #                     action='store_true',
    #                     default=False,
    #                     help='transfer all files under the path')

    args = parser.parse_args()
    return args


def read_json(path):
    results = []
    for i in path:
        with open(i, 'r') as f:
            results.append(json.load(f))
        f.close()
    return results


def load_json_files_by_prefix(prefix, target_path):
    if prefix in data_file_map.keys():
        prefix = data_file_map[prefix]
    result_dict = {}
    for filename in os.listdir(target_path):
        if filename.startswith(prefix) and filename.endswith('.json'):
            file_path = os.path.join(target_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                json_data = json.load(file)
            result_dict[os.path.splitext(filename)[0]] = json_data
    return result_dict


def main(path, mypath, args):

    if args.dataset not in data_searchable_list:
        raise ValueError('invalid dataset input!')

    update_path = path + args.dataset if path[
        -1] == '/' else path + '/' + args.dataset
    update_filename = args.dataset + '_' + args.model + '.json'
    update_goal = update_path + '/' + update_filename

    # update from your path to result station
    if args.operation == 'u':
        mypath_prediction = (mypath + 'predictions/' +
                             args.model) if mypath[-1] == '/' else (
                                 mypath + '/predictions/' + args.model)
        mypath_result = (mypath + 'results/' +
                         args.model) if mypath[-1] == '/' else (mypath +
                                                                '/results/' +
                                                                args.model)

        if os.path.exists(mypath_prediction) and os.path.exists(mypath_result):

            result_dict = load_json_files_by_prefix(args.dataset,
                                                    mypath_result)
            prediction_list = []
            for i in result_dict.keys():
                prediction_dict = load_json_files_by_prefix(
                    i, mypath_prediction)
                for j in range(len(prediction_dict)):
                    for k in prediction_dict[i + '_' + str(j)].keys():
                        prediction_list.append({
                            'prediction':
                            prediction_dict[i + '_' + str(j)][k],
                            'sub_category':
                            i
                        })
            update_dict = {
                'predictions': prediction_list,
                'results': result_dict,
            }

            if not os.path.exists(update_path):
                os.makedirs(update_path)
            if os.path.exists(update_goal):
                input('This result exists! Press any key to continue...')
            with open(update_goal, 'w', encoding='utf-8') as f:
                json.dump(update_dict, f, ensure_ascii=False, indent=4)
            f.close()

    # read from result station to your path
    if args.operation == 'd':
        if not os.path.exists(update_goal):
            raise ValueError('This result does not exist!')
        with open(update_goal, 'r', encoding='utf-8') as f:
            results = json.load(f)
        f.close()
        legal_key_set = {'predictions', 'results'}
        if set(results.keys()) == legal_key_set and isinstance(
                results['predictions'], list) and isinstance(
                    results['results'], dict):
            print('Successfully download result from station!'
                  "you've got a dict with format as follows:"
                  "\n content['precitions', 'results']")
        else:
            raise ValueError('illegal format of the result!')
        save_path = args.my_path if args.my_path[
            -1] == '/' else args.my_path + '/'
        save_path += args.dataset + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(save_path + update_filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        f.close()


if __name__ == '__main__':
    args = parse_args()

    if args.operation == 'ls':
        print('----DATASET LIST----')
        print(data_searchable_list)
        print('----MODEL LIST----')

    else:
        if RESULTS_STATION_PATH is not None:
            path = RESULTS_STATION_PATH
        else:
            path = args.station_path
        if path is None:
            raise ValueError('Please appoint the path of results station!')
        if not os.path.exists(path):
            raise ValueError('Not a valid path of results station!')
        mypath = args.my_path
        if mypath is None:
            raise ValueError('Please appoint your own path!')
        if not os.path.exists(mypath):
            raise ValueError('Not a valid path of your own path!')
        main(path, mypath, args)
