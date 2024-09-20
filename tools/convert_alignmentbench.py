import argparse
import csv
import json
import os
from glob import glob

from tqdm import tqdm


def extract_predictions_from_json(input_folder):

    sub_folder = os.path.join(input_folder, 'submission')
    pred_folder = os.path.join(input_folder, 'predictions')
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)

    for model_name in os.listdir(pred_folder):
        model_folder = os.path.join(pred_folder, model_name)
        try:
            # when use split
            json_paths = glob(
                os.path.join(model_folder, 'alignment_bench_*.json'))
            # sorted by index
            json_paths = sorted(
                json_paths,
                key=lambda x: int(x.split('.json')[0].split('_')[-1]))
        except Exception as e:
            # when only one complete file
            print(e)
            json_paths = [os.path.join(model_folder, 'alignment_bench.json')]

        all_predictions = []
        for json_ in json_paths:
            json_data = json.load(open(json_))
            for _, value in json_data.items():
                prediction = value['prediction']
                all_predictions.append(prediction)

        # for prediction
        output_path = os.path.join(sub_folder, model_name + '_submission.csv')
        with open(output_path, 'w', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            for ans in tqdm(all_predictions):
                writer.writerow([str(ans)])
        print('Saved {} for submission'.format(output_path))


def process_jsonl(file_path):
    new_data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_data = json.loads(line)
            new_dict = {
                'question': json_data['question'],
                'capability': json_data['category'],
                'others': {
                    'subcategory': json_data['subcategory'],
                    'reference': json_data['reference'],
                    'question_id': json_data['question_id']
                }
            }
            new_data.append(new_dict)
    return new_data


def save_as_json(data, output_file='./alignment_bench.json'):
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(description='File Converter')
    parser.add_argument('--mode',
                        default='json',
                        help='The mode of convert to json or convert to csv')
    parser.add_argument('--jsonl',
                        default='./data_release.jsonl',
                        help='The original jsonl path')
    parser.add_argument('--json',
                        default='./alignment_bench.json',
                        help='The results json path')
    parser.add_argument('--exp-folder', help='The results json name')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    mode = args.mode
    if mode == 'json':
        processed_data = process_jsonl(args.jsonl)
        save_as_json(processed_data, args.json)
    elif mode == 'csv':
        extract_predictions_from_json(args.exp_folder)
