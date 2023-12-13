import argparse
import csv
import json
import os


def extract_predictions_from_json(input_path, file_name):
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file == f'{file_name}.json':
                file_path = os.path.join(root, file)
                output_csv = os.path.join(root, f'{file_name}.csv')

                with open(file_path, 'r', encoding='utf-8') as json_file:
                    data = json.load(json_file)
                    predictions = []

                    for key in data:
                        prediction = data[key].get('prediction', '')
                        predictions.append(prediction)

                with open(output_csv, 'w', newline='',
                          encoding='utf-8') as csv_file:
                    writer = csv.writer(csv_file)

                    for prediction in predictions:
                        writer.writerow([prediction])


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
                        default='your prediction file path',
                        help='The results json path')
    parser.add_argument('--name',
                        default='alignment_bench',
                        help='The results json name')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    mode = args.mode
    if mode == 'json':
        processed_data = process_jsonl(args.jsonl)
        save_as_json(processed_data)
    elif mode == 'csv':
        extract_predictions_from_json(args.json, args.name)
