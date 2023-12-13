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


input_path = 'your prediction file path'
file_name = 'alignment_bench_test'
extract_predictions_from_json(input_path, file_name)
