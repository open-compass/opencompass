import json


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


def save_as_json(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


input_jsonl_file = './data_release.jsonl'
output_json_file = './alignment_bench.json'
processed_data = process_jsonl(input_jsonl_file)
save_as_json(processed_data, output_json_file)
