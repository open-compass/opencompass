import argparse
import json
from pathlib import Path

from opencompass.datasets.chatml.verification import VerifyDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='Utils to check the format of your ChatML dataset files')
    parser.add_argument('--path',
                        type=str,
                        help='your dataset file path or category path.')
    return parser.parse_args()


def collect_file_paths(path):

    path_obj = Path(path)
    file_paths = []

    if not path_obj.exists():
        print(f"警告: 路径 '{path}' 不存在")
        return file_paths

    if path_obj.is_file():
        file_paths.append(str(path_obj))
    elif path_obj.is_dir():
        for file_path in path_obj.rglob('*'):
            if file_path.is_file():
                file_paths.append(str(file_path))

    file_paths = [f for f in file_paths if f.endswith('.jsonl')]
    return file_paths


def main():
    args = parse_args()
    all_check_files = collect_file_paths(args.path)
    for path in all_check_files:
        with open(path, 'r', encoding='utf-8-sig') as f:
            data = [json.loads(line) for line in f]
        for i in range(len(data)):
            key_list = list(data[i].keys())
            for key in key_list:
                if key != 'question' and key != 'answer':
                    del data[i][key]

        print(f': checking file {path} ...')
        for i in data:
            try:
                VerifyDataset(**i)
            except Exception as e:
                print(f': check failed. {e}')
                break

    print('format check finished!')


if __name__ == '__main__':
    main()
