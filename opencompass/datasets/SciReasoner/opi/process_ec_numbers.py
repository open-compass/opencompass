import json
import re
from typing import Any


def add_spaces_to_ec_number(text: str) -> str:
    """
    在EC号码中添加空格，格式从 2.7.10.2 变为 2 . 7 . 10 . 2
    """
    # 匹配EC号码格式：数字.数字.数字.数字
    pattern = r'\b(\d+)\.(\d+)\.(\d+)\.(\d+)\b'

    def replace_ec(match):
        return (f'{match.group(1)} . {match.group(2)} .',
                f' {match.group(3)} . {match.group(4)}')

    return re.sub(pattern, replace_ec, text)


def process_json_value(value: Any) -> Any:
    """
    递归处理JSON值，在字符串中添加EC号码空格
    """
    if isinstance(value, str):
        return add_spaces_to_ec_number(value)
    elif isinstance(value, dict):
        return {k: process_json_value(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [process_json_value(item) for item in value]
    else:
        return value


def process_ec_json_file(input_file: str, output_file: str) -> None:
    """
    处理JSON文件，将所有EC号码格式化为带空格的格式
    """
    try:
        # 读取JSON文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 处理数据
        processed_data = process_json_value(data)

        # 写入新文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)

        print(f'处理完成！已保存到 {output_file}')

    except Exception as e:
        print(f'处理文件时出错: {e}')


if __name__ == '__main__':
    input_file = \
        'cot_data/EC_number_train_CLEAN_EC_number_train_train.json_final.json'
    output_file = \
        'cot_data/EC_number_train_CLEAN_EC_number_train_train_spaced.json'

    process_ec_json_file(input_file, output_file)
