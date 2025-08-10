import re


def check_standalization(model_response, prompt_style, type):
    if model_response.startswith(('option 1', 'option 2', '选项一', '选项二')):
        return 0
    else:
        return 1


def check_empty(model_response):
    if model_response == '':
        return 1
    else:
        return 0


def check_repetition(model_response):
    if any(response in model_response for response in [
            'answer (option 1 or option 2 ?)', 'input event:', '答案（选项一或选项二？）',
            '输入事件：'
    ]):
        return 1
    else:
        return 0


def contains_chinese(text):
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
    result = 1 if chinese_pattern.search(text) is not None else 0
    return result


def contains_english(text):
    english_pattern = re.compile(r'[A-Za-z]{2,}')
    result = 1 if english_pattern.search(text) is not None else 0

    return result


def check_abnormality(preds):
    abnormalities = 'All option1' if all(pred == 0 for pred in preds) else \
                    'All option2' if all(pred == 1 for pred in preds) else 0
    return abnormalities
