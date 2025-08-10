# flake8: noqa: E501
import re


def check_standalization(model_response, prompt_style, type):
    if any(match in type for match in ['NIE', 'NDE', 'ETT', 'CDE', 'ATE']):
        if model_response.startswith(
            ("{\"answer\":")) and model_response.endswith(('}')):
            return 0
        else:
            return 1
    elif any(match in type for match in ['PN', 'PS']):
        if model_response.startswith(
            ("{\"prob\":")) and model_response.endswith(('}')):
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
            'input info: imagine a self-contained',
            'provide the calculation result to four decimal places',
            '输入信息：设想一个', '请根据上述信息，给出计算结果（答案保留四位小数）'
    ]):
        return 1
    else:
        return 0


def contains_chinese(text):
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
    result = 1 if chinese_pattern.search(text) is not None else 0

    return result


def contains_english(text):
    english_pattern = re.compile(r'[A-Za-z]{7,}')
    # Taking into account 'fake' and 'random' modes, and
    # considering that the shortest occurrence of English characters
    # in an 'answer' is of length 6, therefore detecting
    # lengths of 7 or more.
    result = 1 if english_pattern.search(text) is not None else 0

    return result


def check_abnormality(preds):
    affect_num = sum(
        1 for pred in preds if pred == 0.1234
    )  # 0.1234 is the example value in prompt for probability computation
    affected = affect_num / len(preds)
    abnormalities = 'All Yes' if affected == 1 else \
                    'All No' if all(pred == 0 for pred in preds) else 0
    return abnormalities
