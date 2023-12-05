import re

from opencompass.datasets.humaneval import humaneval_gpt_postprocess
from opencompass.datasets.record import ReCoRD_postprocess
from opencompass.datasets.xsum import Xsum_postprocess
from opencompass.utils.text_postprocessors import first_option_postprocess


def gsm8k_postprocess(text: str) -> str:
    text = text.split(' ')[::-1]
    flag = False
    ret = ''
    for i in range(len(text)):
        s = text[i]
        for i in range(len(s)):
            if s[i].isdigit():
                flag = True
                ret = s
                break
        if flag:
            break
    ret1 = ''
    for i in range(len(ret)):
        if ret[i].isdigit():
            ret1 += ret[i]
    return ret1


def humaneval_postprocess(text: str) -> str:
    text = '\n'.join(text.split('\n')[1:]).strip()
    if '```' in text:
        blocks = re.findall(r'```(.*?)```', text, re.DOTALL)
        if len(blocks) == 0:
            text = text.split('```')[1]  # fall back to default strategy
        else:
            text = blocks[0]  # fetch the first code block
            if not text.startswith('\n'):  # in case starting with ```python
                text = text[max(text.find('\n') + 1, 0):]
    if text.strip().startswith('from') or text.strip().startswith('import'):
        def_idx = text.find('def')
        if def_idx != -1:
            text = text[max(text.find('\n', def_idx) + 1, 0):]
    if text.strip().startswith('def'):
        text = '\n'.join(text.split('\n')[1:])
    if not text.startswith('    '):
        if text.startswith(' '):
            text = '    ' + text.lstrip()
        else:
            text = '\n'.join(['    ' + line for line in text.split('\n')])
    return text


def lcsts_postprocess(text: str) -> str:
    text = text.strip()
    text = text.replace('1. ', '') if text.startswith('1. ') else text
    text = text.replace('- ', '') if text.startswith('- ') else text
    text = text.strip('“，。！”')
    return text


def mbpp_postprocess(text: str) -> str:
    if text.startswith('Here'):
        text = '\n'.join(text.split('\n')[1:]).strip()
    if '```' in text:
        blocks = re.findall(r'```(.*?)```', text, re.DOTALL)
        if len(blocks) == 0:
            text = text.split('```')[1]  # fall back to default strategy
        else:
            text = blocks[0]  # fetch the first code block
            if not text.startswith('\n'):  # in case starting with ```python
                text = text[max(text.find('\n') + 1, 0):]
    return text


def strategyqa_pred_postprocess(text: str) -> str:
    if text.startswith('Here'):
        text = '\n'.join(text.split('\n')[1:]).strip()
    text = text.split('answer is ')[-1]
    match = re.search(r'(yes|no)', text.lower())
    if match:
        return match.group(1)
    return ''


def flores_postprocess(text: str) -> str:
    text = text.strip().split('\n')[-1].strip()
    return text


def flores_postprocess_chinese(text: str) -> str:
    text = text.strip().split('\n')[-1].strip()
    import jieba
    truncated_text = text.strip().split('\n')[0]
    cleaned_text = re.sub(r'\s+', ' ', truncated_text).strip()
    cleaned_text = ' '.join(jieba.cut(cleaned_text))
    return cleaned_text


def record_postprocess(text: str) -> str:
    match = re.search(r'(?<=refers to )[^.]+', text)

    if match:
        return match.group().strip()  # Outputs: abc def

    return ReCoRD_postprocess(text)


def humaneval_claude2_postprocess(text: str) -> str:
    if text.startswith('Here'):
        text = '\n\n'.join(text.split('\n\n')[1:])
    return humaneval_gpt_postprocess(text)


def xsum_postprocess(text: str) -> str:
    if text.startswith('Here'):
        text = '\n\n'.join(text.split('\n\n')[1:])
    return Xsum_postprocess(text)


def yes_no_postprocess(text: str) -> str:
    if 'yes' in text.lower():
        return 'A'
    elif 'no' in text.lower():
        return 'B'
    return first_option_postprocess(text, 'AB')
