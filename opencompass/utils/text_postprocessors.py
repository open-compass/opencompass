import random
import re

import numpy as np

from opencompass.registry import TEXT_POSTPROCESSORS

random.seed(123)


def normalize_answer(text, unit):
    # ["1,000", "123", "3/4", "56.456", "$56.4", "-3", "-10.02", "-3/2"]

    text = re.sub(r'^[\$]', '', text)
    text = re.sub(r'[\,\.\,\/]$', '', text)

    result = re.match(r'^[-+]?[\d,./]+$', text)

    if result is not None:
        # is number?
        text = text.replace(',', '')
        result = re.match(r'[-+]?\d+$', text)

        if result is not None:
            number = int(text)
        elif '/' in text:
            nums = text.split('/')
            number = round(float(nums[0]) / float(nums[1]), 3)
        else:
            number = round(float(text), 3)
        number = str(number)
        number = re.sub(r'\.[0]+$', '', number)
        return number
    else:
        # is text
        if unit:
            text = text.replace(unit, '').strip()
        return text


def score_string_similarity(str1, str2):
    if str1 == str2:
        return 2.0
    if ' ' in str1 or ' ' in str2:
        str1_split = str1.split(' ')
        str2_split = str2.split(' ')
        overlap = list(set(str1_split) & set(str2_split))
        return len(overlap) / max(len(str1_split), len(str2_split))
    else:
        if str1 == str2:
            return 1.0
        else:
            return 0.0


def extract_prediction(output, options=None, option_inds='ABCDEFGH'):

    # $\\frac{16}{95}$ -> 16/95
    output = re.sub(r'\$?\\frac\{([\d\.\,\-]+)\}\{([\d\.\,]+)\}\$?', r'\1/\2',
                    output)

    output = re.sub(r'(?<![AP]\.M)\.$', '', output)
    output = re.sub(r'(?<=\d)[\=](?=[\-\$\d])', ' = ', output)
    output = re.sub(r'\u2212', '-', output)

    # Multi-choice questions
    if options:
        patterns = [
            r'^\(([A-Za-z])\)$',  # "(b)", "(B)"
            r'^([A-Za-z])$',  # "b", "B"
            r'^([A-Za-z]). ',  # "b", "B"
            r'[Th]he answer is ([A-Z])',  # "The answer is B"
            r'^\(([A-Za-z])\) [\s\S]+$',  # "(A) XXXXX"
            r'[Th]he answer is \(([A-Za-z])\) [\s\S]+$'
        ]

        # have "X" in the output
        for p in patterns:
            pattern = re.compile(p)
            res = pattern.findall(output)
            if len(res) > 0:
                pred = res[0].upper()  # e.g., "B"
                if pred in option_inds:
                    ind = option_inds.index(pred)  # 1
                    if ind >= len(options):
                        ind = random.choice(range(len(options)))
                    prediction = options[ind]
                    return prediction

        # find the most similar options
        scores = [score_string_similarity(x, output) for x in options]
        max_idx = int(
            np.argmax(scores))  # json does not recognize NumPy data types
        prediction = options[max_idx]
        return prediction

    else:
        # free_text QA problems, numeric answer
        patterns = [
            r'[Th]he answer is ([\s\S]+)$',  # "The answer is XXXXX.",
            r'[Th]he table shows that ([\d\$\.\,\/\:]+) ',
            r' = ([\d\$\.\,\/\:]+)',  # "= $1.40"
            r'(?<= be| is) ([\-\d\$\.\,\/\:]{0,}[\d]+)',  # "will be $1.40"
            r'(?<= are| was) ([\-\d\$\.\,\/\:]{0,}[\d]+)',  # "are $1.40"
            r'(?<= were) ([\-\d\$\.\,\/\:]{0,}[\d]+)',  # "are $1.40"
            r' ([\d\$\.\,\/\:]+ [AP]\.M\.)',  # 7:25 P.M.
            r'([\-\d\$\.\,\/\:]{0,}[\d]+)',  # 14.5
        ]

        for p in patterns:
            pattern = re.compile(p)
            res = pattern.findall(output)
            if len(res) > 0:
                prediction = res[-1].strip()
                if prediction.endswith('.') and '.M.' not in prediction:
                    prediction = prediction[:-1]
                return prediction

    return output


@TEXT_POSTPROCESSORS.register_module('general')
def general_postprocess(text: str) -> str:
    # Cut off the first newline, period, or comma
    truncated_text = re.split(r'[\n.,]', text, 1)[0]

    # Remove punctuation
    no_punctuation = re.sub(r'[^\w\s]', '', truncated_text)

    # Remove article
    no_articles = re.sub(r'\b(a|an|the)\b',
                         '',
                         no_punctuation,
                         flags=re.IGNORECASE)

    # Remove duplicated blank spaces
    cleaned_text = re.sub(r'\s+', ' ', no_articles).strip()

    return cleaned_text


@TEXT_POSTPROCESSORS.register_module('general_cn')
def general_cn_postprocess(text: str) -> str:
    truncated_text = re.split(r'[\n.,]', text, 1)[0]

    no_punctuation = re.sub(r'[^\w\s]', '', truncated_text)

    no_articles = re.sub(r'\b(a|an|the)\b',
                         '',
                         no_punctuation,
                         flags=re.IGNORECASE)

    cleaned_text = re.sub(r'\s+', ' ', no_articles).strip()
    import jieba
    cleaned_text = ' '.join(jieba.cut(text))
    return cleaned_text


@TEXT_POSTPROCESSORS.register_module('first-capital')
def first_capital_postprocess(text: str) -> str:
    for t in text:
        if t.isupper():
            return t
    return ''


def first_option_postprocess(text: str, options: str) -> str:
    """Find first valid option for text."""

    patterns = [
        f'[Tt]he answer is [{options}]',
        f'[Tt]he correct answer is [{options}]',
        f'答案是(.*?)[{options}]',
        f'答案为(.*?)[{options}]',
        f'固选(.*?)[{options}]',
        f'答案应该是(.*?)[{options}]',
        f'(\s|^)[{options}][\s。，,\.$]',  # noqa
        f'[{options}]',
    ]

    regexes = [re.compile(pattern) for pattern in patterns]
    for regex in regexes:
        match = regex.search(text)
        if match:
            outputs = match.group(0)
            for i in options:
                if i in outputs:
                    return i
    return ''


@TEXT_POSTPROCESSORS.register_module('first-capital-multi')
def first_capital_postprocess_multi(text: str) -> str:
    match = re.search(r'([A-D]+)', text)
    if match:
        return match.group(1)
    return ''


def last_option_postprocess(text: str, options: str) -> str:
    match = re.findall(rf'([{options}])', text)
    if match:
        return match[-1]
    return ''


def first_number_postprocess(text: str) -> float:
    """Return the first number in a string."""
    # regex pattern to match numbers (both integers and decimals)
    pattern = r'(-?\d*\.?\d+)'

    # search the string for the pattern
    match = re.search(pattern, text)

    # if a match is found, return it. Otherwise, return None.
    return float(match.group(1)) if match else None
