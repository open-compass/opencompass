# flake8: noqa

import re


def parse_float_answer(raw_string, option=''):
    number_pattern = re.compile(r'[-+]?\d+(\.\d+)?([eE][-+]?\d+)?')

    # Search for the first match
    match = number_pattern.search(raw_string)
    if match:
        # Extract the matched number and convert it to float
        return float(match.group())
    else:
        # Return None if no number is found
        return 0


def parse_true_false_answer(raw_string, option=''):
    if 'yes' in raw_string.lower():
        return True
    elif 'no' in raw_string.lower():
        return False
    else:
        return True
