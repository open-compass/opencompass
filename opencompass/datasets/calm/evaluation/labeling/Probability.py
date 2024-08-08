# flake8: noqa: E501
import json
import re

from .common_answers import (add_quotes_to_unquoted, change_quotation,
                             is_numeric)


def get_gt_label(item):
    return item['gt_answer']


# common function for maths
def extract_prob(model_response, prompt_style, type):
    model_response += '}'
    if 'CoT' in prompt_style and any(
            match in type for match in ['NIE', 'NDE', 'ETT', 'CDE', 'ATE']):
        matches = re.findall(r'\{\"answer\":.*?\}', model_response, re.DOTALL)
    else:
        matches = re.findall(r'\{+.*?\}+', model_response,
                             re.DOTALL | re.IGNORECASE)
    matched_str = None
    for match in matches:
        if match:
            matched_str = match.lower()
            if matched_str.startswith('{{') and matched_str.endswith('}}}'):
                matched_str = matched_str[1:-2]
            elif matched_str.startswith('{{') and matched_str.endswith('}}'):
                matched_str = matched_str[1:-1]
            elif matched_str.startswith('{{') and matched_str.endswith('}'):
                matched_str = matched_str[1:]
            elif matched_str.startswith('{') and matched_str.endswith('}}'):
                matched_str = matched_str[:-1]
        else:
            matched_str = None

        if matched_str:
            try:
                inner_json_obj = json.loads(matched_str)
            except json.JSONDecodeError:
                # If parsing fails, try adding quotes to unquoted words and parse again
                fixed_json_str = add_quotes_to_unquoted(matched_str)
                fixed_json_str = change_quotation(fixed_json_str)
                try:
                    inner_json_obj = json.loads(fixed_json_str)
                except:
                    inner_json_obj = {}

            prob_str_value = inner_json_obj.get('prob', None)
            if prob_str_value is not None:
                break
    if matched_str is None:
        prob_str_value = None

    pred_value = float(prob_str_value) if prob_str_value and is_numeric(
        prob_str_value) else None

    return pred_value


def get_pred_label(model_response, item, prompt_style, type):
    model_response = model_response.strip().lower()
    pred = extract_prob(model_response, prompt_style, type)
    return pred
