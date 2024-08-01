# flake8: noqa: E501
from .common_answers import (common_false_list, common_start_false_dict,
                             common_start_true_dict, common_true_list)


def get_gt_label(item):
    return item['gt_answer']


def get_pred_label(model_response, item, prompt_style, type):
    model_response = model_response.strip().lower()
    low_index = len(model_response)
    start_str1_dict = common_start_true_dict
    start_str2_dict = common_start_false_dict

    start_option1_list, start_option2_list = [], []
    # some of the model will give response containing the question, we usually
    # preprocess the response to remove the question part, but sometimes due to
    # the model's response format, some of the question part is not removed, so
    # here we are checking the response with the question part as well.
    for key1, key2 in zip(start_str1_dict.keys(), start_str2_dict.keys()):
        for str1, str2 in zip(start_str1_dict[key1], start_str2_dict[key2]):
            for i in range(key1, len(str1) + 1):
                start_option1_list.append(str1[-i:])
            for i in range(key2, len(str2) + 1):
                start_option2_list.append(str2[-i:])

    inner_option1_list = [
        'serves as the parent node of', 'serves as a parent node of'
    ] + common_true_list
    inner_option2_list = common_false_list
    if model_response.startswith(tuple(start_option1_list)):
        label = 1
    elif model_response.startswith(tuple(start_option2_list)):
        label = 0
    elif any(model_response.find(option)>-1 and (low_index := min(low_index, model_response.find(option))) > -1 for option in inner_option1_list) \
        or 'yes' in model_response and 'is the parent of' in model_response \
            or '是' in model_response and '父节点' in model_response:
        label = 1
        if any(option in model_response
               and model_response.find(option) < low_index
               for option in inner_option2_list):
            label = 0
    elif any(response in model_response for response in inner_option2_list)\
            or ('不是' in model_response and '父节点' in model_response):
        label = 0
    else:
        return -1
    return label
