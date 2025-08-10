# flake8: noqa: E501
from .common_answers import (common_option_1_list, common_option_2_list,
                             common_start_op1_dict, common_start_op2_dict)


def get_gt_label(item):
    return item['gt_answer']


def get_pred_label(model_response, item, prompt_style, type):
    model_response = model_response.strip().lower()
    hypothesis1 = item['hypothesis1'].strip().lower()
    hypothesis2 = item['hypothesis2'].strip().lower()
    len1 = len(hypothesis1)
    len2 = len(hypothesis2)
    low_index = len(model_response)
    ask_for = item['ask-for']

    start_str1_dict = {
        **common_start_op1_dict,
        len(hypothesis1) - 1: [
            f'答案（选项一或选项二？）：{hypothesis1[:-1]}',
            f'answer (option 1 or option 2) : {hypothesis1[:-1]}'
        ]
    }
    start_str2_dict = {
        **common_start_op2_dict,
        len(hypothesis2) - 1: [
            f'答案（选项一或选项二？）：{hypothesis2[:-1]}',
            f'answer (option 1 or option 2) : {hypothesis2[:-1]}'
        ]
    }
    start_option1_list, start_option2_list = [], []
    # some of the model will give response containing the question, we usually preprocess the response to remove the question part, but sometimes due to the model's response format, some of the question part is not removed, so here we are checking the response with the question part as well.
    for key in start_str1_dict.keys():
        for str1 in start_str1_dict[key]:
            for i in range(key, len(str1) + 1):
                start_option1_list.append(str1[-i:])
    for key in start_str2_dict.keys():
        for str2 in start_str2_dict[key]:
            for i in range(key, len(str2) + 1):
                start_option2_list.append(str2[-i:])

    inner_option1_list = [
        'answer (option 1 or option 2 ?): {}'.format(hypothesis1[:len1 - 1]),
        'answer (option 1 or option 2?): {}'.format({hypothesis1[:len1 - 1]}),
        'the {} of the input event is that {}'.format(ask_for,
                                                      hypothesis1[:len1 - 1]),
        'the {} of the input event is option 1'.format(ask_for),
        'because {}'.format(hypothesis1[:len1 - 1]), 'answer is option 1',
        'answer is: option 1', 'answer: option 1', hypothesis1,
        hypothesis1[:len1 - 1], 'should be 1', 'i believe option 1', 'is 1',
        'select option 1', '正确答案是选项一', '答案为选项一', '应该选择选项一', '答案：选项一', '答案是选项一'
    ] + common_option_1_list
    inner_option2_list = [
        'answer (option 1 or option 2 ?): {}'.format(hypothesis2[:len2 - 1]),
        'answer (option 1 or option 2?): {}'.format({hypothesis2[:len2 - 1]}),
        'the {} of the input event is that {}'.format(ask_for,
                                                      hypothesis2[:len1 - 1]),
        'the {} of the input event is option 2'.format(ask_for),
        'because {}'.format(hypothesis2[:len2 - 1]), 'answer is option 2',
        'answer is: option 2', 'answer: option 2', hypothesis2,
        hypothesis2[:len2 - 1], 'should be 2', 'i believe option 2', 'is 2',
        'select option 2', '正确答案是选项二', '答案为选项二', '应该选择选项二', '答案是选项二'
    ] + common_option_2_list

    if model_response.startswith(tuple(start_option1_list)) \
        or any(hypothesis1 == option for option in [model_response, model_response[:len1], model_response + '.']) \
        or model_response in hypothesis1 and len(model_response) > 1:
        label = 0
    elif model_response.startswith(tuple(start_option2_list)) \
        or any(hypothesis2 == option for option in [model_response, model_response[:len2], model_response + '.']) \
        or model_response in hypothesis2 and len(model_response) > 1:
        label = 1
    elif any(
            model_response.find(option) > -1 and
        (low_index := min(low_index, model_response.find(option))) > -1
            for option in inner_option1_list):
        label = 0
        if any(option in model_response
               and model_response.find(option) < low_index
               for option in inner_option2_list):
            label = 1
    elif any(
            model_response.find(option) > -1 for option in inner_option2_list):
        label = 1
    else:
        return -1
    return label
