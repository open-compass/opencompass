# flake8: noqa: E501
from .common_answers import (common_option_1_list, common_option_2_list,
                             common_option_3_list, common_start_op1_dict,
                             common_start_op2_dict, common_start_op3_dict)


def get_gt_label(item):
    return int(item['gt_answer'])


def get_pred_label(model_response, item, prompt_style, type):
    model_response = model_response.strip().lower()
    low_index = len(model_response)
    Answer1 = item['option1'].strip().lower()
    Answer2 = item['option2'].strip().lower()
    Answer3 = item['option3'].strip().lower()
    start_str1_dict = {
        **common_start_op1_dict,
        len(Answer1) - 1: [
            f'答案（选项一或选项二或选项三？）：{Answer1[:-1]}',
            f'答案（选项一或选项二或选项三？）： {Answer1[:-1]}',
            f'answer (option 1 or 2 or 3?):{Answer1[:-1]}',
            f'answer (option 1 or 2 or 3?): {Answer1[:-1]}'
        ]
    }
    start_str2_dict = {
        **common_start_op2_dict,
        len(Answer2) - 1: [
            f'答案（选项一或选项二或选项三？）：{Answer2[:-1]}',
            f'答案（选项一或选项二或选项三？）： {Answer2[:-1]}',
            f'answer (option 1 or 2 or 3?):{Answer2[:-1]}',
            f'answer (option 1 or 2 or 3?): {Answer2[:-1]}'
        ]
    }
    start_str3_dict = {
        **common_start_op3_dict,
        len(Answer3) - 1: [
            f'答案（选项一或选项二或选项三？）：{Answer3[:-1]}',
            f'答案（选项一或选项二或选项三？）： {Answer3[:-1]}',
            f'answer (option 1 or 2 or 3?):{Answer3[:-1]}',
            f'answer (option 1 or 2 or 3?): {Answer3[:-1]}'
        ]
    }

    start_option1_list, start_option2_list, start_option3_list = [], [], []
    # some of the model will give response containing the question, we usually
    # preprocess the response to remove the question part, but sometimes due to
    # the model's response format, some of the question part is not removed, so
    # here we are checking the response with the question part as well.
    for key1, key2, key3 in zip(start_str1_dict.keys(), start_str2_dict.keys(),
                                start_str3_dict.keys()):
        for str1, str2, str3 in zip(start_str1_dict[key1],
                                    start_str2_dict[key2],
                                    start_str3_dict[key3]):
            for i in range(key1, len(str1) + 1):
                start_option1_list.append(str1[-i:])
            for i in range(key2, len(str2) + 1):
                start_option2_list.append(str2[-i:])
            for i in range(key3, len(str3) + 1):
                start_option3_list.append(str3[-i:])

    inner_option1_list = [
        'answer (option 1 or 2 or 3 ?): {}'.format(Answer1[:-1]),
        '(option 1 or 2 or 3?): {}'.format({Answer1[:-1]})
    ] + common_option_1_list
    inner_option2_list = [
        'answer (option 1 or 2 or 3 ?): {}'.format(Answer2[:-1]),
        '(option 1 or 2 or 3?): {}'.format({Answer2[:-1]})
    ] + common_option_2_list
    inner_option3_list = [
        'answer (option 1 or 2 or 3 ?): {}'.format(Answer3[:-1]),
        '(option 1 or 2 or 3?): {}'.format({Answer3[:-1]})
    ] + common_option_3_list

    if any(option in model_response for option in ['选项一或选项二','选项二或选项三','option 1 or option 2', 'option2 or option 3']) \
        or 'option 1' in model_response and 'option 2' in model_response and 'option 3' in model_response \
        or '选项一' in model_response and '选项二' in model_response and '选项三' in model_response \
        or len(model_response) == 0:
        return -1
    elif model_response.startswith(tuple(start_option1_list)) \
        or any(Answer1 == option for option in [model_response]) \
        or len(Answer1) > 1 and len(model_response) > 0 and (model_response in Answer1):
        label = 1
    elif model_response.startswith(tuple(start_option2_list)) \
        or any(Answer2 == option for option in [model_response]) \
        or len(Answer2) > 1 and len(model_response) > 0 and (model_response in Answer2):
        label = 2
    elif model_response.startswith(tuple(start_option3_list)) \
        or any(Answer3 == option for option in [model_response]) \
        or len(Answer3) > 1 and len(model_response) > 0 and (model_response in Answer3):
        label = 3
    elif any(model_response.find(option)>-1 and (low_index:=min(low_index, model_response.find(option)))>-1 for option in inner_option1_list)\
        or '正确答案' in model_response and ('选项一' in model_response):
        label = 1
        if any(option in model_response
               and model_response.find(option) < low_index
               for option in inner_option2_list):
            label = 2
            if any(option in model_response
                   and model_response.find(option) < low_index
                   for option in inner_option3_list):
                label = 3
    elif any(model_response.find(option) > -1 for option in inner_option2_list)\
        or '正确答案' in model_response and ('选项二' in model_response):
        label = 2
        if any(option in model_response
               and model_response.find(option) < low_index
               for option in inner_option3_list):
            label = 3
    elif any(model_response.find(option) > -1 for option in inner_option3_list)\
        or '正确答案' in model_response and ('选项三' in model_response):
        label = 3
    else:
        return -1
    return label
