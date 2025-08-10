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
    # some of the model will give response containing the question,
    # we usually preprocess the response to remove the question part,
    # but sometimes due to the model's response format, some of the
    # question part is not removed, so here we are checking the response
    # with the question part as well.
    for key in start_str1_dict.keys():
        for str1 in start_str1_dict[key]:
            for i in range(key, len(str1) + 1):
                start_option1_list.append(str1[-i:])
    for key in start_str2_dict.keys():
        for str2 in start_str2_dict[key]:
            for i in range(key, len(str2) + 1):
                start_option2_list.append(str2[-i:])

    inner_option1_list = [
        'can be identified', '可以被识别', '能被识别', 'answer (yes or no?): yes',
        'answer is yes', "\"yes\"", 'answer: yes', 'answer is: yes',
        'answer is:\n\nyes', 'answer is:\nyes', 'is identified.',
        'can be identified', '可以被识别', '能被识别', '答案是:是', '答案是:\n\n是', '答案是:\n是',
        '答案:是', '答案是是', "\"是\"", '是的', '答案为“是”', '答案是“是”', '可以识别', '答案：是',
        '答案：可以', '答案：“是”', 'thus answering yes', 'henceforth; answering yes',
        'by answering yes', 'answeristheyes', 'answer would be yes',
        'answer (yes)', 'hence answering yes', 'hence my answer yes',
        'answer would definitely become yes', 'answer remains yes',
        "my answer was 'yes'", 'thus concludes our answer yes',
        'must answer yes', "answer should be 'yes'", "answer remains 'yes'",
        'henceforth answering yes', 'answer should be marked yes',
        'answer comes out yes', "should answer 'yes",
        'our answer should be yes', 'you should answer yes',
        'concluding answer - yes', 'answer should indeed say yes',
        'answer : yes', 'answer should also be yes', 'hence answering yes',
        'the answer is trivially yes', 'answer:  yes', 'the answer is (yes)',
        '答案应为“是”'
    ] + common_true_list
    inner_option2_list = [
        'not identified', '不能被识别', '无法被识别', 'answer (yes or no?): no',
        'answer is no', "\"no\"", 'answer: no', 'answer is: no',
        'answer is:\n\nno', 'answer is:\nno', 'not identified', '不能被识别',
        '无法被识别', '答案是:否', '答案是:\n\n否', '答案是:\n否', '答案:否', '答案是否', "\"否\"",
        '回答是:否', '答案为“否”', '答案是“否”', '因果效应不可被识别', '答案：否', '答案：无法识别',
        '不存在可识别的因果效应', "doesn't have a causal relationship",
        'the correct answer should be no', 'answer would be no',
        'hence answering no', "answering your query 'no'",
        'therefore answering no', 'answer would be “no”', 'thus answering no',
        'this answers no', 'thus, answering no', 'answer should also be no',
        'answer would also turn out to be no', 'answer would have to be no',
        'answer would be – no', 'thus answering “no”', 'answer = no',
        'answer should be no', 'answer would definitely be no',
        'answer would need to be no', 'answer would need to be marked no',
        'hence why i answered “no', "hence answering 'no'",
        'answer must necessarily remain no', 'answer should marked no',
        'answer would most likely be no', 'answer would also be no',
        'answer for now might have to be `no`', 'henceforth - answer no',
        'answer could only be no', 'answer would also be no',
        'henceforth answering “no', 'answer would be no', 'hence answering no',
        'cannot be identified', 'answer (yes or no ?): no', '答案为“不”',
        'henceforth answering no', '答案为:否', '答案应该是“否', '因果效应不可被'
    ] + common_false_list
    if model_response.startswith(tuple(start_option1_list)):
        label = 1
    elif model_response.startswith(tuple(start_option2_list)):
        label = 0
    elif any(
            model_response.find(option) > -1 and
        (low_index := min(low_index, model_response.find(option))) > -1
            for option in inner_option1_list):
        label = 1
        if any(option in model_response
               and model_response.find(option) < low_index
               for option in inner_option2_list):
            label = 0
    elif any(response in model_response for response in inner_option2_list):
        label = 0
    else:
        return -1
    return label
