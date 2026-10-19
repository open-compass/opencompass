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
        'is a causal relationship', '答案为“会', '答案是「是',
        'is an independent causal factor', '构成因果关系', '之间存在因果', '是一个因果关系',
        '体现了因果关系', "因果关系是\"是\"", '因果关系为“是”', '存在一个因果关系', '是因果链关系', '有着因果关系',
        '具有因果', '是明显的因果关系', '具有“因果”关系', '存在了因果关系', "存在\"因果关系", '存在因果关系',
        '答案是真', '因此是“是', '答案为“有', '答案是是', '答案为“(是', "答案为\"\"是", '答案是:“是',
        "答案应为:\"是", "答案应为\"是", '答案为:是', "答案是:\"是", "答案应该是\"是", '答案为“yes',
        '具有因果关系', '答案是 “是', '答案“是', '答案必须是“yes', '答案处为“是', '答案应是“是', '答案為“是',
        '答案可以是“是', '答案的是“是', '答案为「是', "案为“\"是", "答案为 \"是", '答案是有', '答案是： 是',
        '答案为：是', '答案是对', '答案是：是', '答案是：\n是', '答案应为“是', '答案：是', '答案应该是“是',
        '答案(是或否？)：是的', "答案\"是\"", "答案都是\"是", '答案为是', '答案为 “是', "答案为\"是",
        '答案为“是', "答案是\"是\"", '答案是“是', '答案是“是”', 'answer (yes or no?): yes',
        'answer is yes', "\"yes\"", 'answer: yes', 'answer is: yes',
        'answer is:\n\nyes', 'answer is:\nyes', 'is a causal relationship',
        '答案为“是”', 'answer (yes )', 'answering your query, yes',
        'there is a direct causal relationship', 'directly resulted in',
        'there does appear to be a causal relationship',
        'there is no explicit statement of a direct causal relationship between',
        'there is no clear causal relationship', 'leads to', 'are caused by',
        'yes, there is a suggested causal relationship',
        'there could be a causal relationship',
        'there is a potential causal relationship between', 'the result of',
        'thus yes', '答案是：\nye', 'so yes', 'henceforth answering yes',
        'hence yes', 'answer (yes)', 'in short, yes', 'hence - yes',
        'correct response should be yes', 'thus, yes', 'in short - yes',
        '答案是：\n\nyes', 'leading us to believe yes!', 'hence answering yes',
        'therefore should read - yes', 'hence y es', 'therefore, yes',
        'therefore yes', 'the correct response should be marked as “yes.',
        'thus - yes', '因果关系是明确的', '答案是肯定的',
        'there is a direct cause-and-effect relationship between',
        'there is a cause-and-effect relationship between',
        'there is a clear causal relationship between',
        'there is a direct relationship with',
        'there is a direct cause and effect relationship between',
        'which implies a causal relationship between',
        'has a causal relationship with', "the answer is \"yes.\"",
        "therefore, the answer is \"yes,\"", 'answer (yes or no ? ) : yes',
        'answe: yes', 'is a result of',
        'this is a direct causal relationship between',
        'there is a significant causal relationship',
        'there is a clear cause and effect relationship',
        'this would be a direct causal relationship',
        'could be seen as a direct causal relationship between the two events',
        'this causal relationship is direct',
        'the causal relationship between the two events is therefore direct',
        'this indicates a causal relationship',
        'it is therefore a causal relationship',
        'this is a direct causal relationship',
        'this could be a causal relationship',
        'the answer to this question is yes',
        'refers to a cause-and-effect relationship',
        "there's a direct causal relationship",
        'the causal relationship is implied within',
        "there's a causal relationship between",
        'explicitly states a causal relationship',
        'a causal relationship can be inferred',
        'be a causal relationship between', 'the answer will be yes',
        'answer should be yes', 'the answer could be yes',
        'the answer for you is yes', 'this is the causal relationship between',
        'indicates a direct causal relationship',
        'should be a relationship between',
        'definitely got a causal relationship', "thus answering 'yes'",
        'thus answering yes', 'thereby answering yes',
        'answer would thus be yes', "so answering 'yes'",
        "hence answering 'yes'", "therefore answering 'yes",
        'confirming our answer yes',
        'an answer for this question would be yes', 'answer would be: yes',
        'implying a yes answer', 'making the answer yes',
        'incident does have a causal relationship',
        'the cause and effect relationship exists',
        'there is a direct cause relationship',
        'must have a causal relationship', 'answer would be yes',
        'a causal relationship exists between', 'answer(yes',
        'answer for this question is yes', 'answer (yes',
        'answer here is `yes`', 'answer might be yes', 'answer is a yes',
        'the answer yes', 'henceforth – yes', 'thus indicating yes',
        'hence indicating yes', "it's safe to say yes", "hence it's 'yes'",
        "thus answering 'yes’", 'so it’s yes', 'thus it can be said yes',
        'the correct response is yes', 'answering the question with a yes',
        "the correct answer would be \"yes", "the answer is \"yes”",
        "answer \"yes", 'the answer as yes', 'the answer to the question yes',
        'the answer is causality', 'the answer is yes', "the answer is \"yes",
        '答案是:是', '是因果关系', '有因果关系', '存在因果关', '因果关系存在'
    ] + common_true_list
    inner_option2_list = [
        'there is no causal relationship', 'answer (yes or no?): no',
        'answer is no', "\"no\"", 'answer: no', 'answer is: no',
        'answer is:\n\nno', 'answer is:\nno',
        'there is no causal relationship', '答案为“否”', 'answer (no )',
        'answering your query - no', 'there is no direct causal',
        'did not directly cause', 'not the sole or direct cause',
        'not directly causally', 'is not definitively said to be caused by',
        'the direction of causation is unclear',
        'there is not necessarily a causal relationship between', '答案是：no',
        'so no', 'henceforth answering no', 'hence no', 'answer (no)',
        'in short, no', 'making our assessment no',
        'correct response should be no', 'the answ er is no',
        'thus answering no', 'therefore answering no', 'thus no',
        'there is no direct cause and effect relationship between',
        'not directly related',
        'does not contain any direct cause-and-effect relationship between',
        'no clear causal connection between',
        'there is no direct cause-and-effect relationship between',
        'is not a cause-and-effect relationship',
        'is not a direct cause-and-effect relationship',
        'there is not a direct causal relationship between',
        "the answer is \"no.\"", 'the answer is therefore no',
        'was not a direct result of',
        'it is not a cause and effect relationship',
        'there is no clear relationship between',
        'there is no scientific evidence to support any specific causal relationship',
        'there is no evidence to suggest a causal relationship',
        'is not a causal relationship',
        'no scientific evidence to support any causal relationship',
        'does not mention a causal relationship between',
        'the answer to this question is no',
        "there isn't a cause-effect relationship between",
        "there's no causaltiy relationship",
        "this isn't a causal relationship",
        'no causal relationship is observed',
        "there isn't a causal relationship between",
        "doesn't indicate a cause and effect relationship",
        "doesn't indicate a causal relationship between", 'answer=no',
        "don't suggest a causal relationship",
        'does not indicate a causal relationship',
        "doesn't provide any causal relationship", 'hence answering no',
        "hence answering 'no'", 'answer should read : no',
        'therefore answer would be no', "answers should read 'no",
        'answer would need to be no', 'answering your above query : no',
        'answer would be no', "therefore, answering 'no", 'answer:no',
        'answer should remain no', 'the answer to this question would be no',
        'answer is:no', "answer is therefore \"no.\"", 'making the answer no',
        'the cause-and-effect relationship between these two words is not clear',
        'answer must be no', 'answer is therefore no',
        'there is no causality between', 'answer(no)', 'answer is, no',
        "answer might be \"no.\"", 'answer it as no',
        'should be the answer no', 'answering no', "thus answering 'no'",
        'thus, no', "therefore 'no'", 'the answer can be no', 'answer is “no',
        'the answer is mostly no', 'answer is probably not', "answer is \"no",
        '答案是“否”', '答案（是或否？）：否', "答案是\"否\"", "答案为\"否\"", '答案是否', '答案为“否',
        '答案为“不', '答案为“没有', '答案“否', '答案为“非”', '答案为“无”', '答案为”否', "答案为 \"否",
        '答案为否', '答案是\\”否', '答案应该是“否', '答案是：\nno', '答案是：\n否', '答案是：\n不', '答案：否',
        '答案应为“非', "答案\"否", '答案为**否', '答案在“否', '答案可能为“否', '答案返回“否', "答案为\"否",
        '答案是“不', '答案应该为“否', "答案为'否", '答案为不存  在', '答案应为“否', '答案为《否', '答案是“无',
        '答案为\\“否', '答案将是“否', '答案还是“否', '答案：“不', '答案 为“否', '答案应该是否',
        'the answer is no', '不存在“因果”关系', "答案应为\"否", "答案应该是\"否", '答案是:否',
        '答案为:否', "答案选择\"否", "答案是:\"否", "答案应该为\"否", "答案应为\"否", '答案选择为:否',
        '答案为 “否', '答案为“非', '答案为“没', '不存在因果关系', '没有直接因果关系', '没有因果关系',
        '不是一种因果关系', '不一定具有因果', '因果关系并不明确', '不包含因果关系', '并非因果关系', '因果关系“不存在',
        '没有推理上的因果关系', '与因果关系无关', '没有明显的因果关系', '没有什么因果关系', '不是一个因果关系',
        '不属于因果关系', '不能形成因果关系', '没有因果', '无因果关系', '因果关系是不存在', '不存在直接的因果',
        '没有直接的因果', '因果关系不存在', '没有明确的因果', '不存在因果', '无直接因果',
        'there is no implication of a causal relationship',
        'is not a causal story', 'is not a causal factor', '答案是无'
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
