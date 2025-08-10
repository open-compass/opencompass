from ..utils.function_utils import multi_choice_judge

"""
task: multiple choice classification
metric: accuracy
咨询分类
"""

def compute_zxfl(data_dict):
    """
    A reference (R) contains a list of options, each option is from the option_list.
    We will extract the options appearing in the prediction and convert them into a set (P).
    We compute the accuracy between the prediction (P) and the reference (R).
    """


    score_list, abstentions = [], 0
    option_list = ['婚姻家庭', '劳动纠纷', '交通事故', '债权债务', '刑事辩护', '合同纠纷', '房产纠纷', '侵权', '公司法', '医疗纠纷', '拆迁安置', '行政诉讼', '建设工程', '知识产权', '综合咨询', '人身损害', '涉外法律', '海事海商', '消费权益', '抵押担保']
    for example in data_dict:
        question, prediction, answer = example["origin_prompt"], example["prediction"], example["refr"]
        judge = multi_choice_judge(prediction, option_list, answer)
        score_list.append(judge["score"])
        abstentions += judge["abstention"]

    # compute the accuracy of score_list
    final_accuracy_score = sum(score_list) / len(score_list)
    return {'score': final_accuracy_score, 'abstention_rate': abstentions / len(data_dict)}
