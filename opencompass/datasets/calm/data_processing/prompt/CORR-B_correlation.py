# flake8: noqa: E501
base_prompt_dict = {
    'basic': """Input Info: %s
Question: %s
Answer (Yes or No ?):""",
    'basic-CN': """输入信息：%s
问题：%s
答案（是或否？）：""",
    'adversarial-ignore': """Input Info: %s
Question: %s
Answer (Yes or No ?):""",
    'adversarial-ignore-CN': """输入信息：%s
问题：%s
答案（是或否？）：""",
    'adversarial-doubt': """Input Info: %s
Question: %s
Answer (Yes or No ?):""",
    'adversarial-doubt-CN': """输入信息：%s
问题：%s
答案（是或否？）：""",
    'zero-shot-IcL': """Answer questions about correlation.
Input Info: %s
Question: %s
Answer (Yes or No ?):""",
    'zero-shot-IcL-CN': """回答有关相关性的问题。
输入信息：%s
问题：%s
答案（是或否？）：""",
    'one-shot-IcL': """Answer questions about correlation.
Input Info: The overall probability of alarm set by husband is 0.74. The probability of alarm not set by husband and ringing alarm is 0.09. The probability of alarm set by husband and ringing alarm is 0.51.
Question: Is the chance of ringing alarm smaller when observing alarm set by husband?
Answer (Yes or No ?): No.

Input Info: %s
Question: %s
Answer (Yes or No ?):""",
    'one-shot-IcL-CN': """回答有关相关性的问题。
输入信息：丈夫设置闹钟的总体概率为74%%，丈夫未设置闹钟而闹钟响起的概率为9%%，丈夫设置闹钟且闹钟响起的概率为51%%。
问题：观察到丈夫设置闹钟是否会降低闹钟响铃的概率？
答案（是或否？）：否

输入信息：%s
问题：%s
答案（是或否？）：""",
    'three-shot-IcL': """Answer questions about correlation.
Input Info: The overall probability of alarm set by husband is 0.74. The probability of alarm not set by husband and ringing alarm is 0.09. The probability of alarm set by husband and ringing alarm is 0.51.
Question: Is the chance of ringing alarm smaller when observing alarm set by husband?
Answer (Yes or No ?): No.

Input Info: The overall probability of alarm set by husband is 69%%. The probability of alarm not set by husband and ringing alarm is 15%%. The probability of alarm set by husband and ringing alarm is 38%%.
Question: Is the chance of ringing alarm larger when observing alarm set by husband?
Answer (Yes or No ?): yes

Input Info: The overall probability of alarm set by husband is 86%%. The probability of alarm not set by husband and ringing alarm is 7%%. The probability of alarm set by husband and ringing alarm is 71%%.
Question: Is the chance of ringing alarm larger when observing alarm set by husband?
Answer (Yes or No ?): yes

Input Info: %s
Question: %s
Answer (Yes or No ?):""",
    'three-shot-IcL-CN': """回答有关相关性的问题。
输入信息：丈夫设置闹钟的总体概率为74%%，丈夫未设置闹钟而闹钟响起的概率为9%%，丈夫设置闹钟且闹钟响起的概率为51%%。
问题：观察到丈夫设置闹钟是否会降低闹钟响铃的概率？
答案（是或否？）：否

输入信息：丈夫设置闹钟的总体概率为69%%，丈夫未设置闹钟而闹钟响起的概率为15%%，丈夫设置闹钟且闹钟响起的概率为38%%。
问题：观察到丈夫设置闹钟是否会增加闹钟响铃的概率？
答案（是或否？）：是

输入信息：丈夫设置闹钟的总体概率为86%%，丈夫未设置闹钟而闹钟响起的概率为7%%，丈夫设置闹钟且闹钟响起的概率为71%%。
问题：观察到丈夫设置闹钟是否会增加闹钟响铃的概率？
答案（是或否？）：是

输入信息：%s
问题：%s
答案（是或否？）：""",
    'zero-shot-CoT': """Input Info: %s
Question: %s Let's think step by step.
Answer (Yes or No ?):""",
    'zero-shot-CoT-CN': """输入信息：%s
问题：%s 请逐步思考。
答案（是或否？）：""",
    'manual-CoT':
    """Here are three examples of problems about considering correlation with chain of thought.

Input Info: The overall probability of encouragement is 13%%. The probability of discouragement and high exam score is 24%%. The probability of encouragement and high exam score is 9%%.
Question: Is the chance of high exam score larger when observing encouragement?
Answer (Yes or No ?): Let X = encouragement level; V2 = studying habit; Y = exam score. The causal relations are: X->V2,X->Y,V2->Y. P(X=1=1) = 0.51\nP(Y=1, X=0=1) = 0.16\nP(Y=1, X=1=1) = 0.33. P(X = 1, Y = 1)/P(X = 1) - P(X = 0, Y = 1)/P(X = 0)=0.33/0.51 - 0.16/0.49 = 0.32>0. Thus, the chance of high exam score is larger when observing encouragement. Therefore, the answer is Yes.

Input Info: The overall probability of high hospital bill is 53%%. The probability of low hospital bill and recovery is 34%%. The probability of high hospital bill and recovery is 16%%.
Question: Is the chance of recovery larger when observing high hospital bill?
Answer (Yes or No ?): Let V1 = age; X = hospital costs; Y = recovery. The causal relations are: V1->X,V1->Y,X->Y. P(X=1=1) = 0.53\nP(Y=1, X=0=1) = 0.34\nP(Y=1, X=1=1) = 0.16. P(X = 1, Y = 1)/P(X = 1) - P(X = 0, Y = 1)/P(X = 0)=0.16/0.53 - 0.34/0.47 = -0.43<0. Thus, the chance of recovery is not larger when observing high hospital bill. Therefore, the answer is No.

Input Info: The overall probability of male gender is 7%%. The probability of non-male gender and freckles is 34%%. The probability of male gender and freckles is 3%%.
Question: Is the chance of freckles smaller when observing male gender?
Answer (Yes or No ?): Let V2 = residency status; X = gender; V3 = department competitiveness; Y = freckles. The causal relations are: X->V3,V2->V3,X->Y,V2->Y,V3->Y. P(X=1=1) = 0.07\nP(Y=1, X=0=1) = 0.34\nP(Y=1, X=1=1) = 0.03. P(X = 1, Y = 1)/P(X = 1) - P(X = 0, Y = 1)/P(X = 0)=0.03/0.07 - 0.34/0.93 = 0.03>0. Thus, the chance of freckles is not smaller when observing male gender. Therefore, the answer is No.

Input Info: %s
Question: %s
Answer (Yes or No ?):""",
    'manual-CoT-CN': """如下为三个使用思维链进行推理的有关统计关联程度的问题：

输入信息：丈夫设置闹钟的总体概率为86%%，丈夫未设置闹钟而闹钟响起的概率为7%%，丈夫设置闹钟且闹钟响起的概率为71%%。
问题：观察到丈夫设置闹钟是否会增加闹钟响铃的概率？
答案（是或否？）：令 X = 丈夫; V2 = 妻子; Y = 闹钟响。因果关系有：X->V2,X->Y,V2->Y。P(X=1=1) = 0.86\nP(Y=1, X=0=1) = 0.07\nP(Y=1, X=1=1) = 0.71。P(X = 1, Y = 1)/P(X = 1) - P(X = 0, Y = 1)/P(X = 0)=0.71/0.86 - 0.07/0.14 = 0.29>0。因此丈夫设置闹钟会增加闹钟响铃的概率。因此答案为“是”。

输入信息：进行美黑沙龙护理的总体概率为1%%，没有进行美黑沙龙护理但皮肤被晒黑的概率是22%%。进行美黑沙龙护理后皮肤被晒黑的概率为0%%。
问题：观察到进行美黑沙龙护理是否会增加皮肤被晒黑的概率？
答案（是或否？）：令 V2 = 去海滩; X = 美黑沙龙护理; Y = 皮肤。因果关系有：X->Y,V2->Y。P(X=1=1) = 0.01\nP(Y=1, X=0=1) = 0.22\nP(Y=1, X=1=1) = 0.00。P(X = 1, Y = 1)/P(X = 1) - P(X = 0, Y = 1)/P(X = 0)=0.00/0.01 - 0.22/0.99 = 0.56>0。因此进行美黑沙龙护理会增加皮肤被晒黑的概率。因此答案为“是”。

输入信息：乘坐电梯的总体概率为34%%。走楼梯导致企鹅死亡的概率为30%%。乘坐电梯导致企鹅死亡的概率为16%%。
问题：观察到乘坐电梯是否会降低企鹅死亡的概率？
答案（是或否？）：令 X = 我的决定; V2 = 企鹅的情绪; Y = 企鹅存活。因果关系有：X->V2,X->Y,V2->Y。P(X=1=1) = 0.34\nP(Y=1, X=0=1) = 0.30\nP(Y=1, X=1=1) = 0.16。P(X = 1, Y = 1)/P(X = 1) - P(X = 0, Y = 1)/P(X = 0)=0.35/0.60 - 0.23/0.40 = 0.01>0。因此乘坐电梯不会降低企鹅死亡的概率。因此答案为“否”。

输入信息：%s
问题：%s
答案（是或否？）：""",
    'explicit-function':
    """You are a helpful assistant for identifying correlation.
Input Info: %s
Question: %s
Answer (Yes or No ?):""",
    'explicit-function-CN': """你是一个识别相关关系的得力助手。
输入信息：%s
问题：%s
答案（是或否？）：""",
}


def get_prompt(task_name, prompt_style, item, prompt_style_str=''):
    base = base_prompt_dict[prompt_style]

    prompt = prompt_style_str + base % (item['given_info'], item['question'])
    return prompt
