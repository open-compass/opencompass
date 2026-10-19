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
    'zero-shot-IcL': """Answer questions about explaining away effect.
Input Info: %s
Question: %s
Answer (Yes or No ?):""",
    'zero-shot-IcL-CN': """请回答关于相消解释作用的问题。
输入信息：%s
问题：%s
答案（是或否？）：""",
    'one-shot-IcL': """Answer questions about explaining away effect.
Input Info: The overall probability of attractive appearance is 48%%. For people considered unattractive and are not famous, the probability of talent is 3%%. For people considered unattractive and are famous, the probability of talent is 9%%. For people considered attractive and are not famous, the probability of talent is 2%%. For people considered attractive and are famous, the probability of talent is 6%%.
Question: If we look at people who are famous, does the chance of talent increase when attractive appearance?
Answer (Yes or No ?):No.

Input Info: %s
Question: %s
Answer (Yes or No ?):""",
    'one-shot-IcL-CN': """请回答关于相消解释作用的问题。
输入信息：拥有迷人外表的总体概率是48%%。对于被认为外表不迷人且不出名的人来说，有天赋的概率是3%%。对于被认为没有外表不迷人但很出名的人来说，有天赋的概率是9%%。对于被认为外表迷人但不出名的人来说，有天赋的概率是2%%。对于被认为外表迷人且出名的人来说，有天赋的概率是6%%。
问题：如果我们观察那些出名的人，当他们拥有迷人的外表时，其拥有天赋的概率会增加吗？
答案（是或否？）：否

输入信息：%s
问题：%s
答案（是或否？）：""",
    'three-shot-IcL': """Answer questions about explaining away effect.
Input Info: The overall probability of attractive appearance is 48%%. For people considered unattractive and are not famous, the probability of talent is 3%%. For people considered unattractive and are famous, the probability of talent is 9%%. For people considered attractive and are not famous, the probability of talent is 2%%. For people considered attractive and are famous, the probability of talent is 6%%.
Question: If we look at people who are famous, does the chance of talent increase when attractive appearance?
Answer (Yes or No ?):No.

Input Info: The overall probability of attractive appearance is 56%%. For people considered unattractive and are not famous, the probability of talent is 7%%. For people considered unattractive and are famous, the probability of talent is 18%%. For people considered attractive and are not famous, the probability of talent is 4%%. For people considered attractive and are famous, the probability of talent is 14%%.
Question: If we look at people who are famous, does the chance of talent increase when attractive appearance?
Answer (Yes or No ?): no

Input Info: The overall probability of talent is 59%%. For students who are not talented and rejected from elite institutions, the probability of being hard-working is 37%%. For students who are not talented and accepted to elite institutions, the probability of being hard-working is 73%%. For students who are talented and rejected from elite institutions, the probability of being hard-working is 30%%. For students who are talented and accepted to elite institutions, the probability of being hard-working is 63%%.
Question: If we look at students accepted to elite institutions, does the chance of being hard-working decrease when talent?
Answer (Yes or No ?): yes

Input Info: %s
Question: %s
Answer (Yes or No ?):""",
    'three-shot-IcL-CN': """请回答关于相消解释作用的问题。
输入信息：拥有迷人外表的总体概率是48%%。对于被认为外表不迷人且不出名的人来说，有天赋的概率是3%%。对于被认为没有外表不迷人但很出名的人来说，有天赋的概率是9%%。对于被认为外表迷人但不出名的人来说，有天赋的概率是2%%。对于被认为外表迷人且出名的人来说，有天赋的概率是6%%。
问题：如果我们观察那些出名的人，当他们拥有迷人的外表时，其拥有天赋的概率会增加吗？
答案（是或否？）：否

输入信息：拥有迷人外表的总体概率是56%%。对于被认为外表不迷人且不出名的人来说，有天赋的概率是7%%。对于被认为没有外表不迷人但很出名的人来说，有天赋的概率是18%%。对于被认为外表迷人但不出名的人来说，有天赋的概率是4%%。对于被认为外表迷人且出名的人来说，有天赋的概率是14%%。
问题：如果我们观察那些出名的人，当他们拥有迷人的外表时，其拥有天赋的概率会增加吗？
答案（是或否？）：否

输入信息：有天赋的总体概率是59%%。对于没有天赋并被精英学校拒之门外的学生来说，努力工作的概率是37%%。对于没有天赋却被精英学校录取的学生来说，努力工作的概率是73%%。对于有天赋却被精英学校拒之门外的学生来说，努力工作的概率是30%%。对于有天赋并被精英学校录取的学生来说，努力工作的概率是63%%。
问题：如果我们观察那些被精英学校录取的学生，当他们有天赋时，其努力工作的概率会降低吗？
答案（是或否？）：是

输入信息：%s
问题：%s
答案（是或否？）：""",
    'zero-shot-CoT': """Input Info: %s
Question: %s Let's think step by step.
Answer (Yes or No ?):""",
    'zero-shot-CoT-CN': """输入信息：%s
问题：%s请逐步思考。
答案（是或否？）：""",
    'manual-CoT':
    """Here are three examples of problems about explaining away effect with chain of thought.

Input Info: The overall probability of attractive appearance is 81%%. For people considered unattractive and are not famous, the probability of talent is 98%%. For people considered unattractive and are famous, the probability of talent is 92%%. For people considered attractive and are not famous, the probability of talent is 97%%. For people considered attractive and are famous, the probability of talent is 86%%.
Question: If we look at people who are famous, does the chance of talent increase when attractive appearance?
Answer (Yes or No ?): Let Y = talent; X = appearance; V3 = fame. The causal relations in this scenario are: X->V3,Y->V3. According to the question, we have: P(X=1) = 0.81\nP(Y=1 | X=0, V3=0) = 0.98\nP(Y=1 | X=0, V3=1) = 0.92\nP(Y=1 | X=1, V3=0) = 0.97\nP(Y=1 | X=1, V3=1) = 0.86. Calculate P(Y = 1 | X = 1, V3 = 1] - P(Y = 1 | V3 = 1)=P(Y=1 | X=1, V3=1) - (P(X=1) * P(Y=1 | X=1, V3=1) + P(X=0) * P(Y=1 | X=0, V3=1))=0.86 - (0.81*0.86 + 0.19*0.92) = -0.03<0. Thus, if we look at people who are famous, the chance of talent does not increase when attractive appearance. Therefore, the answer is No.

Input Info: The overall probability of speaking english is 96%%. For people who do not speak english and are not famous, the probability of talent is 98%%. For people who do not speak english and are famous, the probability of talent is 95%%. For people who speak english and are not famous, the probability of talent is 98%%. For people who speak english and are famous, the probability of talent is 93%%.
Question: If we look at people who are famous, does the chance of talent decrease when speaking english?
Answer (Yes or No ?): Let Y = talent; X = ability to speak english; V3 = fame. The causal relations in this scenario are: X->V3,Y->V3. According to the question, we have: P(X=1) = 0.96\nP(Y=1 | X=0, V3=0) = 0.98\nP(Y=1 | X=0, V3=1) = 0.95\nP(Y=1 | X=1, V3=0) = 0.98\nP(Y=1 | X=1, V3=1) = 0.93. Calculate P(Y = 1 | X = 1, V3 = 1] - P(Y = 1 | V3 = 1)=P(Y=1 | X=1, V3=1) - (P(X=1) * P(Y=1 | X=1, V3=1) + P(X=0) * P(Y=1 | X=0, V3=1))=0.93 - (0.96*0.93 + 0.04*0.95) = -0.00=0. Thus, if we look at people who are famous, the chance of talent decreases when speaking english. Therefore, the answer is Yes.

Input Info: The overall probability of talent is 82%%. For students who are not talented and rejected from elite institutions, the probability of brown eyes is 99%%. For students who are not talented and accepted to elite institutions, the probability of brown eyes is 82%%. For students who are talented and rejected from elite institutions, the probability of brown eyes is 96%%. For students who are talented and accepted to elite institutions, the probability of brown eyes is 53%%.
Question: If we look at students accepted to elite institutions, does the chance of brown eyes increase when talent?
Answer (Yes or No ?): Let Y = brown eyes; X = talent; V3 = elite institution admission status. The causal relations in this scenario are: X->V3,Y->V3. According to the question, we have: P(X=1) = 0.82\nP(Y=1 | X=0, V3=0) = 0.99\nP(Y=1 | X=0, V3=1) = 0.82\nP(Y=1 | X=1, V3=0) = 0.96\nP(Y=1 | X=1, V3=1) = 0.53. Calculate P(Y = 1 | X = 1, V3 = 1] - P(Y = 1 | V3 = 1)=P(Y=1 | X=1, V3=1) - (P(X=1) * P(Y=1 | X=1, V3=1) + P(X=0) * P(Y=1 | X=0, V3=1))=0.53 - (0.82*0.53 + 0.18*0.82) = -0.12<0. Thus, if we look at students accepted to elite institutions, the chance of brown eyes does not increase when talent. Therefore, the answer is No.

Input Info: %s
Question: %s
Answer (Yes or No ?):""",
    'manual-CoT-CN': """如下为一个使用思维链进行推理的有关解释移除效应的问题：

输入信息：呼吸系统有问题的总体概率是49%%。对于呼吸系统没有问题且未住院的人来说，骨折的概率是15%%。对于呼吸系统没有问题但住院的人来说，骨折的概率是31%%。对于呼吸系统有问题但未住院的人来说，骨折的概率是7%%。对于呼吸系统有问题且已住院的人来说，骨折的概率为27%%。
问题：如果我们观察那些住院患者，当他们呼吸系统出现问题时，其骨折的概率会降低吗？
答案（是或否？）：令 Y = 骨折; X = 呼吸系统问题; V3 = 住院状况; 该问题下的因果关系有： X->V3,Y->V3。由题目信息可知：P(X=1) = 0.49\nP(Y=1 | X=0, V3=0) = 0.15\nP(Y=1 | X=0, V3=1) = 0.31\nP(Y=1 | X=1, V3=0) = 0.07\nP(Y=1 | X=1, V3=1) = 0.27。计算P(Y = 1 | X = 1, V3 = 1] - P(Y = 1 | V3 = 1)=P(Y=1 | X=1, V3=1) - (P(X=1) * P(Y=1 | X=1, V3=1) + P(X=0) * P(Y=1 | X=0, V3=1))=0.27 - (0.49*0.27 + 0.51*0.31) = -0.01<0。因此，如果我们观察那些住院患者，当他们呼吸系统出现问题时，其骨折的概率会降低。因此答案为“是”。

输入信息：%s
问题：%s
答案（是或否？）：""",
    'explicit-function':
    """You are a helpful assistant for explaining away effect.
Input Info: %s
Question: %s
Answer (Yes or No ?):""",
    'explicit-function-CN': """你是一个用于评估相消解释效应的得力助手。
输入信息：%s
问题：%s
答案（是或否？）：""",
}


def get_prompt(task_name, prompt_style, item, prompt_style_str=''):
    base = base_prompt_dict[prompt_style]

    prompt = prompt_style_str + base % (item['given_info'], item['question'])
    return prompt
