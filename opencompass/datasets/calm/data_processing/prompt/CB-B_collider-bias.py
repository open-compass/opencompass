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
    'zero-shot-IcL': """Answer questions about collider bias.
Input Info: %s
Question: %s
Answer (Yes or No ?):""",
    'zero-shot-IcL-CN': """请回答有关碰撞偏见的问题。
输入信息：%s
问题：%s
答案（是或否？）：""",
    'one-shot-IcL': """Answer questions about collider bias.
Input Info: For people who are famous, the correlation between attractive appearance and talent is -0.08.
Question: If we look at people who are famous, does it mean that attractive appearance does not affect talent?
Answer (Yes or No ?):Yes.

Input Info: %s
Question: %s
Answer (Yes or No ?):""",
    'one-shot-IcL-CN': """请回答有关碰撞偏见的问题。
输入信息：对于那些出名的人来说，有吸引力的外表和才华之间的相关系数为-0.08。
问题：如果我们观察那些出名的人，这是否意味着有吸引力的外表不会影响才华?
答案（是或否？）：是

输入信息：%s
问题：%s
答案（是或否？）：""",
    'three-shot-IcL': """Answer questions about collider bias.
Input Info: For people who are famous, the correlation between attractive appearance and talent is -0.08.
Question: If we look at people who are famous, does it mean that attractive appearance does not affect talent?
Answer (Yes or No ?):Yes.

Input Info: For people who are famous, the correlation between attractive appearance and talent is -0.16.
Question: If we look at people who are famous, does it mean that attractive appearance does not affect talent?
Answer (Yes or No ?): yes

Input Info: For people who are famous, the correlation between attractive appearance and talent is -0.23.
Question: If we look at people who are famous, does it mean that attractive appearance affects talent?
Answer (Yes or No ?): no

Input Info: %s
Question: %s
Answer (Yes or No ?):""",
    'three-shot-IcL-CN': """请回答有关碰撞偏见的问题。
输入信息：对于那些出名的人来说，有吸引力的外表和才华之间的相关系数为-0.08。
问题：如果我们观察那些出名的人，这是否意味着有吸引力的外表不会影响才华?
答案（是或否？）：是

输入信息：对于那些出名的人来说，有吸引力的外表和才华之间的相关系数为-0.16。
问题：如果我们观察那些出名的人，这是否意味着有吸引力的外表不会影响才华?
答案（是或否？）：是

输入信息：对于那些出名的人来说，有吸引力的外表和才华之间的相关系数为-0.23。
问题：如果我们观察那些出名的人，这是否意味着有吸引力的外表会影响才华？
答案（是或否？）：否

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
    """Here are eight examples of problems with collider bias answered with chain of thought.

Input Info: For people who are famous, the correlation between attractive appearance and talent is -0.08.
Question: If we look at people who are famous, does it mean that attractive appearance does not affect talent?
Answer (Yes or No ?): Both attractive appearance and talent have direct effects on fame. This collision creates a spurious association between attractive appearance and talent when analysis is limited to famous people. Therefore, the answer is Yes.

Input Info: For hospitalized individuals, the correlation between respiratory issues and broken bones is -0.25.
Question: If we look at hospitalized individuals, does it mean that respiratory issues affects broken bones?
Answer (Yes or No ?): Both respiratory issues and broken bones affect hospitalization status. This collision creates a spurious association between respiratory issues and broken bones when analysis is limited to hospitalized individuals. Therefore, the answer is No.

Input Info: For students accepted to elite institutions, the correlation between listening to jazz and being hard-working is -0.06.
Question: If we look at students accepted to elite institutions, does it mean that listening to jazz does not affect being hard-working?
Answer (Yes or No ?): Both listening to jazz and effort affect elite institution admission status. This collision creates a spurious association between listening to jazz and hard-working when analysis is limited to students accepted to elite institutions. Therefore, the answer is Yes.

Input Info: For those who are yupt, the correlation between jyka and kwox is 0.02.
Question: If we look at those who are yupt, does it mean that jyka does not affect kwox?
Answer (Yes or No ?): Both jyka and kwox affect yupt. This collision creates a spurious association between jyka and kwox when analysis is limited to those who are yupt. Therefore, the answer is Yes.

Input Info: For those who are zupj, the correlation between yupt and muvq is -0.15.
Question: If we look at those who are zupj, does it mean that yupt affects muvq?
Answer (Yes or No ?): Both yupt and muvq affect zupj. This collision creates a spurious association between yupt and muvq when analysis is limited to those who are zupj. Therefore, the answer is No.

Input Info: For those who are swoq, the correlation between kwox and kwoz is -0.25.
Question: If we look at those who are swoq, does it mean that kwox affects kwoz?
Answer (Yes or No ?): Both kwox and kwoz affect swoq. This collision creates a spurious association between kwox and kwoz when analysis is limited to those who are swoq. Therefore, the answer is No.

Input Info: For those who are wibl, the correlation between zuph and uvzi is -0.01.
Question: If we look at those who are wibl, does it mean that zuph affects uvzi?
Answer (Yes or No ?): Both zuph and uvzi affect wibl. This collision creates a spurious association between zuph and uvzi when analysis is limited to those who are wibl. Therefore, the answer is No.

Input Info: For those who are jyka, the correlation between zuph and glimx is -0.04.
Question: If we look at those who are jyka, does it mean that zuph does not affect glimx?
Answer (Yes or No ?): Both zuph and glimx affect jyka. This collision creates a spurious association between zuph and glimx when analysis is limited to those who are jyka. Therefore, the answer is Yes.

Input Info: %s
Question: %s
Answer (Yes or No ?):""",
    'manual-CoT-CN': """如下为三个使用思维链进行推理的对撞偏差问题：

输入信息：对于那些出名的人来说，有吸引力的外表和才华之间的相关系数为-0.08。
问题：如果我们观察那些出名的人，这是否意味着有吸引力的外表不会影响才华?
答案（是或否？）：有吸引力的外表和才华都会影响名气。如果只分析出名的人，这些影响可能会造成有吸引力的外表和才华之间的虚假关系。因此答案为“是”。

输入信息：对于住院患者，呼吸问题与骨折之间的相关系数为-0.25。
问题：如果我们观察住院患者，这是否意味着呼吸问题会影响骨折？
答案（是或否？）：呼吸问题和骨折都会导致患者住院。如果只分析住院患者，这些影响可能会造成呼吸问题和骨折之间的虚假关系。因此答案为“否”。

输入信息：对于那些swoq的人来说，kwox和kwoz之间的相关系数为-0.25。
问题：如果我们观察那些swoq的人，这是否意味着kwox会影响kwoz？
答案（是或否？）：kwox和kwoz都会对swoq产生直接影响。如果只分析那些swoq的人，这些影响可能会造成kwox和kwoz之间的虚假关系。因此答案为“否”。

输入信息：%s
问题：%s
答案（是或否？）""",
    'explicit-function':
    """You are a helpful assistant for collider bias analysis.
Input Info: %s
Question: %s
Answer (Yes or No ?):""",
    'explicit-function-CN': """你是一个用于分析汇聚偏差的得力助手。
输入信息：%s
问题：%s
答案（是或否？）：""",
}


def get_prompt(task_name, prompt_style, item, prompt_style_str=''):
    base = base_prompt_dict[prompt_style]

    prompt = prompt_style_str + base % (item['given_info'], item['question'])
    return prompt
