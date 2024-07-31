# flake8: noqa: E501
base_prompt_dict = {
    'basic':
    """Input Event: If %s.
Question: Does %s cause %s?
Answer (Yes or No ?): """,
    'basic-CN':
    """输入信息：如果%s。
问题：%s是否导致%s？
答案（是或否？）：""",
    'adversarial-ignore':
    """Input Event: If %s.
Question: Does %s cause %s?
Answer (Yes or No ?): """,
    'adversarial-ignore-CN':
    """输入信息：如果%s。
问题：%s是否导致%s？
答案（是或否？）：""",
    'adversarial-doubt':
    """Input Event: If %s.
Question: Does %s cause %s?
Answer (Yes or No ?): """,
    'adversarial-doubt-CN':
    """输入信息：如果%s。
问题：%s是否导致%s？
答案（是或否？）：""",
    'zero-shot-IcL':
    """Answer questions based on causal relations in a given causal graph.
Input Event: If %s.
Question: Does %s cause %s?
Answer (Yes or No ?):""",
    'zero-shot-IcL-CN':
    """根据给定因果图中的因果关系回答问题。
输入信息：如果%s。
问题：%s是否导致%s？
答案（是或否？）：""",
    'one-shot-IcL':
    """Answer questions based on causal relations in a given causal graph.
Input Event: If A causes D, A causes E, B causes E, C causes D, and D causes E.
Question: Does C cause A?
Answer (Yes or No ?): No

Input Event: If %s.
Question: Does %s cause %s?
Answer (Yes or No ?):""",
    'one-shot-IcL-CN':
    """根据给定因果图中的因果关系回答问题。
输入信息：如果A导致D, A导致E, B导致E, C导致D, 以及D导致E。
问题：C是否导致A？
答案（是或否？）：否

输入信息：如果%s。
问题：%s是否导致%s？
答案（是或否？）：""",
    'three-shot-IcL':
    """Answer questions based on causal relations in a given causal graph.
Input Event: If A causes D, A causes E, B causes E, C causes D, and D causes E.
Question: Does C cause A?
Answer (Yes or No ?): No

Input Event: If A causes B, A causes E, B causes E, B causes D, C causes E, and C causes D.
Question: Does C cause D?
Answer (Yes or No ?): Yes

Input Event: If A causes D, A causes C, B causes E, C causes D, and D causes E.
Question: Does E cause E?
Answer (Yes or No ?): No

Input Event: If %s.
Question: Does %s cause %s?
Answer (Yes or No ?):""",
    'three-shot-IcL-CN':
    """根据给定因果图中的因果关系回答问题。
输入信息：如果A导致D, A导致E, B导致E, C导致D, 以及D导致E。
问题：C是否导致A？
答案（是或否？）：否

输入信息：如果A导致B, A导致E, B导致E, B导致D, C导致E, 以及C导致D。
问题：C是否导致D？
答案（是或否？）：是

输入信息：如果A导致D, A导致C, B导致E, C导致D, 以及D导致E。
问题：E是否导致E？
答案（是或否？）：否

输入信息：如果%s。
问题：%s是否导致%s？
答案（是或否？）：""",
    'zero-shot-CoT':
    """Input Event: If %s.
Question: Does %s cause %s? Let's think step by step.
Answer (Yes or No ?):""",
    'zero-shot-CoT-CN':
    """输入信息：如果%s。
问题：%s是否导致%s？请逐步思考。
答案（是或否？）：""",
    'manual-CoT':
    """Here are three examples of causal abstract reasoning using chain of thought, and a question to answer.

Input Event: If A causes D, A causes C, A causes B, B causes E, B causes D, and C causes D.
Question: Does A cause C?
Answer (Yes or No ?): The input states that A causes C. Therefore, the answer is Yes.

Input Event: If A causes B, A causes C, A causes D, B causes E, and C causes E.
Question: Does E cause A?
Answer (Yes or No ?): A is not caused by anything in the input, thus E does not cause A. Therefore, the answer is No.

Input Event: If A causes C, A causes H, A causes E, A causes B, B causes H, B causes G, B causes F, B causes C, C causes F, C causes H, C causes D, C causes E, D causes E, E causes G, E causes H, E causes F, F causes H, and F causes G.
Question: Does D cause F?
Answer (Yes or No ?): Given D causes E, and E causes F, such that D causes F. Therefore, the answer is Yes.

Input Event: If %s.
Question: Does %s cause %s?
Answer (Yes or No ?):
""",
    'manual-CoT-CN':
    """如下为三个使用思维链进行推理的因果抽象推理的示例，和一个需要回答的问题。

输入信息：A导致D, A导致C, A导致B, B导致E, B导致D, 以及C导致D。
问题：A是否导致C？
答案（是或否？）：输入信息中说明了A导致C。因此答案为“是”。

输入信息：A导致B, A导致C, A导致D, B导致E, 以及C导致E。
问题：E是否导致A？
答案（是或否？）：输入信息中没有任何元素导致A，因此E没有导致A。因此答案为“否”。

输入信息：如果A导致C, A导致H, A导致E, A导致B, B导致H, B导致G, B导致F, B导致C, C导致F, C导致H, C导致D, C导致E, D导致E, E导致G, E导致H, E导致F, F导致H, 以及F导致G。
问题：D是否导致F？
答案（是或否？）：D导致E，E导致F，所以D导致F。因此答案为“是”。

输入信息：如果%s。
问题：%s是否导致%s？
答案（是或否？）：
""",
    'explicit-function':
    """You are a helpful assistant for abstract reasoning.
Input Event: If %s.
Question: Does %s cause %s?
Answer (Yes or No ?):""",
    'explicit-function-CN':
    """你是一个用于抽象推理的得力助手。
输入信息：如果%s。
问题：%s是否导致%s？
答案（是或否？）：""",
}


def get_prompt(task_name, prompt_style, item, prompt_style_str=''):
    base = base_prompt_dict[prompt_style]

    prompt = prompt_style_str + base % (item['ar_edges'], item['former'],
                                        item['latter'])
    return prompt
