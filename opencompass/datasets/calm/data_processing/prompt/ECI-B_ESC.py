# flake8: noqa: E501
base_prompt_dict = {
    'basic':
    """Input: %s
Question: is there a causal relationship between \"%s\" and \"%s\" ?
Answer (Yes or No ?):""",
    'basic-CN':
    """输入：%s
问题：\"%s\"和\"%s\"之间是否存在因果关系？
答案（是或否？）：""",
    'adversarial-ignore':
    """Input: %s
Question: is there a causal relationship between \"%s\" and \"%s\" ?
Answer (Yes or No ?):""",
    'adversarial-ignore-CN':
    """输入：%s
问题：\"%s\"和\"%s\"之间是否存在因果关系？
答案（是或否？）：""",
    'adversarial-doubt':
    """Input: %s
Question: is there a causal relationship between \"%s\" and \"%s\" ?
Answer (Yes or No ?):""",
    'adversarial-doubt-CN':
    """输入：%s
问题：\"%s\"和\"%s\"之间是否存在因果关系？
答案（是或否？）：""",
    'zero-shot-IcL':
    """determine whether there is a causal relationship between two events in a sentence.
Input: %s
Question: is there a causal relationship between \"%s\" and \"%s\" ?
Answer (Yes or No ?):""",
    'zero-shot-IcL-CN':
    """判断句子中的两个事件之间是否存在因果关系。
输入：%s
问题：\"%s\"和\"%s\"之间是否存在因果关系？
答案（是或否？）：""",
    'one-shot-IcL':
    """determine whether there is a causal relationship between two events in a sentence.
Input: The break in an undersea cable on that affected Seacom has been repaired
Question: is there a causal relationship between "affected" and "Seacom" ?
Answer (Yes or No ?):No.

Input: %s
Question: is there a causal relationship between \"%s\" and \"%s\" ?
Answer (Yes or No ?):""",
    'one-shot-IcL-CN':
    """判断句子中的两个事件之间是否存在因果关系。
输入：影响东南非洲海底光缆系统的海底电缆断裂处已经修复。
问题："影响"和"东南非洲海底光缆系统"之间是否存在因果关系？
答案（是或否？）：否

输入：%s
问题：\"%s\"和\"%s\"之间是否存在因果关系？
答案（是或否？）：""",
    'three-shot-IcL':
    """determine whether there is a causal relationship between two events in a sentence.
Input: The break in an undersea cable on that affected Seacom has been repaired
Question: is there a causal relationship between "affected" and "Seacom" ?
Answer (Yes or No ?):No.

Input: The actress , 26 , checked in late Thursday night , TMZ reports , barely making the deadline and dodging an arrest warrant .
Question: is there a causal relationship between "checked in" and "deadline" ?
Answer (Yes or No ?): Yes

Input: In a statement , the White House said it would do " whatever is necessary " to ensure compliance with the sanctions .
Question: is there a causal relationship between "ensure" and "do" ?
Answer (Yes or No ?): No.

Input: %s
Question: is there a causal relationship between \"%s\" and \"%s\" ?
Answer (Yes or No ?):""",
    'three-shot-IcL-CN':
    """判断句子中的两个事件之间是否存在因果关系。
输入：影响东南非洲海底光缆系统的海底电缆断裂处已经修复。
问题："影响"和"东南非洲海底光缆系统"之间是否存在因果关系？
答案（是或否？）：否

输入：据TMZ报道，这位26岁的女演员在周四深夜赶到法院报到，刚好赶上截止日期并避免了被逮捕。
问题："报到"和"截止日期"之间是否存在因果关系？
答案（是或否？）：是

输入：在一份声明中，白宫表示将采取’’一切必要手段”确保制裁得到遵守。
问题："确保"和"采取"之间是否存在因果关系？
答案（是或否？）：否

输入：%s
问题：\"%s\"和\"%s\"之间是否存在因果关系？
答案（是或否？）：""",
    'zero-shot-CoT':
    """Input: %s
Question: is there a causal relationship between \"%s\" and \"%s\" ? Let's think step by step.
Answer:""",
    'zero-shot-CoT-CN':
    """输入：%s
问题：\"%s\"和\"%s\"之间是否存在因果关系？请逐步思考。
答案（是或否？）：""",
    'manual-CoT':
    """Here we will provide eight chain-of-thought exemplars, followed by a causality identification question that needs to be answered.

Input: He was sentenced to life in prison for indecency with a child , aggravated sexual assault and two counts of aggravated assault with a deadly weapon , MyFoxHouston . com reports .
Question: is there a causal relationship between "indecency" and "sentenced" ?
Answer(Yes or No? With chain-of-thought): The other charges, such as "aggravated sexual assault" and "aggravated assault with a deadly weapon," also play a significant role in the severity of the sentence.  In this case, the causal relationship might not be direct, as the sentence could be the result of the cumulative impact of all the charges brought against the individual. Thus, the answer is No.

Input: A fire - bomb attack on a bank in Greece killed at least three people Wednesday as police fought pitched battles with striking protestors furious at brutal budget cuts designed to avert national bankruptcy .
Question: is there a causal relationship between "cuts" and "battles" ?
Answer(Yes or No? With chain-of-thought): The severe budget cuts, aimed at averting national bankruptcy, have triggered public anger and protests. These protests have, in turn, escalated into violent clashes with the police. Thus, the answer is Yes.

Input: “ Tonight was a peaceful vigil that devolved into a riot , ” Williams wrote .
Question: is there a causal relationship between "vigil" and "devolved" ?
Answer(Yes or No? With chain-of-thought): In this context, the transition from a vigil to a riot does not necessarily imply a direct causal relationship between the two. Rather, it indicates a shift or transformation from one type of event to another due to various factors. Thus, the answer is No.

Input: Klitschko finally landed a long , straight right in the fifth , and the round ended with Thompson struggling on the ropes .
Question: is there a causal relationship between "fifth" and "round" ?
Answer(Yes or No? With chain-of-thought): There isn't a direct causal relationship between the "fifth" and the "round." The numbering of the round doesn't inherently cause an action or event; it merely designates the order. Thus, the answer is No.

Input: Lyons said that Comeaux used a wheelchair in prison – he "claimed it was necessary for his mobility" – and added , "Since he fled on foot , that's obviously in question . "
Question: is there a causal relationship between "question" and "fled" ?
Answer(Yes or No? With chain-of-thought): The "question" about the necessity of Comeaux's wheelchair is directly caused by the fact that he "fled on foot," which contradicts his claim of needing the wheelchair for mobility. Thus, the answer is Yes.

Input: Man charged with arson over Waitrose fire in Wellington
Question: is there a causal relationship between "arson" and "fire" ?
Answer(Yes or No? With chain-of-thought): The term "arson" inherently involves causing a fire intentionally. The act of arson is directly causal to the fire that results from it. Thus, the answer is Yes.

Input: On Friday , 36 - year - old Duncan Raite died after slipping and falling about 60 metres ( 200 feet ) from a ridge .
Question: is there a causal relationship between "slipping" and "falling" ?
Answer(Yes or No? With chain-of-thought): When Duncan Raite slipped, he lost his footing or traction, which directly led to the effect of falling. The fall was a direct consequence of the slip. Thus, the answer is Yes.

Input: Riots over harsh new austerity measures left three bank workers dead and engulfed the streets of Athens on Wednesday , as angry protesters tried to storm parliament , hurled Molotov cocktails at police and torched buildings .
Question: is there a causal relationship between "storm" and "tried" ?
Answer(Yes or No? With chain-of-thought): "Tried" doesn't directly cause "storm"; instead, "tried" describes the initial intention or effort that leads to the subsequent action of "storming." Thus, the answer is No.

Input: %s
Question: is there a causal relationship between \"%s\" and \"%s\" ?
Answer (Yes or No ?):""",
    'manual-CoT-CN':
    """如下为三个使用思维链进行推理的问题:

输入：陪审团展示了一个可怕的时刻，一位无辜的母亲在接女儿放学时遭遇帮派袭击，在保护孩子时被枪杀
问题：\"展示\”和\”袭击\”之间是否存在因果关系？
答案（是或否？）：“陪审团展示”和“母亲被袭击”之间没有因果关系，因此答案为“否”。

输入：警长副手被枪击致死
问题：\”枪击\”和\”致死\”之间是否存在因果关系？
答案（是或否？）：枪击是导致警长副手死亡的原因，因此答案为“是”。

输入：地震真的很强烈，人们惊慌失措地涌上街头。
问题：\"地震\"和\"惊慌\"之间是否存在因果关系？
答案（是或否？）：强烈的“地震”导致了人们“惊慌”的情绪，因此答案为“是”。

输入：%s
问题：\"%s\"和\"%s\"之间是否存在因果关系？
答案（是或否？）：""",
    'explicit-function':
    """You are a helpful assistant for event causality identification.
Input: %s
Question: is there a causal relationship between \"%s\" and \"%s\" ?
Answer (Yes or No ?):""",
    'explicit-function-CN':
    """你是一个识别事件因果关系的得力助手。
输入：%s
问题：\"%s\"和\"%s\"之间是否存在因果关系？
答案（是或否？）：""",
}


def get_prompt(task_name, prompt_style, item, prompt_style_str=''):
    base = base_prompt_dict[prompt_style]
    if prompt_style in [
            'basic', 'adversarial-ignore', 'adversarial-doubt',
            'zero-shot-CoT', 'manual-CoT', 'zero-shot-IcL', 'one-shot-IcL',
            'three-shot-IcL', 'explicit-function'
    ]:
        words = item['words']
        sent = ' '.join(words)
        events = item['events']
        event1 = ' '.join([words[t] for t in events[0]])
        event2 = ' '.join([words[t] for t in events[1]])
        prompt = prompt_style_str + base % (sent, event1, event2)
    else:
        prompt = prompt_style_str + base % (item['sent'], item['event1'],
                                            item['event2'])

    return prompt
