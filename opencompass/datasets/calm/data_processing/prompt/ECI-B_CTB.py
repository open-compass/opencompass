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
    """Here we will provide eight chain-of-thought exemplars, followed by a causality identification question that needs to be answered with chain-of-thought.

Input: The truck maker said the significant drop in net income will result in lower earnings for the fiscal year .
Question: is there a causal relationship between "earnings" and "drop" ?
Answer(Yes or No? With chain-of-thought): The term "drop" indicates a decrease or reduction in something, which in this case is the net income. Net income is directly related to earnings, as it represents the amount of profit left after deducting expenses from revenue. Thus, the answer is Yes.

Input: said it plans to aggressively discount its major beer brands , setting the stage for a potentially bruising price war as the maturing industry 's growth continues to slow .
Question: is there a causal relationship between "continues" and "setting" ?
Answer(Yes or No? With chain-of-thought): The term "setting the stage" suggests preparing or creating a context for something. The word "continues" refers to the ongoing slowing down of the industry's growth. The slowing growth of the industry isn't directly a result of the company's action of setting the stage for a price war. Thus, the answer is No.

Input: The charges were offset in part by a gain from the sale of the company 's construction division .
Question: is there a causal relationship between "sale" and "gain" ?
Answer(Yes or No? With chain-of-thought): The term "gain" suggests a positive financial outcome or benefit. The sale of the construction division directly leads to the gain mentioned. The act of selling the construction division causes or results in the gain mentioned. Thus, the answer is Yes.

Input: The Atlanta-based airline , the third largest in the U.S., attributed the increase to higher passenger traffic , new international routes and reduced service by rival Eastern Airlines , which is in bankruptcy proceedings in the wake of a strike that began last spring .
Question: is there a causal relationship between "began" and "increase" ?
Answer(Yes or No? With chain-of-thought): The strike likely led to disruptions in Eastern Airlines' services. The text mentions that the Atlanta-based airline attributed an increase to "reduced service by rival Eastern Airlines." The strike (began) caused disruptions in Eastern Airlines' services, which in turn could have caused an increase in passenger traffic for the Atlanta-based airline. Thus, the answer is Yes.

Input: We in fact have seen hate group numbers dropping through the nineties , uh but this year they jumped up uh twenty percent , quite a dramatic rise .
Question: is there a causal relationship between "jumped" and "seen" ?
Answer(Yes or No? With chain-of-thought): The term "jumped up" indicates a sudden increase in hate group numbers. The term "seen" suggests that the speaker has observed this increase. The act of seeing the increase (jumped) doesn't directly cause the increase itself. Thus, the answer is No.

Input: Bertin Nadeau , newly appointed chairman and interim chief executive of Provigo , would n't say if Mr. Lortie was asked to leave .
Question: is there a causal relationship between "leave" and "asked" ?
Answer(Yes or No? With chain-of-thought): Bertin Nadeau wouldn't confirm whether Mr. Lortie was asked to leave. The actions here are Mr. Lortie leaving and Mr. Lortie being asked to leave. The act of Mr. Lortie leaving isn't directly caused by him being asked to leave. Thus, the answer is No.

Input: The Kearny , N.J.-based maker of hair accessories and other cosmetic products said it cut the dividend due to its third-quarter loss of $ 992,000 , or 15 cents a share .
Question: is there a causal relationship between "loss" and "said" ?
Answer(Yes or No? With chain-of-thought): Losses can negatively impact a company's finances, reducing the funds available for distribution to shareholders (dividends). The term "said" indicates the company's communication about the dividend cut. The financial loss (loss) led to the company's decision to cut its dividend, which is the reason behind the communication (said) about the dividend cut. Thus, the answer is Yes.

Input: Officials said the president himself met with Sununu Sunday .
Question: is there a causal relationship between "met" and "said" ?
Answer(Yes or No? With chain-of-thought):  The president met with Sununu on Sunday. Said" introduces the report or statement about the meeting, but it's not the cause of the meeting itself. The act of "saying" isn't the cause of the president "meeting" with Sununu. Thus, the answer is No.

Input: %s
Question: is there a causal relationship between \"%s\" and \"%s\" ?
Answer (Yes or No ?):""",
    'manual-CoT-CN':
    """如下为三个使用思维链进行推理的问题:

输入：Sloan股份有限公司表示，该公司聘请了一家投资银行公司协助评估重组或合并方案，并报告截至9月30日的第三季度净亏损810万美元，即每股214美元。
问题：\”协助\”和\”报道\”之间是否存在因果关系？
答案（是或否？）：“协助”和“报道”表示投资银行公司的两个不同的动作，“报道”是“协助”的后续动作，但并没有因果关系。因此答案为“否”。

输入：该公司表示，由于服务中心减少了库存、汽车市场低迷以及建筑市场竞争加剧等原因，导致其出货量下降。
问题：\"低迷\"和\"减少\"之间是否存在因果关系？
答案（是或否？）：“服务中心减少了库存”是“出货量下降”的原因之一，因此答案为“是”。

输入：伦敦股市在纽约隔夜下跌以及洛威辞职后英镑贬值的情况下，初期受到压制。
问题：\"下跌\"和\"压制\"之间是否存在因果关系？
答案（是或否？）：伦敦股市下跌和英镑贬值等因素是导致股市受到压制的原因之一。因此答案为“是”。

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
