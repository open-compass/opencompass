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

Input: The events of the Dismissal led to only minor constitutional change .
Question: is there a causal relationship between "events" and "change" ?
Answer (Yes or No ?): The word "led to" indicates a causal relationship between "events" and "change." Thus, the answer is Yes.

Input: They went into hiding secretly gaining support and strength .
Question: is there a causal relationship between "went" and "gaining" ?
Answer (Yes or No ?): "Went into hiding" indicates an action of seeking refuge or concealing oneself, which is separate from the action of "gaining support and strength." Thus, the answer is No.

Input: On 7 January , the number of houses destroyed throughout the affected area was revised down from 38 to 32 and again down to 27 a few days later .
Question: is there a causal relationship between "destroyed" and "affected" ?
Answer (Yes or No ?): Throughout the affected area" suggests that the destruction of houses is a consequence of some event or situation that has affected the area. Thus, the answer is Yes.

Input: There were both independent and signed bands who were booked to play , as well as many vendors for music and related paraphernalia .
Question: is there a causal relationship between "signed" and "play" ?
Answer (Yes or No ?): Both independent and signed bands were booked to play at the event. Thus, the answer is No.

Input: Strong winds lashed North Florida , with sustained winds of 125 mph ( 205 km/h ) observed in St. Augustine .
Question: is there a causal relationship between "lashed" and "observed" ?
Answer (Yes or No ?): "Lashed" and "observed” are describing different aspects of the weather conditions, but are not causally linked. Thus, the answer is No.

Input: In Thailand , the system produced significant storm surge , damaged or destroyed 1,700 homes , and killed two people .
Question: is there a causal relationship between "storm" and "damaged" ?
Answer (Yes or No ?): The storm in Thailand produced a significant storm surge, causing damage to or destruction of 1,700 homes. Thus, the answer is Yes.

Input: Valencia , meanwhile , defeated English sides Arsenal and Leeds United in the knockout phase en route to the final .
Question: is there a causal relationship between "defeated" and "final" ?
Answer (Yes or No ?): Valencia defeated English sides Arsenal and Leeds United in the knockout phase of the competition, and as a result of these victories, they progressed or advanced to the final. Thus, the answer is Yes.

Input: Arnold was injured early in the attack , and Morgan led the assault in his place before he became trapped in the lower city and was forced to surrender .
Question: is there a causal relationship between "forced" and "injured" ?
Answer (Yes or No ?): Arnold being injured early in the attack, and later, Morgan being forced to surrender in the lower city, these two events are not causally connected. Thus, the answer is No.

Input: %s
Question: is there a causal relationship between \"%s\" and \"%s\" ?
Answer (Yes or No ?):""",
    'manual-CoT-CN':
    """如下为三个使用思维链进行推理的问题:

输入：然而，当塔赫马斯普目睹一个强大帝国曾经辉煌的首都的贫困和悲惨状况时，他泪流满面。
问题：\"泪流满面\"和\"眼泪\"之间是否存在因果关系？
答案（是或否？）：塔赫马斯普的泪流满面是因为他目睹了首都的贫困和悲惨状况。因此答案为“是”。

输入：1811年11月29日的行动是拿破仑战争亚得里亚海战役期间，两个护卫舰中队在亚得里亚海上进行的一次小型海军交战。
问题：\"交战\"和\"战役\"之间是否存在因果关系？
答案（是或否？）："交战"和"战役"都涉及到军事行动，但它们不一定具有因果关系。因此答案为“否”。

输入：阿富汗队在决赛中的风云人物拉希德·汗说，赢得洲际杯对我们来说是测试板球的良好准备”。
问题：\"赢得\"和\"准备\"之间是否存在因果关系？
答案（是或否？）：他们赢得洲际杯是为了准备测试板球比赛，因此可以认为赢得比赛导致了他们的准备行动。因此答案为“是”。

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
