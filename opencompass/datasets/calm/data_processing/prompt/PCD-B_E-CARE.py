# flake8: noqa: E501
base_prompt_dict = {
    'basic':
    """Event A: %s
Event B: %s
Question: is there a causal relationship between Event A and Event B ?
Answer (Yes or No ?):""",
    'basic-CN':
    """事件一：%s
事件二：%s
问题：事件一和事件二之间是否存在因果关系？
答案（是或否？）：""",
    'adversarial-ignore':
    """Event A: %s
Event B: %s
Question: is there a causal relationship between Event A and Event B ?
Answer (Yes or No ?):""",
    'adversarial-ignore-CN':
    """事件一：%s
事件二：%s
问题：事件一和事件二之间是否存在因果关系？
答案（是或否？）：""",
    'adversarial-doubt':
    """Event A: %s
Event B: %s
Question: is there a causal relationship between Event A and Event B ?
Answer (Yes or No ?):""",
    'adversarial-doubt-CN':
    """事件一：%s
事件二：%s
问题：事件一和事件二之间是否存在因果关系？
答案（是或否？）：""",
    'zero-shot-IcL':
    """determine whether there is a causal relationship between the two input events.
Event A: %s
Event B: %s
Question: is there a causal relationship between Event A and Event B ?
Answer (Yes or No ?):""",
    'zero-shot-IcL-CN':
    """确定两个输入事件之间是否存在因果关系。
事件一：%s
事件二：%s
问题：事件一和事件二之间是否存在因果关系？
答案（是或否？）：""",
    'one-shot-IcL':
    """determine whether there is a causal relationship between the two input events.
Event A: My body cast a shadow over the grass.
Event B: The sun was rising.
Question: is there a causal relationship between Event A and Event B ?
Answer (Yes or No ?): Yes
Event A: %s
Event B: %s
Question: is there a causal relationship between Event A and Event B ?
Answer (Yes or No ?):""",
    'one-shot-IcL-CN':
    """确定两个输入事件之间是否存在因果关系。
事件一：我的身体投下了阴影，落在草地上。
事件二：太阳正在升起。
问题：事件一和事件二之间是否存在因果关系？
答案（是或否？）：是
事件一：%s
事件二：%s
问题：事件一和事件二之间是否存在因果关系？
答案（是或否？）：""",
    'three-shot-IcL':
    """determine whether there is a causal relationship between the two input events.
Event A: My body cast a shadow over the grass.
Event B: The sun was rising.
Question: is there a causal relationship between Event A and Event B ?
Answer (Yes or No ?): Yes

Event A: The politician lost the election.
Event B: He ran negative campaign ads.
Question: is there a causal relationship between Event A and Event B ?
Answer (Yes or No ?): No

Event A: The physician misdiagnosed the patient.
Event B: The patient filed a malpractice lawsuit against the physician.
Question: is there a causal relationship between Event A and Event B ?
Answer (Yes or No ?): Yes

Event A: %s
Event B: %s
Question: is there a causal relationship between Event A and Event B ?
Answer (Yes or No ?):""",
    'three-shot-IcL-CN':
    """确定两个输入事件之间是否存在因果关系。
事件一：我的身体投下了阴影，落在草地上。
事件二：太阳正在升起。
问题：事件一和事件二之间是否存在因果关系？
答案（是或否？）：是

事件一：政治家在选举中落败了。
事件二：他播放了负面竞选广告。
问题：事件一和事件二之间是否存在因果关系？
答案（是或否？）：否

事件一：这位医生误诊了病人。
事件二：病人向医生提起了医疗事故诉讼。
问题：事件一和事件二之间是否存在因果关系？
答案（是或否？）：是

事件一：%s
事件二：%s
问题：事件一和事件二之间是否存在因果关系？
答案（是或否？）：""",
    'zero-shot-CoT':
    """Event A: %s
Event B: %s
Question: is there a causal relationship between Event A and Event B ? Let's think step by step.
Answer (Yes or No ?):""",
    'zero-shot-CoT-CN':
    """事件一：%s
事件二：%s
问题：事件一和事件二之间是否存在因果关系？请逐步思考。
答案（是或否？）：""",
    'manual-CoT':
    """Here we will provide eight chain-of-thought exemplars, followed by a binary question that needs to be answered.

Event A: Black's sweat always drips into his eyes.
Event B: Black pulled out his eyelashes for beauty.
Question: is there a causal relationship between Event A and Event B ?
Answer(Yes or No with chain of thought): If Black intentionally removed his eyelashes, it could potentially lead to sweat dripping into his eyes due to the lack of eyelashes to provide some protection. Therefore, the answer is yes.

Event A: It's half way through autumn.
Event B: It has difficulty in running.
Question: is there a causal relationship between Event A and Event B ?
Answer(Yes or No with chain of thought): Autumn is a season characterized by falling leaves and cooler temperatures. It doesn't inherently imply any difficulty in running. Therefore, the answer is no.

Event A: The man planned to make Tin by himself.
Event B: He had to design the necessary components.
Question: is there a causal relationship between Event A and Event B ?
Answer(Yes or No with chain of thought): While planning to make Tin might suggest that components need to be designed, the act of planning does not necessarily dictate that designing components is the only option. Therefore, the answer is no.

Event A:  He was shocked by his chemical deficiency.
Event B: The patient with addiction watched the neuroscientist's lecture.
Question: is there a causal relationship between Event A and Event B ?
Answer(Yes or No with chain of thought): Neuroscience gives a deep insight of chemical imbalances in one's body. He might come to realize the lack of some  nutrients in his body after the neuroscience lecture, thus felt shocked. Therefore, the answer is yes.

Event A:  Waterwheels started work efficiently.
Event B: The mills can set to work.
Question: is there a causal relationship between Event A and Event B ?
Answer(Yes or No with chain of thought): When the waterwheels are working efficiently, it enables the mills to start operating. Therefore, the answer is yes.

Event A: Mary has two pieces of farmland, but only one of them is used to grow crops every year.
Event B: The less often used farmland produces more crops than the often used one.
Question: is there a causal relationship between Event A and Event B ?
Answer(Yes or No with chain of thought): While it's possible that less frequent usage can contribute to better soil health and potentially higher yields, the quality of soil, weather conditions, irrigation practices, and crop choices could also influence the yield. Therefore, the answer is no.

Event A: Tom bought a lot of mangoes and coconuts.
Event B: Tom buys imported tropical fruit every day.
Question: is there a causal relationship between Event A and Event B ?
Answer(Yes or No with chain of thought): The fact that Tom bought mangoes and coconuts (Event A) doesn't necessarily indicate a consistent pattern or preference for buying imported tropical fruit (Event B) every day. Therefore, the answer is no.

Event A: He can just see something clearly in a short distance.
Event B: Tom turned on his flashlight.
Question: is there a causal relationship between Event A and Event B ?
Answer(Yes or No with chain of thought): Turning on a flashlight provides additional light in the immediate vicinity, making objects visible in a short distance. Therefore, the answer is yes.

Event A: %s
Event B: %s
Question: is there a causal relationship between Event A and Event B ?
Answer (Yes or No ?): """,
    'manual-CoT-CN':
    """如下为三个使用思维链进行推理的问题:

事件一：莎莉的喉咙严重发炎了。
事件二：莎莉发不出声音。
问题：事件一和事件二之间是否存在因果关系？
答案（是或否？）：喉咙严重发炎会导致声音嘶哑或失声，因此答案是“是”。

事件一：很多昆虫都被它们吃掉了。
事件二：果园里有很多麻雀。
问题：事件一和事件二之间是否存在因果关系？
答案（是或否？）：麻雀等鸟类喜欢吃昆虫，因此答案是“是”。

事件一：它具有糖酵解功能。
事件二：肌原纤维中含有不同数量的肌原丝。
问题：事件一和事件二之间是否存在因果关系？
答案（是或否？）：糖酵解功能和肌原丝数量之间没有直接的因果联系，因此答案是“否”。

事件一：%s
事件二：%s
问题：事件一和事件二之间是否存在因果关系？
答案（是或否？）：""",
    'explicit-function':
    """You are a helpful assistant for event causality identification.
Event A: %s
Event B: %s
Question: is there a causal relationship between Event A and Event B ?
Answer (Yes or No ?):""",
    'explicit-function-CN':
    """你是一个用于因果发现的得力助手。
事件一：%s
事件二：%s
问题：事件一和事件二之间是否存在因果关系？
答案（是或否？）：""",
}


def get_prompt(task_name, prompt_style, item, prompt_style_str=''):
    base = base_prompt_dict[prompt_style]

    prompt = prompt_style_str + base % (item['premise'], item['hypothesis'])
    return prompt
