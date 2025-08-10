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

Event A: My body cast a shadow over the grass.
Event B: The sun was rising.
Question: is there a causal relationship between Event A and Event B ?
Answer(yes or no with chain of thought): The shadow is mostly being cast by the speaker’s body. There must be a light source in the correct position to form the shadow. And the sun is the most plausible cause of the shadow. Thus, Event B may be the cause of Event A. Therefore, the answer is yes.

Event A: I hung up the phone.
Event B: The caller identified himself to me.
Question: is there a causal relationship between Event A and Event B ?
Answer(yes or no with chain of thought): People always hung up the phone after the ending of their conversation, while they always identify themselves at the beginning of the call. Therefore, the answer is no.

Event A: The cook stirred the ingredients in the bowl.
Event B: The ingredients melted.
Question: is there a causal relationship between Event A and Event B ?
Answer(yes or no with chain of thought): Stirring is a common method used in cooking to blend and mix ingredients. But melting ingredients always need high temperature, which can not be brought by stirring. Therefore, the answer is no.

Event A: The book became a huge bestseller.
Event B: It was adapted into a movie.
Question: is there a causal relationship between Event A and Event B ?
Answer(yes or no with chain of thought): When a book becomes a huge bestseller, it often attracts the attention of filmmakers and can lead to movie adaptations, and authors generally gain more recognition and fame. Thus, Event B may be the effect of Event A. Therefore, the answer is yes.

Event A: The man anticipated cold weather on his trip.
Event B: He travelled with a big suitcase.
Question: is there a causal relationship between Event A and Event B ?
Answer(yes or no with chain of thought): When someone expects cold weather, they may take some warm clothes or other things to keep warm. But it is not logical for them to take a big suitcase. Therefore, the answer is no.

Event A: I turned on the fan.
Event B: I felt cool air pass over me.
Question: is there a causal relationship between Event A and Event B ?
Answer(yes or no with chain of thought): A typical function of a fan is to circulates air and creates a cooling effect. Thus, Event B may be the effect of Event A. Therefore, the answer is yes.

Event A: The woman struggled to walk.
Event B: She wore high heels.
Question: is there a causal relationship between Event A and Event B ?
Answer(yes or no with chain of thought): High heels can be uncomfortable and challenging to walk in for some individual. Therefore, Event B may be the cause of Event A. Therefore, the answer is yes.

Event A: I vacuumed the carpet.
Event B: My roommate spilled punch.
Question: is there a causal relationship between Event A and Event B ?
Answer(yes or no with chain of thought): Vacuum cleaners generally can't handle liquids like punch. Therefore, the answer is no.

Event A: %s
Event B: %s
Question: is there a causal relationship between Event A and Event B ?
Answer (Yes or No ?): """,
    'manual-CoT-CN':
    """如下为三个使用思维链进行推理的问题:

事件一：那个女孩许了一个愿望。
事件二：她看到了一只黑猫。
问题：事件一和事件二之间是否存在因果关系？
答案（是或否？）：看到一只黑猫通常不会导致人们许愿，因此答案是“否”。

事件一：龙卷风袭击了这座城镇。
事件二：法院大楼的屋顶被吹掉了。
问题：事件一和事件二之间是否存在因果关系？
答案（是或否？）：龙卷风通常会带来强风，破坏建筑物，因此答案是“是”。

事件一：商店收银员叫保安了。
事件二：客户使用了假钞。
问题：事件一和事件二之间是否存在因果关系？
答案（是或否？）：商店收银员叫保安通常是因为有可疑和异常情况，包括客户用假钞，因此答案是“是”。

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
