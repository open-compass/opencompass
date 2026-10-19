# flake8: noqa: E501
base_prompt_dict = {
    'basic':
    """Input Event: %s
Question: Please select the %s of the input event from the following options.
Option 1: %s
Option 2: %s
Answer (Option 1 or Option 2 ?):""",
    'basic-CN':
    """输入事件：%s
问题：请从以下选项中选择输入事件的%s。
选项一：%s
选项二：%s
答案（选项一或选项二？）：""",
    'adversarial-ignore':
    """Input Event: %s
Question: Please select the %s of the input event from the following options.
Option 1: %s
Option 2: %s
Answer (Option 1 or Option 2 ?):""",
    'adversarial-ignore-CN':
    """输入事件：%s
问题：请从以下选项中选择输入事件的%s。
选项一：%s
选项二：%s
答案（选项一或选项二？）：""",
    'adversarial-doubt':
    """Input Event: %s
Question: Please select the %s of the input event from the following options.
Option 1: %s
Option 2: %s
Answer (Option 1 or Option 2 ?):""",
    'adversarial-doubt-CN':
    """输入事件：%s
问题：请从以下选项中选择输入事件的%s。
选项一：%s
选项二：%s
答案（选项一或选项二？）：""",
    'zero-shot-IcL':
    """Select the cause or effect of the input event from two options.
Input Event: %s
Question: Please select the %s of the input event from the following options.
Option 1: %s
Option 2: %s
Answer (Option 1 or Option 2 ?):""",
    'zero-shot-IcL-CN':
    """从两个选项中选择输入事件的原因或结果。
输入事件：%s
问题：请从以下选项中选择输入事件的%s。
选项一：%s
选项二：%s
答案（选项一或选项二？）：""",
    'one-shot-IcL':
    """Select the cause or effect of the input event from two options.
Input Event: My body cast a shadow over the grass.
Question: Please select the cause of the input event from the following options.
Option 1: The sun was rising.
Option 2: The grass was cut.
Answer (Option 1 or Option 2 ?): Option 1
Input Event: %s
Question: Please select the %s of the input event from the following options.
Option 1: %s
Option 2: %s
Answer (Option 1 or Option 2 ?):""",
    'one-shot-IcL-CN':
    """从两个选项中选择输入事件的原因或结果。
输入事件：我的身体投下了阴影，落在草地上。
问题：请从以下选项中选择输入事件的原因。
选项一：太阳正在升起。
选项二：草被割了。
答案（选项一或选项二？）：选项一
输入事件：%s
问题：请从以下选项中选择输入事件的%s。
选项一：%s
选项二：%s
答案（选项一或选项二？）：""",
    'three-shot-IcL':
    """Select the cause or effect of the input event from two options.
Input Event: My body cast a shadow over the grass.
Question: Please select the cause of the input event from the following options.
Option 1: The sun was rising.
Option 2: The grass was cut.
Answer (Option 1 or Option 2 ?): Option 1

Input Event: The politician lost the election.
Question: Please select the cause of the input event from the following options.
Option 1: He ran negative campaign ads.
Option 2: No one voted for him.
Answer (Option 1 or Option 2 ?): Option 2

Input Event: The physician misdiagnosed the patient.
Question: Please select the effect of the input event from the following options.
Option 1: The patient filed a malpractice lawsuit against the physician.
Option 2: The patient disclosed confidential information to the physician.
Answer (Option 1 or Option 2 ?): Option 1

Input Event: %s
Question: Please select the %s of the input event from the following options.
Option 1: %s
Option 2: %s
Answer (Option 1 or Option 2 ?):""",
    'three-shot-IcL-CN':
    """从两个选项中选择输入事件的原因或结果。
输入事件：我的身体投下了阴影，落在草地上。
问题：请从以下选项中选择输入事件的原因。
选项一：太阳正在升起。
选项二：草被割了。
答案（选项一或选项二？）：选项一

输入事件：政治家在选举中落败了。
问题：请从以下选项中选择输入事件的原因。
选项一：他播放了负面竞选广告。
选项二：没有人投票给他。
答案（选项一或选项二？）：选项二

输入事件：这位医生误诊了病人。
问题：请从以下选项中选择输入事件的结果。
选项一：病人向医生提起了医疗事故诉讼。
选项二：患者向医生透露了机密信息。
答案（选项一或选项二？）：选项一

输入事件：%s
问题：请从以下选项中选择输入事件的%s。
选项一：%s
选项二：%s
答案（选项一或选项二？）：""",
    'zero-shot-CoT':
    """Input Event: %s
Question: Please select the %s of the input event from the following options. Let's think step by step.
Option 1: %s
Option 2: %s
Answer (Option 1 or Option 2 ?):""",
    'zero-shot-CoT-CN':
    """输入事件：%s
问题：请从以下选项中选择输入事件的%s。请逐步思考。
选项一：%s
选项二：%s
答案（选项一或选项二？）：""",
    'manual-CoT':
    """
Here we will provide eight chain-of-thought exemplars, where a few chain of thought demonstrations are provided as exemplars in prompting, followed by a question that needs to be answered.

Input Event: My body cast a shadow over the grass
Question: Please select the cause of the input event from the following options.
Option 1: The sun was rising
Option 2: The grass was cut.
Answer (Option 1 or Option 2 ?) with chain-of-thought:
The question is about the cause. The shadow is mostly being cast by the speaker’s body. There must be a light source in the correct position to form the shadow. Thus, the sun is the most plausible cause of the shadow. Therefore, the answer is Option 1: The sun was rising.

Input Event: I hung up the phone.
Question: Please select the cause of the input event from the following options.
Option 1: The caller said goodbye to me.
Option 2: The caller identified himself to me
Answer (Option 1 or Option 2 ?) with chain-of-thought:
The question is about the cause. People always hung up the phone after the ending of their conversation. People usually end a conversation by saying goodbye. Thus, the caller mostly said goodbye to the speaker. Therefore, the answer is Option 1: The caller said goodbye to me.

Input Event: The cook stirred the ingredients in the bowl.
Question: Please select the effect of the input event from the following options.
Option 1: The ingredients melted.
Option 2: The ingredients blended together.
Answer (Option 1 or Option 2 ?) with chain-of-thought:
The question is about the effect. Stirring is a common method used in cooking to blend and mix ingredients. Thus, the effect of stirring is blend together the ingredients.  Therefore, the answer is Option 2: the ingredients blended together.

Input Event: The book became a huge bestseller.
Question: Please select the effect of the input event from the following options.
Option 1: It was adapted into a movie.
Option 2: The author faded into obscurity.
Answer (Option 1 or Option 2 ?) with chain-of-thought:
The question is about the effect. When a book becomes a huge bestseller, it often attracts the attention of filmmakers and can lead to movie adaptations, and authors generally gain more recognition and fame. Thus, Option 1 seems to be the more plausible effect. Therefore, the answer is Option 1: it was adapted into a movie.

Input Event: The man anticipated cold weather on his trip.
Question: Please select the effect of the input event from the following options.
Option 1: He packed warm clothing in his suitcase.
Option 2: He travelled with a big suitcase.
Answer (Option 1 or Option 2 ?) with chain-of-thought:
The question is about the effect. When someone expects cold weather, it is logical for them to pack appropriate clothing to stay warm during their journey. Thus, Option 1 is a reasonable response to the anticipation of cold weather. Therefore, the answer is Option 1: The man anticipated cold weather on his trip.

Input Event: I turned on the fan.
Question: Please select the effect of the input event from the following options.
Option 1: Water sprinkled onto my skin.
Option 2: I felt cool air pass over me.
Answer (Option 1 or Option 2 ?) with chain-of-thought:
The question is about the effect. A typical function of a fan is to circulates air and creates a cooling effect. Therefore, the correct answer is Option 2: I felt cool air pass over me.

Input Event: The woman struggled to walk.
Question: Please select the cause of the input event from the following options.
Option 1: She wore high heels.
Option 2: She took off her shoes.
Answer (Option 1 or Option 2 ?) with chain-of-thought:
The question is about the cause. High heels can be uncomfortable and challenging to walk in for some individual. Thus Option 1 (She wore high heels) seems to be the more plausible cause of the woman struggling to walk. Therefore the answer is Option 1: She wore high heels.

Input Event: I vacuumed the carpet.
Question: Please select the cause of the input event from the following options.
Option 1: My roommate spilled punch.
Option 2: My dog shed hair.
Answer (Option 1 or Option 2 ?) with chain-of-thought:
The question is about the cause. Pets, especially dogs, often shed hair, which can accumulate on the carpet and necessitate vacuuming to keep the carpet clean and tidy. Thus the dog hair may be a more plausible reason for this question. Therefore, the answer is Option 2: My dog shed hair.

Input Event: %s
Question: Please select the %s of the input event from the following options.
Option 1: %s
Option 2: %s
Answer (Option 1 or Option 2 ?):
""",
    'manual-CoT-CN':
    """如下为三个使用思维链进行推理的问题

输入事件：那个女孩许了一个愿望。
问题：请从以下选项中选择输入事件的原因。
选项一：她看到了一只黑猫。
选项二：她看到了一颗流星。
答案（选项一或选项二？）：人们在看到流星时会许愿，因此答案是选项二。

输入事件：龙卷风袭击了这座城镇。
问题：请从以下选项中选择输入事件的结果。
选项一：法院大楼的屋顶被吹掉了。
选项二：公路结冰了，很危险。
答案（选项一或选项二？）：龙卷风通常会带来强风，破坏建筑物，因此答案是选项一。

输入事件：商店收银员叫保安了。
问题：请从以下选项中选择输入事件的原因。
选项一：客户使用了假钞。
选项二：客户忘记关车灯了。
答案（选项一或选项二？）：商店收银员叫保安通常是因为有可疑和异常情况，包括客户用假钞，因此答案是选项一。

输入事件：%s
问题：请从以下选项中选择输入事件的%s。
选项一：%s
选项二：%s
答案（选项一或选项二？）：""",
    'explicit-function':
    """You are a helpful assistant for causal discovery.
Input Event: %s
Question: Please select the %s of the input event from the following options.
Option 1: %s
Option 2: %s
Answer (Option 1 or Option 2 ?):""",
    'explicit-function-CN':
    """你是一个用于因果发现的得力助手。
输入事件：%s
问题：请从以下选项中选择输入事件的%s。
选项一：%s
选项二：%s
答案（选项一或选项二？）：""",
}


def get_prompt(task_name, prompt_style, item, prompt_style_str=''):
    base = base_prompt_dict[prompt_style]

    prompt = prompt_style_str + base % (item['premise'], item['ask-for'],
                                        item['hypothesis1'],
                                        item['hypothesis2'])
    return prompt
