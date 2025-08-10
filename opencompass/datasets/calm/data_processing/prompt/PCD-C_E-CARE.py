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
    """Here we will provide eight chain-of-thought exemplars, where a few chain of thought demonstrations are provided as exemplars in prompting, followed by a question that needs to be answered.

Input Event: Black's sweat always drips into his eyes.
Question: Please select the cause of the input event from the following options.
Option 1: Black pulled out his eyelashes for beauty.
Option 2: Black doesn't like sleeping.
Answer (Option 1 or Option 2 ?) with chain-of-thought:
The question is about the cause. If Black intentionally removed his eyelashes, it could potentially lead to sweat dripping into his eyes due to the lack of eyelashes to provide some protection. Therefore, the answer is Option 1: Black pulled out his eyelashes for beauty.

Input Event: It's half way through autumn.
Question: Please select the effect of the input event from the following options.
Option 1: It has difficulty in running.
Option 2: It rains more in half autumn than in spring and summer combined.
Answer (Option 1 or Option 2 ?) with chain-of-thought:
The question is about the effect. Autumn is commonly associated with changing weather patterns, including increased rainfall in some regions. During the half autumn, there is more rainfall compared to the combined total of spring and summer. Therefore, the answer is Option 2: It rains more in half autumn than in spring and summer combined.

Input Event: The man  planned to make Tin  by himself.
Question: Please select the effect of the input event from the following options.
Option 1: He had to design the necessary components.
Option 2: He found cassiterite, carbon and  furnace.
Answer (Option 1 or Option 2 ?) with chain-of-thought:
The question is about the effect. Cassiterite, carbon and furnace are essential components for the process of extracting tin from its mineral deposit. Thus, finding cassiterite, carbon and furnace is a clear and relevant effect resulting from the man's plan to make Tin by himself. Therefore, the answer is Option 2: He found cassiterite, carbon and  furnace.

Input Event: He was shocked by his chemical deficiency.
Question: Please select the cause of the input event from the following options.
Option 1: The food Tom ate contained bacteria.
Option 2: The patient with addiction watched the neuroscientist's lecture.
Answer (Option 1 or Option 2 ?) with chain-of-thought:
The question is about the cause. Neuroscience gives a deep insight of chemical imbalances in one's body. He might come to realize the lack of some  nutrients in his body after the neuroscience lecture, thus felt shocked. Therefore, the answer is Option 2: The patient with addiction watched the neuroscientist's lecture.

Input Event: Tom bought a lot of mangoes and coconuts.
Question: Please select the cause of the input event from the following options.
Option 1: Tom buys imported tropical fruit every day.
Option 2: The doctor advised Tom to eat more tropical fruits to supplement his vitamins.
Answer (Option 1 or Option 2 ?) with chain-of-thought:
The question is about the cause. The doctor's advice to eat more tropical fruits, like mangoes and coconuts, as a source of vitamins could make Tom buy a lot of mangoes and coconuts. Therefore, the answer is Option 2: The doctor advised Tom to eat more tropical fruits to supplement his vitamins.

Input Event: Waterwheels started work efficiently.
Question: Please select the effect of the input event from the following options.
Option 1: The mills can set to work.
Option 2: The scientists have invented replacements.
Answer (Option 1 or Option 2 ?) with chain-of-thought:
The question is about the effect. When the waterwheels are working efficiently, it enables the mills to start operating. Therefore, the answer is Option 1: The mills can set to work.

Input Event: Mary has two pieces of farmland, but only one of them is used to grow crops every year.
Question: Please select the effect of the input event from the following options.
Option 1: The often used farmland produces a lot more crops than the less often used one.
Option 2: The less often used farmland produces more crops than the often used one.
Answer (Option 1 or Option 2 ?) with chain-of-thought:
The question is about the effect. Since regular cultivation can lead to healthier soil and better yields, the farmland that is used more frequently for growing crops is more productive. Therefore, the answer is Option 1: The often used farmland produces a lot more crops than the less often used one.

Input Event: He can just see something clearly in a short distance.
Question: Please select the cause of the input event from the following options.
Option 1: Tom turned on his flashlight.
Option 2: Tom measured the energy of the lightning during a thunderstorm.
Answer (Option 1 or Option 2 ?) with chain-of-thought:
The question is about the cause. Turning on a flashlight provides additional light in the immediate vicinity, making objects visible in a short distance. Therefore, the answer is Option 1: Tom turned on his flashlight.

Input Event: %s
Question: Please select the %s of the input event from the following options.
Option 1: %s
Option 2: %s
Answer (Option 1 or Option 2 ?) :""",
    'manual-CoT-CN':
    """如下为三个使用思维链进行推理的问题:

输入事件：莎莉的喉咙严重发炎了。
问题：请从以下选项中选择输入事件的结果。
选项一：莎莉发不出声音。
选项二：她的眼睛受伤了。
答案（选项一或选项二？）：喉咙严重发炎会导致声音嘶哑或失声，因此答案是选项一。

输入事件：很多昆虫都被它们吃掉了。
问题：请从以下选项中选择输入事件的原因。
选项一：果园里有很多麻雀。
选项二：人类需要营养丰富的食物来维持生存。
答案（选项一或选项二？）：麻雀等鸟类喜欢吃昆虫，因此答案是选项一。

输入事件：它具有糖酵解功能。
问题：请从以下选项中选择输入事件的原因。
选项一：肌原纤维中含有不同数量的肌原丝。
选项二：这种酶促进葡萄糖的分解。
答案（选项一或选项二？）：酶有促进糖降解的功能，因此答案是选项二。

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
