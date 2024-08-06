# flake8: noqa: E501
base_prompt_dict = {
    'basic':
    """Cause: %s
Effect: %s
Question: why the cause can lead to the effect ?
Answer:""",
    'basic-CN':
    """原因：%s
结果：%s
问题：为什么原因会导致这样的结果？
答案：""",
    'adversarial-ignore':
    """Cause: %s
Effect: %s
Question: why the cause can lead to the effect ?
Answer:""",
    'adversarial-ignore-CN':
    """原因：%s
结果：%s
问题：为什么原因会导致这样的结果？
答案：""",
    'adversarial-doubt':
    """Cause: %s
Effect: %s
Question: why the cause can lead to the effect ?
Answer:""",
    'adversarial-doubt-CN':
    """原因：%s
结果：%s
问题：为什么原因会导致这样的结果？
答案：""",
    'zero-shot-IcL':
    """generate explanations for causal relations between events.
Cause: %s
Effect: %s
Question: why the cause can lead to the effect ?
Answer:""",
    'zero-shot-IcL-CN':
    """请生成事件之间因果关系的解释。
原因：%s
结果：%s
问题：为什么原因会导致这样的结果？
答案：""",
    'one-shot-IcL':
    """generate explanations for causal relations between events.
Cause: The woman gave birth to a child.
Effect: The child brought psycho-physical phenomena on a new life.
Question: why the cause can lead to the effect ?
Answer: Birth is the arising of the psycho-physical phenomena.

Cause: %s
Effect: %s
Question: why the cause can lead to the effect ?
Answer:""",
    'one-shot-IcL-CN':
    """请生成事件之间因果关系的解释。
原因：这位女士生下了一个孩子。
结果：这个孩子给新生活带来了心理-生理现象。
问题：为什么原因会导致这样的结果？
答案：出生是心理-生理现象的产生原因。

原因：%s
结果：%s
问题：为什么原因会导致这样的结果？
答案：""",
    'three-shot-IcL':
    """generate explanations for causal relations between events.
Cause: The woman gave birth to a child.
Effect: The child brought psycho-physical phenomena on a new life.
Question: why the cause can lead to the effect ?
Answer: Birth is the arising of the psycho-physical phenomena.

Cause: Otters enter their new habitat.
Effect: Otters start looking for abalone for food.
Question: why the cause can lead to the effect ?
Answer: Abalone are one of the first food items taken by otters as they move into new habitat.

Cause: Lila loves classification of her things.
Effect: Lila can find what she wants quickly.
Question: why the cause can lead to the effect ?
Answer: Classifications yield accuracy.

Cause: %s
Effect: %s
Question: why the cause can lead to the effect ?
Answer:""",
    'three-shot-IcL-CN':
    """请生成事件之间因果关系的解释。
原因：这位女士生下了一个孩子。
结果：这个孩子给生活带来了新的心理-生理现象。
问题：为什么原因会导致这样的结果？
答案：出生是心理-生理现象的起源。

原因：水獭进入它们的新栖息地。
结果：水獭开始寻找鲍鱼作为食物。
问题：为什么原因会导致这样的结果？
答案：鲍鱼是水獭搬进新栖息地时最先吃的食物之一。

原因：莉拉喜欢对她的东西进行分类。
结果：莉莉可以很快地找到她想要的东西。
问题：为什么原因会导致这样的结果？
答案：分类可以提高准确度。

原因：%s
结果：%s
问题：为什么原因会导致这样的结果？
答案：""",
    'zero-shot-CoT':
    """Cause: %s
Effect: %s
Question: why the cause can lead to the effect ? Let's think step by step.
Answer:""",
    'zero-shot-CoT-CN':
    """原因：%s
结果：%s
问题：为什么原因会导致这样的结果？请逐步思考。
答案：""",
    'manual-CoT':
    """Here we will provide eight chain-of-thought exemplars, followed by a causal explanation generating question that needs to be answered with chain-of-thought.

Cause: His action led to the movement of the wheels.
Effect: The machine was set in motion.
Question: why the cause can lead to the effect?
Answer(with chain-of-thought): Movement results in motion. The initial movement caused by the action eventually builds up and transitions into the sustained motion of the machine.

Cause: All relatives entered the family room.
Effect: They sat on the chairs one by one.
Question: why the cause can lead to the effect?
Answer(with chain-of-thought): Chairs sit in family rooms. The presence of chairs in the family room sets the stage for the expected behavior of sitting down when relatives enter the room.

Cause: Seals are mammals.
Effect: They can live well in winter.
Question: why the cause can lead to the effect ? Let's think step by step.
Answer(with chain-of-thought): Seals are protected from the cold by a thick layer of blubber combined with a thick fur coat. Thus, they could withstand cold temperatures and maintain their body heat. This adaptation aligns with the effect of being able to live well in winter.

Cause: A stove is an enclosed space in which fuel is burned to provide heating.
Effect: Its surfaces protect people from hurting.
Question: why the cause can lead to the effect?
Answer(with chain-of-thought): Stoves have surfaces. Stove surfaces are a crucial safety feature that shields individuals from direct contact with the heat and flames generated during the burning of fuel inside the stove.

Cause: The student majored in medicine had to choose a research interest.
Effect: He chose Psychiatry.
Question: why the cause can lead to the effect?
Answer(with chain-of-thought): Psychiatry is a branch of medicine. The student's background in medicine makes Psychiatry a logical and suitable research interest.

Cause: The doctor told William that his eyesight was gradually losing.
Effect: The doctor used radiotherapy to treat William.
Question: why the cause can lead to the effect?
Answer(with chain-of-thought): Radiotherapy uses low dose radiation to stop the progression of vision loss on the retina. It is a medical intervention that can be utilized to address certain conditions causing vision loss on the retina.

Cause: The angel controls the Kingdom of Heaven.
Effect: Dominion is part of his responsibility.
Question: why the cause can lead to the effect?
Answer(with chain-of-thought): Dominion is a type of the Kingdom of Heaven. By controlling the Kingdom of Heaven, the angel's responsibilities include exercising authority and rule, which align with the concept of dominion.

Cause: The government published a new policy.
Effect: The public knew its meaning.
Question: why the cause can lead to the effect?
Answer(with chain-of-thought): Policy makes senses. Policies are constructed to convey information in a way that makes sense to the readers.

Cause: %s
Effect: %s
Question: why the cause can lead to the effect ?
Answer:""",
    'manual-CoT-CN':
    """如下为三个使用思维链进行推理的问题:

原因：莱勒有眼科医生。
结果：莱尔的医生用激光治疗了他。
问题：为什么原因会导致这样的结果？
答案：眼科医生通常用激光治疗增生性视网膜病变。

原因：作者运用了拟人手法来描述无生命物体。
结果：读者觉得它好像有人类的能力。
问题：为什么原因会导致这样的结果？
答案：拟人手法是将无生命物体描述成具有人类特征的表达方式。

原因：约翰想种一棵半耐寒多年生植物。
结果：他种了蒲公英。
问题：为什么原因会导致这样的结果？
答案：蒲公英是半耐寒多年生植物。

原因：%s
结果：%s
问题：为什么原因会导致这样的结果？
答案：""",
    'explicit-function':
    """You are a helpful assistant for causal explanation generation.
Cause: %s
Effect: %s
Question: why the cause can lead to the effect ?
Answer:""",
    'explicit-function-CN':
    """你是一个用于因果解释生成的得力助手。
原因：%s
结果：%s
问题：为什么原因会导致这样的结果？
答案""",
}


def get_prompt(task_name, prompt_style, item, prompt_style_str=''):
    base = base_prompt_dict[prompt_style]

    prompt = prompt_style_str + base % (item['cause'], item['effect'])
    return prompt
