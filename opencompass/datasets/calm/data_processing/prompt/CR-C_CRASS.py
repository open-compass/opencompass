# flake8: noqa: E501
base_prompt_dict = {
    'basic':
    """Input Event: %s
Counterfactual Question: %s
Option 1: %s
Option 2: %s
Option 3: %s
Option 4: %s
Answer (Option 1 or 2 or 3 or 4?):""",
    'basic-CN':
    """输入事件：%s
反事实问题：%s
选项一：%s
选项二：%s
选项三：%s
选项四：%s
答案（选项一或选项二或选项三或选项四？）:""",
    'adversarial-ignore':
    """Input Event: %s
Counterfactual Question: %s
Option 1: %s
Option 2: %s
Option 3: %s
Option 4: %s
Answer (Option 1 or 2 or 3 or 4?):""",
    'adversarial-ignore-CN':
    """输入事件：%s
反事实问题：%s
选项一：%s
选项二：%s
选项三：%s
选项四：%s
答案（选项一或选项二或选项三或选项四？）:""",
    'adversarial-doubt':
    """Input Event: %s
Counterfactual Question: %s
Option 1: %s
Option 2: %s
Option 3: %s
Option 4: %s
Answer (Option 1 or 2 or 3 or 4?):""",
    'adversarial-doubt-CN':
    """输入事件：%s
反事实问题：%s
选项一：%s
选项二：%s
选项三：%s
选项四：%s
答案（选项一或选项二或选项三或选项四？）:""",
    'zero-shot-IcL':
    """Predict the effects of causal events by contemplating hypothetical situations or alternate realities. This involves altering specific elements or conditions of an actual event or situation.
Input Event: %s
Counterfactual Question: %s
Option 1: %s
Option 2: %s
Option 3: %s
Option 4: %s
Answer (Option 1 or 2 or 3 or 4?):""",
    'zero-shot-IcL-CN':
    """通过思考假设情况或另一种现实，预测因果事件的影响。这涉及改变实际事件或情况的特定元素或条件。
输入事件：%s
反事实问题：%s
选项一：%s
选项二：%s
选项三：%s
选项四：%s
答案（选项一或选项二或选项三或选项四？）：""",
    'one-shot-IcL':
    """Predict the effects of causal events by contemplating hypothetical situations or alternate realities. This involves altering specific elements or conditions of an actual event or situation.
Input Event: A woman opens a treasure chest.
Counterfactual Question: What would have happened if the woman had not opened the treasure chest?
Option 1:
Option 2: The treasure chest would have been open.
Option 3: That is not possible.
Option 4: The treasure chest would have remained closed.
Answer (Option 1 or 2 or 3 or 4?): 4

Input Event: %s
Counterfactual Question: %s
Option 1: %s
Option 2: %s
Option 3: %s
Option 4: %s
Answer (Option 1 or 2 or 3 or 4?):""",
    'one-shot-IcL-CN':
    """通过思考假设情况或另一种现实，预测因果事件的影响。这涉及改变实际事件或情况的特定元素或条件。
输入事件：一名女子打开了一个宝藏箱。
反事实问题：如果那位女士没有打开宝藏箱会怎么样？
选项一：
选项二：这个宝藏箱可能已经被打开了。
选项三：那是不可能的。
选项四：这个宝藏箱可能还会保持关闭状态。
答案（选项一或选项二或选项三或选项四？）：四

输入事件：%s
反事实问题：%s
选项一：%s
选项二：%s
选项三：%s
选项四：%s
答案（选项一或选项二或选项三或选项四？）：""",
    'three-shot-IcL':
    """Predict the effects of causal events by contemplating hypothetical situations or alternate realities. This involves altering specific elements or conditions of an actual event or situation.
Input Event: A woman opens a treasure chest.
Counterfactual Question: What would have happened if the woman had not opened the treasure chest?
Option 1:
Option 2: The treasure chest would have been open.
Option 3: That is not possible.
Option 4: The treasure chest would have remained closed.
Answer (Option 1 or 2 or 3 or 4?): 4

Input Event: A police officer calms down a hostage-taker.
Counterfactual Question: What would have happened if the police officer had not calmed the hostage-taker?
Option 1:
Option 2: The hostages would have remained in danger.
Option 3: That is not possible.
Option 4: The hostage-taker would have released the hostages anyway.
Answer (Option 1 or 2 or 3 or 4?): 2

Input Event: A man talks about a lion.
Counterfactual Question: What would have happened if the man had talked to the lion?
Option 1: Without a barrier, the lion would have been eaten.
Option 2:
Option 3: Without a barrier, the man would have been eaten.
Option 4: Nothing special would have happened.
Answer (Option 1 or 2 or 3 or 4?): 3

Input Event: %s
Counterfactual Question: %s
Option 1: %s
Option 2: %s
Option 3: %s
Option 4: %s
Answer (Option 1 or 2 or 3 or 4?):""",
    'three-shot-IcL-CN':
    """通过思考假设情况或另一种现实，预测因果事件的影响。这涉及改变实际事件或情况的特定元素或条件。
输入事件：一名女子打开了一个宝藏箱。
反事实问题：如果那位女士没有打开宝藏箱会怎么样？
选项一：
选项二：这个宝藏箱可能已经被打开了。
选项三：那是不可能的。
选项四：这个宝藏箱可能还会保持关闭状态。
答案（选项一或选项二或选项三或选项四？）：四

输入事件：一个警察安抚了一位挟持者的情绪。
反事实问题：如果警察没有安抚劫匪，会发生什么？
选项一：
选项二：这些人质可能仍然处于危险之中。
选项三：那是不可能的。
选项四：这位劫持者可能最终还是会释放人质的。
答案（选项一或选项二或选项三或选项四？）：二

输入事件：一位男子谈论一只狮子。
反事实问题：如果那个男子跟狮子说话了会怎样？
选项一：如果没有屏障的保护，狮子就会被吃掉。
选项二：
选项三：如果没有屏障的保护，那个男人就会被吃掉了。
选项四：没什么特别的事情发生了。
答案（选项一或选项二或选项三或选项四？）：三

输入事件：%s
反事实问题：%s
选项一：%s
选项二：%s
选项三：%s
选项四：%s
答案（选项一或选项二或选项三或选项四？）：""",
    'zero-shot-CoT':
    """Input Event: %s
    Counterfactual Question: %s Let's think step by step.
    Option 1: %s
    Option 2: %s
    Option 3: %s
    Option 4: %s
    Answer (Option 1 or 2 or 3 or 4?):""",
    'zero-shot-CoT-CN':
    """输入事件：%s请逐步思考。
反事实问题：%s
选项一：%s
选项二：%s
选项三：%s
选项四：%s
答案（选项一或选项二或选项三或选项四？）:""",
    'manual-CoT':
    """Here we will provide eight chain-of-thought exemplars, followed by a multi-choice question that needs to be answered with chain-of-thought.

Input Event: A bird lands in a forest.
Counterfactual Question: What would have happened if a meteor had landed in the forest?
Option 1: The bird would have liked the meteor.
Option 2:
Option 3: A big one would have started a wildfire.
Option 4: That is not possible.
Answer (Option 1 or 2 or 3 or 4? With chain-of-thought): The initial scenario is that a bird lands in a forest. The counterfactual question introduces a hypothetical scenario where a meteor lands in the same forest. The meteor has a high temperature and could have triggered a wildfire due to the resulting heat. Thus, the answer is Option 3: A big one would have started a wildfire.

Input Event: A man reports a crime.
Counterfactual Question: What would have happened if the man had cleared up the crime?
Option 1: He would have been a detective.
Option 2: The room would have been clean by now.
Option 3: That is not possible.
Option 4: He would have been the owner of the crime scene.
Answer (Option 1 or 2 or 3 or 4? With chain-of-thought): The initial scenario is that a man reports a crime. The counterfactual question introduces a hypothetical scenario where the man had cleared up the crime. It suggests that he could have demonstrated qualities and skills similar to those of a detective. Thus, the answer is Option 1: He would have been a detective.

Input Event: A country loses a war.
Counterfactual Question: What would have happened if the country had won the war?
Option 1: That is not possible.
Option 2: Most people of the winning country would have been sad.
Option 3: Most people of the winning country would have been happy.
Option 4: The war would have continued.
Answer (Option 1 or 2 or 3 or 4? With chain-of-thought): The initial scenario is that a country loses a war. The counterfactual question introduces a hypothetical scenario where the country had won the war. The prevailing sentiment among the population would be happiness due to the successful outcome of the conflict. Thus, the answer is Option 3: Most people of the winning country would have been happy.

Input Event: A bird flies over a bridge.
Counterfactual Question: What would have happened if the bird had hit the bridge?
Option 1: The bird would have caused damage to the bridge.
Option 2: That is not possible.
Option 3:
Option 4: The bird would have been injured.
Answer (Option 1 or 2 or 3 or 4? With chain-of-thought): The initial scenario is that a bird flies over a bridge. The counterfactual question introduces a hypothetical scenario where the bird had hit the bridge. Then the bird will get injured due to the collision. Thus, the answer is Option 4: The bird would have been injured.

Input Event: A girl mopped her floor.
Counterfactual Question: What would have happened if the girl had poured mud on her floor?
Option 1:
Option 2: The floor would be dirty.
Option 3: That is not possible.
Option 4: The floor would be clean.
Answer (Option 1 or 2 or 3 or 4? With chain-of-thought): The initial scenario is that a  girl mopped her floor. The counterfactual question introduces a hypothetical scenario where the girl had poured mud on her floor. Then the floor becomes dirty due to the presence of mud. Thus, the answer is Option 2: The floor would be dirty.

Input Event: You accepted an MTurk HIT.
Counterfactual Question: What would have happened if you had rejected the MTurk HIT?
Option 1: The turker would have been disappointed.
Option 2: That is not possible.
Option 3:
Option 4: The turker would have been happy.
Answer (Option 1 or 2 or 3 or 4? With chain-of-thought): The initial scenario is that you have accepted a task on Amazon Mechanical Turk (MTurk). The counterfactual question introduces a hypothetical scenario where you reject the MTurk HIT. The worker (referred to as "turker") who submitted the work would likely have been disappointed since their efforts would not result in compensation. Thus, the answer is Option 1: The turker would have been disappointed.

Input Event: A woman does not write an article.
Counterfactual Question: What would have happened if the woman had written an article?
Option 1: She would have gotten it published.
Option 2:
Option 3: That is not possible.
Option 4: She would not have gotten it published.
Answer (Option 1 or 2 or 3 or 4? With chain-of-thought): The initial scenario is that a woman does not write an article. The counterfactual question introduces a hypothetical scenario where the woman had written an article. Then she could have succeeded in getting it published. Thus, the answer is Option 1: She would have gotten it published.

Input Event: A woman does not put pen to paper.
Counterfactual Question: What would have happened if she had put pen to paper?
Option 1: The woman would have moved her right hand.
Option 2: That is not possible.
Option 3: The woman would not have moved her right hand.
Option 4:
Answer (Option 1 or 2 or 3 or 4? With chain-of-thought): The initial scenario is that a  woman does not put pen to paper. The counterfactual question introduces a hypothetical scenario where she had put pen to paper. Then she would have naturally moved her right hand to perform the writing action. Thus, the answer is Option 1: The woman would have moved her right hand.

Input Event: A woman opens a treasure chest.
Counterfactual Question: What would have happened if the woman had not opened the treasure chest?
Option 1:
Option 2: The treasure chest would have been open.
Option 3: That is not possible.
Option 4: The treasure chest would have remained closed.
Answer (Option 1 or 2 or 3 or 4? With chain-of-thought): The initial scenario is that a woman opens a treasure chest. The counterfactual question introduces a hypothetical scenario where the woman had not opened the treasure chest. Then the treasure chest would have remained closed. Thus, the answer is Option 3: That is not possible.

Input Event: %s
Counterfactual Question: %s
Option 1: %s
Option 2: %s
Option 3: %s
Option 4: %s
Answer (Option 1 or 2 or 3 or 4?):""",
    'manual-CoT-CN':
    """如下为三个使用思维链进行推理的问题:

输入事件：位调酒师递来了饮料。
反事实问题：如果这位调酒师喝掉这些饮料会怎么样？
选项一：那是不可能的。
选项二：客人可能就不会拿到他们点的饮料了。
选项三：客人可能就会得到他们的酒水了。
选项四：
答案（选项一或选项二或选项三或选项四？）: 如果调酒师喝掉了饮料，那么显然这些饮料将不再可用，客人可能就不会拿到他们点的饮料了。因此答案是选项二。

输入事件：一位女士聒噪且惹人讨厌。
反事实问题：她要是更安静点会怎么样？
选项一：在她身边可能会很高兴。
选项二：
选项三：和她呆在一起可能会很不愉快。
选项四：那是不可能的。
答案（选项一或选项二或选项三或选项四？）: 如果这位女士更安静点，周围的人不会再受到她的干扰，可能会很高兴。因此答案是选项一。

输入事件：一位女性被一所大学录取了。
反事实问题：这位女士如果被这所大学拒绝了会怎么样？
选项一：那是不可能的。
选项二：这位女士可能会很开心。
选项三：
选项四：这位女士可能会感到悲伤。
答案（选项一或选项二或选项三或选项四？）: 这位女士可能会感到悲伤，因为她没有被这所大学录取。因此答案是选项四。

输入事件：%s
反事实问题：%s
选项一：%s
选项二：%s
选项三：%s
选项四：%s
答案（选项一或选项二或选项三或选项四？）:""",
    'explicit-function':
    """You are a helpful assistant for counterfactual reasoning.
Input Event: %s
Counterfactual Question: %s
Option 1: %s
Option 2: %s
Option 3: %s
Option 4: %s
Answer (Option 1 or 2 or 3 or 4?):""",
    'explicit-function-CN':
    """你是一个用于反事实推理的得力助手。
输入事件：%s
反事实问题：%s
选项一：%s
选项二：%s
选项三：%s
选项四：%s
答案（选项一或选项二或选项三或选项四？）""",
}


def get_prompt(task_name, prompt_style, item, prompt_style_str=''):
    base = base_prompt_dict[prompt_style]

    prompt = prompt_style_str + base % (item['premise'], item['QCC'],
                                        item['Answer1'], item['Answer2'],
                                        item['Answer3'], item['Answer4'])
    return prompt
