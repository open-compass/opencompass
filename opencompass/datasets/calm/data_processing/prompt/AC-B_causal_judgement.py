# flake8: noqa: E501
base_prompt_dict = {
    'basic': """Question: %s
Answer (Yes or No ?):""",
    'basic-CN': """问题：%s
答案（是或否？）：""",
    'adversarial-ignore': """Question: %s
Answer (Yes or No ?):""",
    'adversarial-ignore-CN': """问题：%s
答案（是或否？）：""",
    'adversarial-doubt': """Question: %s
Answer (Yes or No ?):""",
    'adversarial-doubt-CN': """问题：%s
答案（是或否？）：""",
    'zero-shot-IcL': """
Define actual causality in a way that is consistent with how humans naturally attribute cause and related concepts of responsibility, blame, and explanations to events and their outcomes.
Question: %s
Answer (Yes or No ?):""",
    'zero-shot-IcL-CN': """
请用类似于人们定义事件的因果以及与其相关的责任、过错、解释等概念的方式，定义真实的因果性。
问题：%s
答案（是或否？）：""",
    'one-shot-IcL': """
Define actual causality in a way that is consistent with how humans naturally attribute cause and related concepts of responsibility, blame, and explanations to events and their outcomes.
Question: How would a typical person answer each of the following questions about causation?
A machine is set up in such a way that it will short circuit if both the black wire and the red wire touch the battery at the same time. The machine will not short circuit if just one of these wires touches the battery. The black wire is designated as the one that is supposed to touch the battery, while the red wire is supposed to remain in some other part of the machine. One day, the black wire and the red wire both end up touching the battery at the same time. There is a short circuit. Did the black wire cause the short circuit?
Options:
- Yes
- No
Answer (Yes or No ?): No

Question: %s
Answer (Yes or No ?):""",
    'one-shot-IcL-CN': """
请用类似于人们定义事件的因果以及与其相关的责任、过错、解释等概念的方式，定义真实的因果性。
问题：对于以下关于因果关系的问题，一个普通人会怎么回答？
一台机器被设置成如果黑线和红线同时碰到电池，那么机器就会短路。而如果只有其中一根线碰到电池，则不会短路。黑线被指定为应该接触电池的那根，而红线则应该留在机器的其他部分。有一天，黑线和红线同时接触到电池，从而机器出现了短路，是黑线造成了短路吗？
选项：
- 是
- 否
答案（是或否？）：否

问题：%s
答案（是或否？）：""",
    'three-shot-IcL': """
Define actual causality in a way that is consistent with how humans naturally attribute cause and related concepts of responsibility, blame, and explanations to events and their outcomes.
Question: How would a typical person answer each of the following questions about causation?
A machine is set up in such a way that it will short circuit if both the black wire and the red wire touch the battery at the same time. The machine will not short circuit if just one of these wires touches the battery. The black wire is designated as the one that is supposed to touch the battery, while the red wire is supposed to remain in some other part of the machine. One day, the black wire and the red wire both end up touching the battery at the same time. There is a short circuit. Did the black wire cause the short circuit?
Options:
- Yes
- No
Answer (Yes or No ?): No

Question: How would a typical person answer each of the following questions about causation?
Claire's parents bought her an old computer. Claire uses it for schoolwork, but her brother Daniel sometimes logs on to play games. Claire has told Daniel, "Please don't log on to my computer. If we are both logged on at the same time, it will crash". One day, Claire and Daniel logged on to the computer at the same time. The computer crashed. Later that day, Claire's mother is talking with the computer repairman. The repairman says, "I see that Daniel was logged on, but this computer will only crash if two people are logged on at the same time. So, I still don't see quite why the computer crashed." Did Daniel cause the computer crash?
Options:
- Yes
- No
Answer (Yes or No ?): Yes

Question: How would a typical person answer each of the following questions about causation?
Suzy and Billy are working on a project that is very important for our nation's security. The boss tells Suzy: "Be sure that you are here at exactly 9 am. It is absolutely essential that you arrive at that time." Then he tells Billy: "Be sure that you do not come in at all tomorrow morning. It is absolutely essential that you not appear at that time." Both Billy and Suzy arrive at 9 am. As it happens, there was a motion detector installed in the room where they arrived. The motion detector was set up to be triggered if at least one person appeared in the room at the same time. So the motion detector went off. Did Billy cause the motion detector to go off?
Options:
- Yes
- No
Answer (Yes or No ?): Yes

Question: %s
Answer (Yes or No ?):""",
    'three-shot-IcL-CN': """
请用类似于人们定义事件的因果以及与其相关的责任、过错、解释等概念的方式，定义真实的因果性。
问题：对于以下关于因果关系的问题，一个普通人会怎么回答？
一台机器被设置成如果黑线和红线同时碰到电池，那么机器就会短路。而如果只有其中一根线碰到电池，则不会短路。黑线被指定为应该接触电池的那根，而红线则应该留在机器的其他部分。有一天，黑线和红线同时接触到电池，从而机器出现了短路，是黑线造成了短路吗？
选项：
- 是
- 否
答案（是或否？）：否

问题：对于以下关于因果关系的问题，一个普通人会怎么回答？
克洛伊的父母给她买了一台旧电脑。克洛伊用它做作业，但她的兄弟丹尼尔有时会登录来玩游戏。克洛伊告诉过丹尼尔：“请不要登录我的电脑。如果我们两个都在线，电脑就会死机。”有一天，克洛伊和丹尼尔同时登录了电脑。电脑死机了。当天晚些时候，克洛伊的母亲正在与电脑修理工交谈。技术员说：“我看到丹尼尔已经登录了，但这个电脑只有在两个人同时登录时才会死机。所以，我仍然不明白为什么电脑会死机。”丹尼尔导致了电脑死机吗？
选项：
- 是
- 否
答案（是或否？）：是

问题：对于以下关于因果关系的问题，一个普通人会怎么回答？
苏茜和比利正在合作完成一份关乎国家安全的重要项目。老板告诉苏茜：“你务必在早上九点准时到达。切记要按时到。”然后他告诉比利：“不要在明天早上九点来。你要保证千万别在那个时间出现。”两人都在上午九点到达了。恰好，房间里装有运动探测器。只要有至少一个人在房间内出现，运动探测器就会触发响起。是比利导致了运动探测器响起的吗？
选项：
- 是
- 否
答案（是或否？）：是

问题：%s
答案（是或否？）：""",
    'zero-shot-CoT': """Question: %s\nLet's think step by step.
Answer (Yes or No ?):""",
    'zero-shot-CoT-CN': """问题：%s请逐步思考。
答案（是或否？）：""",
    'manual-CoT':
    """Here we will provide three chain-of-thought exemplars, followed by a binary question that needs to be answered.

Question: How would a typical person answer each of the following questions about causation?
Frank T., had an ongoing dispute with his neighbor over a stretch of land and one day decided to shoot his neighbor in the body. Frank T. had no experience with guns, his hand slipped on the barrel of the gun, and the shot went wild. Nonetheless, the bullet bounced off a large boulder several feet away and hit the neighbor's body, causing significant injury. Did Frank T. intentionally shoot his neighbor in the body?
Options:
- Yes
- No
Answer(with chain of thought): Here in this question, we are told that "Frank T. had no experience with guns, his hand slipped on the barrel of the gun, and the shot went wild." A typical person would assume that this passage suggests that Frank T. had no intention of shooting and injuring someone and that the bullet accidentally hit the neighbor's body; therefore, we conclude that Frank T. did not intentionally hit his neighbor. So the answer is No.

Question: How would a typical person answer each of the following questions about causation?
Suzy and Billy are working on a project that is very important for our nation's security. The boss tells them both: "Be sure that you are here at exactly 9 am. It is absolutely essential that you arrive at that time." Both Billy and Suzy arrive at 9 am. As it happens, there was a motion detector installed in the room where they arrived. The motion detector was set up to be triggered if at least one person appeared in the room at the same time. So the motion detector went off. Did Billy cause the motion detector to go off?
Options:
- Yes
- No
Answer(with chain of thought):
Here in this question, we are told that the boss ordered them both to arrive at the meeting room at the same time and that the motion detector was set up to be triggered if at least one person appeared in the room at the same time." A typical person would assume that the person probably meant to say the detector was set up to be triggered if "both persons" appeared in the room at the same time, not at least one person, since otherwise the phrase "at the same time" would not make much sense in that sentence. Because the motion detector went off, a typical person would therefore come to the conclusion that both Suzy and Billy triggered the motion detector to go off; hence, Billy did indeed cause the motion detector to go off. So the answer is Yes.

Question: How would a typical person answer each of the following questions about causation?
George and his sister Lena reunite at their parents' house for Thanksgiving. Whereas George just got into medical school, Lena is unhappy in her marriage and recently lost her job. Over the course of the day, George and Lena get into a number of heated arguments. Later in the afternoon they play a game of darts. They split the first two games, and the third game is close until the end. Who will win comes down to George's last shot. If he hits a high point region, he wins; if he hits a low point region, Lena wins. George thinks of the difficult time Lena is having, and he really wants to let her win. He aims the dart at the low point region. He sets up his shot and the dart lands in the low point region. After his shot, Lena wins the game and is very happy. Did George hit the low point region intentionally?
Options:
- Yes
- No
Answer(with chain of thought):
Here in this question, we are told that "He aims the dart at the low point region." A typical person might therefore think George did intentionally hit the low point region, because he wanted to lift up the spirit of his sister Lena. So the answer is Yes.

Question: %s
Answer (Yes or No ?):""",
    'manual-CoT-CN': """如下为一个使用思维链进行推理的因果归因的示例，和一个需要回答的问题。

问题：对于以下关于因果关系的问题，一个普通人会怎么回答？\n研发部门的代表向董事会报告并说:\"我们正在考虑启动一项新计划。这将有助于增加利润，也有助于保护环境。\"董事会的回答是,\"我们根本不在乎帮助环境。只想尽可能地多赚钱。我们实施计划吧。\"果然，该计划对环境保护起了作用。董事会是有意做环境保护的吗？\n选项：\n- 是\n- 否
答案（是或否？）：董事会的回应表明他们只关心赚钱，根本不在乎环境保护，只是该计划恰好也有助于保护环境。因此答案是“否”。

问题：%s
答案（是或否？）：
""",
    'explicit-function': """You are a helpful assistant for causal attribution.
Question: %s
Answer (Yes or No ?):""",
    'explicit-function-CN': """你是一个用于因果归因的得力助手。
问题：%s
答案（是或否？）：""",
}


def get_prompt(task_name, prompt_style, item, prompt_style_str=''):
    base = base_prompt_dict[prompt_style]

    prompt = prompt_style_str + base % (item['input'])
    return prompt
