# flake8: noqa: E501
base_prompt_dict = {
    'basic': """Input Info: %s
Question: %s
Answer (Yes or No ?):""",
    'basic-CN': """输入信息：%s
问题：%s
答案（是或否？）：""",
    'adversarial-ignore': """Input Info: %s
Question: %s
Answer (Yes or No ?):""",
    'adversarial-ignore-CN': """输入信息：%s
问题：%s
答案（是或否？）：""",
    'adversarial-doubt': """Input Info: %s
Question: %s
Answer (Yes or No ?):""",
    'adversarial-doubt-CN': """输入信息：%s
问题：%s
答案（是或否？）：""",
    'zero-shot-IcL':
    """Answer questions by considering what constitutes a valid adjustment set that can block all backdoor spurious correlations between two events.
Input Info: %s
Question: %s
Answer (Yes or No ?):""",
    'zero-shot-IcL-CN': """通过考虑什么构成一个有效的调整集，以阻断两个事件之间所有后门伪相关，来回答问题。
输入信息：%s
问题：%s
答案（是或否？）：""",
    'one-shot-IcL':
    """Answer questions by considering what constitutes a valid adjustment set that can block all backdoor spurious correlations between two events.
Input Info: Method 1: We look at how husband correlates with alarm clock case by case according to wife. Method 2: We look directly at how husband correlates with alarm clock in general.
Question: To understand how husband affects alarm clock, is it more correct to use the Method 1 than Method 2?
Answer (Yes or No ?): no

Input Info: %s
Question: %s
Answer (Yes or No ?):""",
    'one-shot-IcL-CN': """通过考虑什么构成一个有效的调整集，以阻断两个事件之间所有后门伪相关，来回答问题。
输入信息：方法1：根据妻子的情况，我们逐个研究丈夫与闹钟之间的关联；方法2：我们直接研究一般情况下丈夫与闹钟之间的关联。
问题：要了解丈夫如何影响闹钟，使用方法1比方法2更准确吗？
答案（是或否？）：否

输入信息：%s
问题：%s
答案（是或否？）：""",
    'three-shot-IcL':
    """Answer questions by considering what constitutes a valid adjustment set that can block all backdoor spurious correlations between two events.
Input Info: Method 1: We look at how husband correlates with alarm clock case by case according to wife. Method 2: We look directly at how husband correlates with alarm clock in general.
Question: To understand how husband affects alarm clock, is it more correct to use the Method 1 than Method 2?
Answer (Yes or No ?): no

Input Info: Method 1: We look directly at how husband correlates with alarm clock in general. Method 2: We look at this correlation case by case according to wife.
Question: To understand how husband affects alarm clock, is it more correct to use the Method 1 than Method 2?
Answer (Yes or No ?): yes

Input Info: Method 1: We look directly at how the man in the room correlates with room in general. Method 2: We look at this correlation case by case according to the candle.
Question: To understand how the man in the room affects room, is it more correct to use the Method 1 than Method 2?
Answer (Yes or No ?): yes

Input Info: %s
Question: %s
Answer (Yes or No ?):""",
    'three-shot-IcL-CN': """通过考虑什么构成一个有效的调整集，以阻断两个事件之间所有后门伪相关，来回答问题。
输入信息：方法1：根据妻子的情况，我们逐个研究丈夫与闹钟之间的关联；方法2：我们直接研究一般情况下丈夫与闹钟之间的关联。
问题：要了解丈夫如何影响闹钟，使用方法1比方法2更准确吗？
答案（是或否？）：否

输入信息：方法1：我们直接研究一般情况下丈夫与闹钟之间的关联。方法2：根据妻子的情况，我们逐个研究这种关联。
问题：要了解丈夫如何影响闹钟，使用方法1比方法2更准确吗？
答案（是或否？）：是

输入信息：方法1: 我们直接研究一般情况下房间里的男人与房间之间的关联;方法2:根据蜡烛，我们逐个研究这种关联。
问题：要了解房间里的男子如何影响房间，使用方法1比方法2更准确吗？
答案（是或否？）：是

输入信息：%s
问题：%s
答案（是或否？）：""",
    'zero-shot-CoT': """Input Info: %s
Question: %s Let's think step by step.
Answer (Yes or No ?):""",
    'zero-shot-CoT-CN': """输入信息：%s
问题：%s请逐步思考。
答案（是或否？）：""",
    'manual-CoT':
    """Here are three examples for problems about considering backdoor adjustment set with chain of thought.
Input Info: Method 1: We look directly at how jyka correlates with lirg in general. Method 2: We look at this correlation case by case according to gyzp.
Question: To understand how jyka affects lirg, is it more correct to use the Method 1 than Method 2?
Answer (Yes or No ?): Since gyzp is a confounder, both affects jyka and lirg, looking directly at the relation between jyka and lirg like Method 1 is not correct. Therefore, the answer is No.

Input Info: Method 1: We look directly at how encouragement level correlates with brown eyes in general. Method 2: We look at this correlation case by case according to studying habit.
Question: To understand how encouragement level affects brown eyes, is it more correct to use the Method 1 than Method 2?
Answer (Yes or No ?): Since studying habit is a result of encouragement level, there is no need to consider studying habit when studying the relation between encouragement level and brown eyes. Therefore, the answer is Yes.

Input Info: Method 1: We look directly at how zuph correlates with glimx in general. Method 2: We look at this correlation case by case according to zory.
Question: To understand how zuph affects glimx, is it more correct to use the Method 1 than Method 2?
Answer (Yes or No ?): Since zory is a confounder, both affects zuph and glimx, looking at the correlation without considering zory is not correct. Therefore, the answer is No.

Input Info: %s
Question: %s
Answer (Yes or No ?):""",
    'manual-CoT-CN': """如下为三个使用思维链进行推理的有关后门变量集合的问题：

输入信息：方法1: 我们直接研究一般情况下房间里的男人与房间之间的关联;方法2:根据蜡烛，我们逐个研究这种关联。
问题：要了解房间里的男子如何影响房间，使用方法1比方法2更准确吗？
答案（是或否？）：因为房间里的男人和蜡烛对房间的影响是相互独立的，所以蜡烛不会影响房间里的男人和房间之间的关联。因此方法1更好。因此答案为“是”。

输入信息：方法1：我们直接研究一般情况下jyka与lirg之间的关联。方法2：根据gyzp，我们逐个研究这种关联。
问题：要了解gyzp如何影响lirg，使用方法1比方法2更准确吗？
答案（是或否？）：因为gyzp作为混淆变量会同时影响jyka和lirg，使用方法1会导致对jyka和lirg之间的关联产生错误判断。因此答案为“否”。

输入信息：方法1：我们直接研究一般情况下鼓励程度与考试成绩之间的关联。方法2：根据学习习惯，我们逐个研究这种关联。
问题：要了解鼓励程度如何影响考试成绩，使用方法1比方法2更准确吗？
答案（是或否？）：因为学习成绩是鼓励程度的结果，不会影响鼓励程度和考试成绩之间的关联。因此方法1更好。因此答案为“是”。

输入信息：%s
问题：%s
答案（是或否？）""",
    'explicit-function':
    """You are a helpful assistant for backdoor adjustment set.
Input Info: %s
Question: %s
Answer (Yes or No ?):""",
    'explicit-function-CN': """你是一个用于后门调节的得力助手。
输入信息：%s
问题：%s
答案（是或否？）：""",
}


def get_prompt(task_name, prompt_style, item, prompt_style_str=''):
    base = base_prompt_dict[prompt_style]

    prompt = prompt_style_str + base % (item['given_info'], item['question'])
    return prompt
