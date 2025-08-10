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
    'zero-shot-IcL': """Answer questions about deterministic counterfactual.
Input Info: %s
Question: %s
Answer (Yes or No ?):""",
    'zero-shot-IcL-CN': """请回答有关确定性反事实的问题。
输入信息：%s
问题：%s
答案（是或否？）：""",
    'one-shot-IcL': """Answer questions about deterministic counterfactual.
Input Info: We know that alarm set by husband causes alarm not set by wife. alarm set by husband or alarm set by wife causes ringing alarm.
Question: Would the alarm rings the next morning if alarm not set by husband instead of alarm set by husband?
Answer (Yes or No ?): Yes

Input Info: %s
Question: %s
Answer (Yes or No ?):""",
    'one-shot-IcL-CN': """请回答有关确定性反事实的问题。
输入信息：我们知道丈夫设置闹钟会导致妻子没有设置闹钟，丈夫设置闹钟或妻子设置闹钟会导致闹钟响铃。
问题：如果丈夫没有设置闹钟，而不是丈夫设置闹钟，第二天早上闹钟会响吗？
答案（是或否？）：是

输入信息：%s
问题：%s
答案（是或否？）：""",
    'three-shot-IcL': """Answer questions about deterministic counterfactual.
Input Info: We know that alarm set by husband causes alarm not set by wife. alarm set by husband or alarm set by wife causes ringing alarm.
Question: Would the alarm rings the next morning if alarm not set by husband instead of alarm set by husband?
Answer (Yes or No ?): Yes

Input Info: We know that alarm set by husband causes alarm set by wife. alarm set by husband or alarm set by wife causes ringing alarm.
Question: Would the alarm rings the next morning if alarm not set by husband instead of alarm set by husband?
Answer (Yes or No ?): no

Input Info: We know that alarm set by husband causes alarm set by wife. alarm set by husband or alarm set by wife causes ringing alarm.
Question: Would the alarm doesn't ring the next morning if alarm set by husband instead of alarm not set by husband?
Answer (Yes or No ?): no

Input Info: %s
Question: %s
Answer (Yes or No ?):""",
    'three-shot-IcL-CN': """请回答有关确定性反事实的问题。
输入信息：我们知道丈夫设置闹钟会导致妻子没有设置闹钟，丈夫设置闹钟或妻子设置闹钟会导致闹钟响铃。
问题：如果丈夫没有设置闹钟，而不是丈夫设置闹钟，第二天早上闹钟会响吗？
答案（是或否？）：是

输入信息：我们知道丈夫设置闹钟会导致妻子设置闹钟，丈夫设置闹钟或妻子设置闹钟会导致闹钟响铃。
问题：如果丈夫没有设置闹钟，而不是丈夫设置闹钟，第二天早上闹钟会响吗？
答案（是或否？）：否

输入信息：我们知道丈夫设置闹钟会导致妻子设置闹钟，丈夫设置闹钟或妻子设置闹钟会导致闹钟响铃。
问题：如果是丈夫设置闹钟，而不是丈夫没有设置闹钟，第二天早上闹钟不会响吗？
答案（是或否？）：否

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
    """Here are three examples of problems about deterministic counterfactual with chain of thought.

Input Info: We know that having a sister causes the corporal shooting and the private not shooting. the corporal shooting and the private shooting causes the prisoner's death.
Question: Would the prisoner is dead if not having a sister instead of having a sister?
Answer (Yes or No ?): Let X = having a sister; V3 = the private; V2 = the corporal; Y = prisoner. The causal relations are: X->V3,X->V2,V2->Y,V3->Y. Set Y_{X=0} = 1 | , then solve for Y, given the evidence and the action. V2 = X\nV3 = not V2\nY = V2 and V3. Then we get Y = [0] = 0 and 1. Thus, the prisoner would not be dead if not having a sister instead of having a sister. Therefore, the answer is No.

Input Info: We know that citrus intake causes vitamin C deficiency, and we know that sufficient vitamin C causes straight hair.
Question: Would the patient has curly hair if citrus intake instead of absence of citrus?
Answer (Yes or No ?): Let X = eating citrus; V2 = vitmain C; Y = curly hair. The causal relations are: X->V2,V2->Y. Set Y_{X=1} = 1 | , then solve for Y, given the evidence and the action. V2 = not X\nY = not V2. Then we get Y = [1] = not 0. Thus, the patient would have curly hair if citrus intake instead of absence of citrus. Therefore, the answer is Yes.

Input Info: We know that zuph causes not rixq. zuph and rixq causes xevu. We observed an individual is zuph.
Question: Would an individual is not xevu if not rixq instead of rixq?
Answer (Yes or No ?): Let V1 = zuph; X = rixq; Y = xevu. The causal relations are: V1->X,V1->Y,X->Y. Set Y_{X=0} = 0 | V1=1, then solve for Y, given the evidence and the action. V1 = 1\nX = not V1\nY = V1 and X. Then we get Y = 0 = 1 and 0. Thus, an individual would not be xevu if not rixq instead of rixq. Therefore, the answer is Yes.

Input Info: %s
Question: %s
Answer (Yes or No ?):
""",
    'manual-CoT-CN': """如下为三个使用思维链进行推理的有关反事实的问题：

输入信息：我们知道丈夫设置闹钟会导致妻子设置闹钟，丈夫设置闹钟或妻子设置闹钟会导致闹钟响铃。
问题：如果丈夫没有设置闹钟，而不是丈夫设置闹钟，第二天早上闹钟会响吗？
答案（是或否？）：令 X = 丈夫; V2 = 妻子; Y = 闹钟响铃; 该问题下因果关系有：X->V2,X->Y,V2->Y。令Y_{X=0} = 1 | , 在已知事实和动作下求解Y。V2 = X\nY = X or V2。解得Y = 0 = 0 or 0。因此如果丈夫没有设置闹钟，而不是丈夫设置闹钟，第二天早上闹钟不会响。因此答案为“否”。

输入信息：我们知道晚起床和交通拥堵会导致准时到校，我们观察到路上有严重的交通堵塞。
问题：如果爱丽丝晚起床而不是准时起床，她会上学迟到吗？
答案（是或否？）：令 V2 = 交通; X = 爱丽丝起床; Y = 爱丽丝到学校; 该问题下因果关系有：X->Y,V2->Y。令Y_{X=1} = 0 | V2=1，在在已知事实和动作下求解Y。V2 = 1\nY = X and V2。解得Y = 1 = 1 and 1。因此如果爱丽丝晚起床而不是准时起床，她不会上学迟到。因此答案为“否”。

输入信息：我们知道摄入柑橘会导致维生素C缺乏，我们也知道摄入足够的维生素C会导致坏血病。
问题：如果患者摄入柑橘而不是不摄入柑橘，他会从坏血病中康复吗？
答案（是或否？）：令 X = 摄入柑橘; V2 = 维生素C; Y = 坏血病; 该问题下因果关系有：X->V2,V2->Y. Set Y_{X=1} = 0 | ，在在已知事实和动作下求解Y。V2 = not X\nY = V2。解得Y = [0] = 0。因此如果患者摄入柑橘而不是不摄入柑橘，他会从坏血病中康复。因此答案为“是”。

输入信息：%s
问题：%s
答案（是或否？）：""",
    'explicit-function':
    """You are a helpful assistant for deterministic counterfactual.
Input Info: %s
Question: %s
Answer (Yes or No ?):""",
    'explicit-function-CN': """你是用于决定论反事实的得力助手。
输入信息：%s
问题：%s
答案（是或否？）""",
}


def get_prompt(task_name, prompt_style, item, prompt_style_str=''):
    base = base_prompt_dict[prompt_style]

    prompt = prompt_style_str + base % (item['given_info'], item['question'])
    return prompt
