# flake8: noqa: E501
base_prompt_dict = {
    'basic':
    """Input Info: %s
%s
Instruction: %s
Question: %s
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}:""",
    'basic-CN':
    """输入信息：%s
%s
指令：%s
问题：%s
请根据上述信息，给出计算结果（答案保留四位小数），并给出最终答案“是“或”否“。请以JSON格式返回最终结果，例如，{"ANSWER":"是","PROB":"0.1234"}：""",
    'adversarial-ignore':
    """Input Info: %s
%s
Instruction: %s
Question: %s
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}:""",
    'adversarial-ignore-CN':
    """输入信息：%s
%s
指令：%s
问题：%s
请根据上述信息，给出计算结果（答案保留四位小数），并给出最终答案“是“或”否“。请以JSON格式返回最终结果，例如，{"ANSWER":"是","PROB":"0.1234"}：""",
    'adversarial-doubt':
    """Input Info: %s
%s
Instruction: %s
Question: %s
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}:""",
    'adversarial-doubt-CN':
    """输入信息：%s
%s
指令：%s
问题：%s
请根据上述信息，给出计算结果（答案保留四位小数），并给出最终答案“是“或”否“。请以JSON格式返回最终结果，例如，{"ANSWER":"是","PROB":"0.1234"}：""",
    'zero-shot-IcL':
    """Answer questions about the Effect of the Treatment on the Treated (ETT). Computing the Effect of the Treatment on the Treated  involves focusing solely on the individuals who actually received the treatment. You compare their observed outcomes with what their outcomes would have been had they not received the treatment.
Input Info: %s
%s
Instruction: %s
Question: %s
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}:""",
    'zero-shot-IcL-CN':
    """回答有关 "治疗对受试者的影响"（ETT）的问题。计算治疗对受试者的影响时，只需关注实际接受治疗的个体。将观察到的结果与未接受治疗时的结果进行比较。
输入信息：%s
%s
指令：%s
问题：%s
请根据上述信息，给出计算结果（答案保留四位小数），并给出最终答案“是“或”否“。请以JSON格式返回最终结果，例如，{"ANSWER":"是","PROB":"0.1234"}：""",
    'one-shot-IcL':
    """Answer questions about the Effect of the Treatment on the Treated (ETT). Computing the Effect of the Treatment on the Treated  involves focusing solely on the individuals who actually received the treatment. You compare their observed outcomes with what their outcomes would have been had they not received the treatment.

Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: Smku has a direct effect on eons. Smku has a direct effect on pgqh. Arbu has a direct effect on eons. Eons has a direct effect on pgqh.
For those with arbu being low, the probability of eons being high is 0.2617. For those with arbu being high, the probability of eons being high is 0.0291.
Instruction: Consider the effect of treatment on the treated (ETT) of arbu on eons.
Question: For those with arbu being low, if their arbu had been high, would the eons have been more likely to be high?
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}: {"ANSWER": "No", "PROB": "0.2326"}

Input Info: %s
%s
Instruction: %s
Question: %s
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}:""",
    'one-shot-IcL-CN':
    """回答有关 "治疗对受试者的影响"（ETT）的问题。计算治疗对受试者的影响时，只需关注实际接受治疗的个体。将观察到的结果与未接受治疗时的结果进行比较。

输入信息：设想一个只有以下条件，而没有其他因素或因果关系的假设世界：Smku对eons有直接影响。Smku对pgqh有直接影响。Arbu对eons有直接影响。Eons对pgqh有直接影响。
在arbu为低的条件下, eons为高的概率为0.2617。在arbu为高的条件下, eons为高的概率为0.0291。
指令：考虑arbu作用于eons的“对被干预者的干预效果”(effect of treatment on the treated, ETT)。
问题：对于那些arbu为低，假如arbu为高，那么eons更有可能为高吗？
请根据上述信息，给出计算结果（答案保留四位小数），并给出最终答案“是“或”否“。请以JSON格式返回最终结果，例如，{"ANSWER":"是","PROB":"0.1234"}： {"ANSWER":"否","PROB":"0.2326"}

输入信息：%s
%s
指令：%s
问题：%s
请根据上述信息，给出计算结果（答案保留四位小数），并给出最终答案“是“或”否“。请以JSON格式返回最终结果，例如，{"ANSWER":"是","PROB":"0.1234"}：""",
    'three-shot-IcL':
    """Answer questions about the Effect of the Treatment on the Treated (ETT). Computing the Effect of the Treatment on the Treated  involves focusing solely on the individuals who actually received the treatment. You compare their observed outcomes with what their outcomes would have been had they not received the treatment.

Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: Smku has a direct effect on eons. Smku has a direct effect on pgqh. Arbu has a direct effect on eons. Eons has a direct effect on pgqh.
For those with arbu being low, the probability of eons being high is 0.2617. For those with arbu being high, the probability of eons being high is 0.0291.
Instruction: Consider the effect of treatment on the treated (ETT) of arbu on eons.
Question: For those with arbu being low, if their arbu had been high, would the eons have been more likely to be high?
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}: {"ANSWER": "No", "PROB": "0.2326"}

Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: Sales performance has a direct effect on air pressure. Sales performance has a direct effect on work-life balance. Air pressure has a direct effect on quality of teaching. Work-life balance has a direct effect on quality of teaching.

Instruction: Consider the effect of treatment on the treated (ETT) of air pressure on work-life balance.
Question: For those with air pressure being high, if their air pressure had been low, would the work-life balance have been more likely to be high?
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}: {"ANSWER": "No", "PROB": "0.0000"}

Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: Weather conditions has a direct effect on temperature. Weather conditions has a direct effect on humidity. Temperature has a direct effect on humidity. Temperature has a direct effect on precipitation.
For those with weather conditions being good, the probability of humidity being low is 0.8897. For those with weather conditions being bad, the probability of humidity being low is 0.7378.
Instruction: Consider the effect of treatment on the treated (ETT) of weather conditions on humidity.
Question: For those with weather conditions being good, if their weather conditions had been bad, would the humidity have been more likely to be low?
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}: {"ANSWER": "No", "PROB": "0.1519"}

Input Info: %s
%s
Instruction: %s
Question: %s
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}:""",
    'zero-shot-CoT':
    """Input Info: %s
%s
Instruction: %s
Question: %s Let's think step by step.
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}:""",
    'zero-shot-CoT-CN':
    """输入信息：%s
%s
指令：%s
问题：%s请逐步思考。
请根据上述信息，给出计算结果（答案保留四位小数），并给出最终答案“是“或”否“。请以JSON格式返回最终结果，例如，{"ANSWER":"是","PROB":"0.1234"}：""",
    'manual-CoT':
    """Here are three examples for math problems about effect of treatment on the treated (ETT) task with chain of thought.

Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: Exxp has a direct effect on hnzi. Hnzi has a direct effect on mlhx. Ovlq has a direct effect on hnzi. Wtel has a direct effect on mlhx.
For those with ovlq being low, the probability of hnzi being low is 0.5625. For those with ovlq being high, the probability of hnzi being low is 0.5062.
Instruction: Consider the effect of treatment on the treated (ETT) of ovlq on hnzi.
Question: For those with ovlq being low, if their ovlq had been high, would the hnzi have been more likely to be low?
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}: With B represents ovlq and C represents hnzi, we find P(C=0|B=0)=0.5625; P(C=0|B=1)=0.5062; Considering there is a path B->C from B to C, and in this situation empty set is a valid backdoor adjustment set, we calculate: ETT=E[C_{B=0}-C_{B=1}|B=0]=P(C=0|B=0)-P(C=0|B=1)=0.5625-0.5062=0.0563>0. The answer is: {"ANSWER": "No", "PROB": "0.0563"}.

Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: Fcun has a direct effect on xmtp. Xmtp has a direct effect on mwcs. Xmtp has a direct effect on bkzf.
For those with xmtp being low, the probability of mwcs being low is 0.8041. For those with xmtp being high, the probability of mwcs being low is 0.9343.
Instruction: Consider the effect of treatment on the treated (ETT) of xmtp on mwcs.
Question: For those with xmtp being low, if their xmtp had been high, would the mwcs have been more likely to be low?
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}: With B represents xmtp and C represents mwcs, we have P(C=0|B=0)=0.8041; P(C=0|B=1)=0.9343; Considering there is a path B->C from B to C, and in this situation, empty set is a valid backdoor adjustment set, we calculate ETT=E[C_{B=0}-C_{B=1}|B=0]=P(C=0|B=0)-P(C=0|B=1)=0.8041-0.9343=-0.1302<0. The answer is: {"ANSWER": "Yes", "PROB": "-0.1302"}.

Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: Bfgu has a direct effect on fskd. Bfgu has a direct effect on nbzx.

Instruction: Consider the effect of treatment on the treated (ETT) of nbzx on bfgu.
Question: For those with nbzx being high, if their nbzx had been low, would the bfgu have been more likely to be low?
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}: With A represents bfgu and C represents nbzx, there is no path from C to A. The answer is: {"ANSWER": "No", "PROB": "0.0000"}.

Input Info: %s
%s
Instruction: %s
Question: %s
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}:""",
    'manual-CoT-CN':
    """如下为一个使用思维链进行推理的关于“对被干预者的干预效果”(effect of treatment on the treated, ETT)任务的数学问题：

输入信息：设想一个只有以下条件，而没有其他因素或因果关系的假设世界：工作生活平衡水平对收入水平有直接影响。工作生活平衡水平对天赋水平有直接影响。工作生活平衡水平对政府政策有直接影响。收入水平对天赋水平有直接影响。天赋水平对政府政策有直接影响。
在工作生活平衡水平为高的条件下, 政府政策为高的概率为0.1633。在工作生活平衡水平为低的条件下, 政府政策为高的概率为0.5540。
指令：考虑工作生活平衡水平作用于政府政策的“对被干预者的干预效果”(effect of treatment on the treated, ETT)。
问题：对于那些工作生活平衡水平为高，假如工作生活平衡水平为低，那么政府政策更有可能为高吗？
请根据上述信息，给出计算结果（答案保留四位小数）。请以JSON格式返回最终结果，例如，{"ANSWER":"是","PROB":"0.1234"}：用A代表工作生活平衡水平, B代表收入水平, C代表天赋水平, D代表政府政策，A到D有一条或多条有向路径(例如A->B->C->D)，所以节点A是节点D的原因。考虑到P(D=1|A=1)=0.1633，P(D=1|A=0)=0.5540，且该问题中有一个合法的后门调整集合：空集，所以ETT=E[D_{A=1}-D_{A=0}|A=1]=P(D=1|A=1)-P(D=1|A=0)=0.1633-0.5540=-0.3907<0。因此答案为{"ANSWER":"是","PROB":"-0.3907"}。

输入信息：%s
%s
指令：%s
问题：%s
请根据上述信息，给出计算结果（答案保留四位小数）。请以JSON格式返回最终结果，例如，{"ANSWER":"是","PROB":"0.1234"}：""",
    'explicit-function':
    """You are a helpful assistant for math probability.
Input Info: %s
%s
Instruction: %s
Question: %s
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}:""",
    'explicit-function-CN':
    """你是一个用于计算数学概率的得力助手。
输入信息：%s
%s
指令：%s
问题：%s
请根据上述信息，给出计算结果（答案保留四位小数），并给出最终答案“是“或”否“。请以JSON格式返回最终结果，例如，{"ANSWER":"是","PROB":"0.1234"}：""",
}


def get_prompt(task_name, prompt_style, item, prompt_style_str=''):
    base = base_prompt_dict[prompt_style]

    prompt = prompt_style_str + base % (item['given_info'],
                                        item['Background']['data_info'],
                                        item['Instruction'], item['Question'])
    return prompt
