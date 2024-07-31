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
    """Answer questions about the Natural Direct Effect (NDE). Computing the Natural Direct Effect involves comparing the outcomes for individuals under two scenarios: receiving the treatment and not receiving the treatment, while allowing a mediator variable to take its natural course under each scenario.
Input Info: %s
%s
Instruction: %s
Question: %s
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}:""",
    'zero-shot-IcL-CN':
    """回答有关自然直接效应（NDE）的问题。计算自然直接效应需要比较两种情况下的个人结果：接受治疗和不接受治疗，同时允许中介变量在每种情况下自然发展。
输入信息：%s
%s
指令：%s
问题：%s
请根据上述信息，给出计算结果（答案保留四位小数），并给出最终答案“是“或”否“。请以JSON格式返回最终结果，例如，{"ANSWER":"是","PROB":"0.1234"}：""",
    'one-shot-IcL':
    """Answer questions about the Natural Direct Effect (NDE). Computing the Natural Direct Effect involves comparing the outcomes for individuals under two scenarios: receiving the treatment and not receiving the treatment, while allowing a mediator variable to take its natural course under each scenario.

Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: Fbge has a direct effect on vijq. Fbge has a direct effect on twac. Fbge has a direct effect on vdla.
For those with fbge being high, the probability of vdla being low is 0.1851. For those with fbge being low, the probability of vdla being low is 0.5311.
Instruction: Consider the natural direct effect (NDE) of fbge on vdla.
Question: Suppose the mediator keeps constant when fbge is changed to be high, would the vdla have been more likely to be low?
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}: {"ANSWER": "No", "PROB": "-0.3460"}

Input Info: %s
%s
Instruction: %s
Question: %s
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}:""",
    'one-shot-IcL-CN':
    """回答有关自然直接效应（NDE）的问题。计算自然直接效应需要比较两种情况下的个人结果：接受治疗和不接受治疗，同时允许中介变量在每种情况下自然发展。

输入信息：设想一个只有以下条件，而没有其他因素或因果关系的假设世界：Fbge对vijq有直接影响。Fbge对twac有直接影响。Fbge对vdla有直接影响。
在fbge为高的条件下, vdla为低的概率为0.1851。在fbge为低的条件下, vdla为低的概率为0.5311。
指令：考虑fbge作用于vdla的“自然直接效果”(natural direct effect, NDE)。
问题：假如所有中间变量保持不变，而fbge变化为高，那么vdla更有可能为低吗？
请根据上述信息，给出计算结果（答案保留四位小数），并给出最终答案“是“或”否“。请以JSON格式返回最终结果，例如，{"ANSWER":"是","PROB":"0.1234"}： {"ANSWER":"否","PROB":"-0.3460"}

输入信息：%s
%s
指令：%s
问题：%s
请根据上述信息，给出计算结果（答案保留四位小数），并给出最终答案“是“或”否“。请以JSON格式返回最终结果，例如，{"ANSWER":"是","PROB":"0.1234"}：""",
    'two-shot-IcL':
    """Answer questions about the Natural Direct Effect (NDE). Computing the Natural Direct Effect involves comparing the outcomes for individuals under two scenarios: receiving the treatment and not receiving the treatment, while allowing a mediator variable to take its natural course under each scenario.

Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: Fbge has a direct effect on vijq. Fbge has a direct effect on twac. Fbge has a direct effect on vdla.
For those with fbge being high, the probability of vdla being low is 0.1851. For those with fbge being low, the probability of vdla being low is 0.5311.
Instruction: Consider the natural direct effect (NDE) of fbge on vdla.
Question: Suppose the mediator keeps constant when fbge is changed to be high, would the vdla have been more likely to be low?
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}: {"ANSWER": "No", "PROB": "-0.3460"}

Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: Parent has a direct effect on first-born child. Parent has a direct effect on second-born child. Parent has a direct effect on third-born child. First-born child has a direct effect on second-born child. First-born child has a direct effect on third-born child.
For those with parent being supportive, the probability of first-born child being favored is 0.2759. For those with parent being neglectful, the probability of first-born child being favored is 0.3249.
Instruction: Consider the natural direct effect (NDE) of parent on first-born child.
Question: Suppose the mediator keeps constant when parent is changed to be supportive, would the first-born child have been more likely to be favored?
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}: {"ANSWER": "No", "PROB": "-0.0490"}

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
    """Here are three examples for math problems about natural direct effect (NDE) task with chain of thought.

Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: Vqpf has a direct effect on uhxm. Vqpf has a direct effect on ezwx.
For those with vqpf being high, the probability of uhxm being low is 0.8005. For those with vqpf being low, the probability of uhxm being low is 0.8489.
Instruction: Consider the natural direct effect (NDE) of vqpf on uhxm.
Question: Suppose the mediator keeps constant when vqpf is changed to be high, would the uhxm have been more likely to be low?
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}: With A represents vqpf and B represents uhxm, we have P(B=0|A=1)=0.8005; P(B=0|A=0)=0.8489; Considering edge A->B exists, and in this situation a valid mediator set: empty set, we calculate NDE=P(B=0|A=1)-P(B=0|A=0)=0.8005-0.8489=-0.0484<0. The answer is: {"ANSWER": "No", "PROB": "-0.0484"}.

Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: Number of hours of studying for a test has a direct effect on test score. Test score has a direct effect on final grade in the class.

Instruction: Consider the natural direct effect (NDE) of number of hours of studying for a test on final grade in the class.
Question: Suppose the mediator keeps constant when number of hours of studying for a test is changed to be many, would the final grade in the class have been more likely to be high?
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}: With A represents number of hours of studying for a test and C represents final grade in the class, the edge A->C does not exist. The answer is: {"ANSWER": "No", "PROB": "0.0000"}.

Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: Eosj has a direct effect on taaz. Taaz has a direct effect on sozj. Sozj has a direct effect on mffx.
For those with taaz being high, the probability of sozj being high is 0.4763. For those with taaz being low, the probability of sozj being high is 0.3920.
Instruction: Consider the natural direct effect (NDE) of taaz on sozj.
Question: Suppose the mediator keeps constant when taaz is changed to be high, would the sozj have been more likely to be high?
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}: With B represents taaz and C represents sozj, we have P(C=1|B=1)=0.4763; P(C=1|B=0)=0.3920; Considering edge B->C exists, and in this situation, NDE=P(C=1|B=1)-P(C=1|B=0)=0.4763-0.3920=0.0843>0. The answer is: {"ANSWER": "Yes", "PROB": "0.0843
"}.

Input Info: %s
%s
Instruction: %s
Question: %s
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}:""",
    'manual-CoT-CN':
    """如下为一个使用思维链进行推理的关于“自然直接效果”(natural direct effect, NDE)任务的数学问题：

输入信息：设想一个只有以下条件，而没有其他因素或因果关系的假设世界：顾客对产品的满意度对产品的正面评价数量有直接影响。顾客对产品的满意度对产品收入有直接影响。产品的正面评价数量对产品销售表现有直接影响。产品销售表现对产品收入有直接影响。
在顾客对产品的满意度为低的条件下, 产品的正面评价数量为高的概率为0.4636。在顾客对产品的满意度为高的条件下, 产品的正面评价数量为高的概率为0.9016。
指令：考虑顾客对产品的满意度作用于产品的正面评价数量的“自然直接效果”(natural direct effect, NDE)。
问题：假如所有中间变量保持不变，而顾客对产品的满意度变化为低，那么产品的正面评价数量更有可能为高吗？
请根据上述信息，给出计算结果（答案保留四位小数），并给出最终答案“是“或”否“。请以JSON格式返回最终结果，例如，{"ANSWER":"是","PROB":"0.1234"}：用A代表顾客对产品的满意度, B代表产品的正面评价数量，边A->B存在。考虑到P(B=1|A=0)=0.4636，P(B=1|A=1)=0.9016，且该问题中有一个合法的中间变量集合: 空集。所以NDE=P(B=1|A=0)-P(B=1|A=1)=0.4636-0.9016=-0.4380<0。因此答案为{"ANSWER":"否,”PROB":"-0.4380"}。

输入信息：%s
%s
指令：%s
问题：%s
请根据上述信息，给出计算结果（答案保留四位小数），并给出最终答案“是“或”否“。请以JSON格式返回最终结果，例如，{"ANSWER":"是","PROB":"0.1234"}：""",
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
