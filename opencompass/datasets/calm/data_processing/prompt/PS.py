# flake8: noqa: E501
base_prompt_dict = {
    'basic':
    """Input Info: %s
%s
Instruction: %s
Question: %s
Provide the calculation result to four decimal places in JSON format, like {"PROB": "0.1234"}:""",
    'basic-CN':
    """输入信息：%s
%s
指令：%s
问题：%s
请根据上述信息，给出计算结果（答案保留四位小数）。请以JSON格式返回最终结果，例如，{"PROB":"0.1234"}：""",
    'adversarial-ignore':
    """Input Info: %s
%s
Instruction: %s
Question: %s
Provide the calculation result to four decimal places in JSON format, like {"PROB": "0.1234"}:""",
    'adversarial-ignore-CN':
    """输入信息：%s
%s
指令：%s
问题：%s
请根据上述信息，给出计算结果（答案保留四位小数）。请以JSON格式返回最终结果，例如，{"PROB":"0.1234"}：""",
    'adversarial-doubt':
    """Input Info: %s
%s
Instruction: %s
Question: %s
Provide the calculation result to four decimal places in JSON format, like {"PROB": "0.1234"}:""",
    'adversarial-doubt-CN':
    """输入信息：%s
%s
指令：%s
问题：%s
请根据上述信息，给出计算结果（答案保留四位小数）。请以JSON格式返回最终结果，例如，{"PROB":"0.1234"}：""",
    'zero-shot-IcL':
    """Answer questions about the Probability of Sufficiency (PS). Calculating the Probability of Sufficiency involves looking at the outcomes of individuals who received the treatment and experienced the desired effect. The Probability of Sufficiency is the proportion of these individuals for whom the treatment was enough to achieve the outcome, even if other pathways could also have led to the same result.
Input Info: %s
%s
Instruction: %s
Question: %s
Provide the calculation result to four decimal places in JSON format, like {"PROB": "0.1234"}:""",
    'zero-shot-IcL-CN':
    """回答有关充分概率 (PS) 的问题。计算 "充分概率 "需要查看接受治疗并取得预期效果的个体的结果。充分概率是指即使其他途径也可能导致相同结果，但对这些人来说，治疗足以实现结果的比例。
输入信息：%s
%s
指令：%s
问题：%s
请根据上述信息，给出计算结果（答案保留四位小数）。请以JSON格式返回最终结果，例如，{"PROB":"0.1234"}：""",
    'one-shot-IcL':
    """Answer questions about the Probability of Sufficiency (PS). Calculating the Probability of Sufficiency involves looking at the outcomes of individuals who received the treatment and experienced the desired effect. The Probability of Sufficiency is the proportion of these individuals for whom the treatment was enough to achieve the outcome, even if other pathways could also have led to the same result.

Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: Level of education has a direct effect on job performance. Job performance has a direct effect on salary. Job performance has a direct effect on job satisfaction. Salary has a direct effect on job satisfaction.
For those with job performance being excellent, the probability of salary being low is 0.0539. The probability of salary being low is 0.0857. The probability of job performance being poor and salary being low is 0.0585.
Instruction: Consider the probability of sufficiency (PS) of job performance on salary.
Question: Given that job performance was poor and salary was low, what is the lower bound of the probability that salary would have been high if the job performance had been excellent?
Provide the calculation result to four decimal places in JSON format, like {"PROB": "0.1234"}: {"PROB": "0.5436"}

Input Info: %s
%s
Instruction: %s
Question: %s
Provide the calculation result to four decimal places in JSON format, like {"PROB": "0.1234"}:""",
    'one-shot-IcL-CN':
    """回答有关充分概率 (PS) 的问题。计算 "充分概率 "需要查看接受治疗并取得预期效果的个体的结果。充分概率是指即使其他途径也可能导致相同结果，但对这些人来说，治疗足以实现结果的比例。

输入信息：设想一个只有以下条件，而没有其他因素或因果关系的假设世界：教育水平对工作表现有直接影响。工作表现对薪水有直接影响。工作表现对工作满意度有直接影响。薪水对工作满
意度有直接影响。
在工作表现为出色的条件下, 薪水为低的概率为0.0539。薪水为低的概率为0.0857。工作表现为差劲且薪水为低的概率为0.0585。
指令：考虑工作表现作用于薪水的充分性概率(probability of sufficiency, PS)。
问题：给定工作表现为差劲且薪水为低, 假如工作表现为出色，此时薪水为高的概率的下界是多少？
请根据上述信息，给出计算结果（答案保留四位小数），并给出最终答案“是“或”否“。请以JSON格式返回最终结果，例如，{"ANSWER":"是","PROB":"0.1234"}： {"PROB":"0.5436"}

输入信息：%s
%s
指令：%s
问题：%s
请根据上述信息，给出计算结果（答案保留四位小数）。请以JSON格式返回最终结果，例如，{"PROB":"0.1234"}：""",
    'two-shot-IcL':
    """Answer questions about the Probability of Sufficiency (PS). Calculating the Probability of Sufficiency involves looking at the outcomes of individuals who received the treatment and experienced the desired effect. The Probability of Sufficiency is the proportion of these individuals for whom the treatment was enough to achieve the outcome, even if other pathways could also have led to the same result.

Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: Level of education has a direct effect on job performance. Job performance has a direct effect on salary. Job performance has a direct effect on job satisfaction. Salary has a direct effect on job satisfaction.
For those with job performance being excellent, the probability of salary being low is 0.0539. The probability of salary being low is 0.0857. The probability of job performance being poor and salary being low is 0.0585.
Instruction: Consider the probability of sufficiency (PS) of job performance on salary.
Question: Given that job performance was poor and salary was low, what is the lower bound of the probability that salary would have been high if the job performance had been excellent?
Provide the calculation result to four decimal places in JSON format, like {"PROB": "0.1234"}: {"PROB": "0.5436"}

Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: Ajlk has a direct effect on mzbw. Mzbw has a direct effect on bduo. Mzbw has a direct effect on vlmn. Bduo has a direct effect on vlmn.
For those with ajlk being high, the probability of mzbw being low is 0.1978. The probability of mzbw being low is 0.2593. The probability of ajlk being low and mzbw being low is 0.1797.
Instruction: Consider the probability of sufficiency (PS) of ajlk on mzbw.
Question: Given that ajlk was low and mzbw was low, what is the lower bound of the probability that mzbw would have been high if the ajlk had been high?
Provide the calculation result to four decimal places in JSON format, like {"PROB": "0.1234"}: {"PROB": "0.3422"}

Input Info: %s
%s
Instruction: %s
Question: %s
Provide the calculation result to four decimal places in JSON format, like {"PROB": "0.1234"}:""",
    'zero-shot-CoT':
    """Input Info: %s
%s
Instruction: %s
Question: %s Let's think step by step.
Provide the calculation result to four decimal places in JSON format, like {"PROB": "0.1234"}:""",
    'zero-shot-CoT-CN':
    """输入信息：%s
%s
指令：%s
问题：%s请逐步思考。
请根据上述信息，给出计算结果（答案保留四位小数）。请以JSON格式返回最终结果，例如，{"PROB":"0.1234"}：""",
    'manual-CoT':
    """Here are two examples for math problems about probability of sufficiency (PS) task with chain of thought.

Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: Time spent exercising has a direct effect on physical fitness level. Time spent exercising has a direct effect on overall health condition.
For those with time spent exercising being adequate, the probability of overall health condition being poor is 0.0635. The probability of overall health condition being poor is 0.0912. The probability of time spent exercising being not enough and overall health condition being poor is 0.0534.
Instruction: Consider the probability of sufficiency (PS) of time spent exercising on overall health condition.
Question: Given that time spent exercising was not enough and overall health condition was poor, what is the lower bound of the probability that overall health condition would have been good if the time spent exercising had been adequate?
Provide the calculation result to four decimal places in JSON format, like {"PROB": "0.1234"}: With A represents time spent exercising and C represents overall health condition, we have P(C=0|A=1)=0.0635; P(C=0)=0.0912; P(A=0,C=0)=0.0534; CalculateP(C=0|do(A=1))=P(C=0|A=1)=0.0635, then the lower bound of PS is max{0, [P(C=0)-P(C=0|do(A=1))]/P(A=0,C=0)}\n=max{0, (0.0912-0.0635)/0.0534}\n=max{0, 0.5187}\n=0.5187. The answer is: {"PROB": "0.5187"}.

Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: Iykj has a direct effect on nptw. Iykj has a direct effect on tmex. Nptw has a direct effect on sgnl. Sgnl has a direct effect on tmex.
For those with nptw being high, the probability of sgnl being low is 0.0632. The probability of sgnl being low is 0.0781. The probability of nptw being low and sgnl being low is 0.0375.
Instruction: Consider the probability of sufficiency (PS) of nptw on sgnl.
Question: Given that nptw was low and sgnl was low, what is the lower bound of the probability that sgnl would have been high if the nptw had been high?
Provide the calculation result to four decimal places in JSON format, like {"PROB": "0.1234"}: With B represents nptw, C represents sgnl, we have P(C=0|B=1)=0.0632; P(C=0)=0.0781; P(B=0,C=0)=0.0375; Calculate P(C=0|do(B=1))=P(C=0|B=1)=0.0632, then the lower bound of PS is max{0, [P(C=0)-P(C=0|do(B=1))]/P(B=0,C=0)}\n=max{0, (0.0781-0.0632)/0.0375}\n=max{0, 0.3973}\n=0.3973. The answer is: {"PROB": "0.3973"}.

Input Info: %s
%s
Instruction: %s
Question: %s
Provide the calculation result to four decimal places in JSON format, like {"PROB": "0.1234"}:""",
    'manual-CoT-CN':
    """如下为一个使用思维链进行推理的关于充分性概率(probability of sufficiency, PS)的数学问题：

输入信息：设想一个只有以下条件，而没有其他因素或因果关系的假设世界：Clwa对hvxd有直接影响。Clwa对szak有直接影响。
在clwa为高的条件下, hvxd为低的概率为0.5569。hvxd为低的概率为0.6454。clwa为低且hvxd为低的概率为0.3623。
指令：考虑clwa作用于hvxd的充分性概率(probability of sufficiency, PS)。
问题：给定clwa为低且hvxd为低, 假如clwa为高，此时hvxd为高的概率的下界是多少？
请根据上述信息，给出计算结果（答案保留四位小数）。请以JSON格式返回最终结果，例如，{"PROB":"0.1234"}：用A代表clwa, B代表hvxd，所以P(B=0|A=1)=0.5569; P=0.6454; P(A=0,B=0)=0.3623; 计算P(B=0|do)=P(B=0|A=1)=0.5569，所以PS的下界:为max{0, [P-P(B=0|do)]/P(A=0,B=0)}\n=max{0, (0.6454-0.5569)/0.3623}\n=max{0, 0.2443}\n=0.2443。因此答案为{"PROB":"0.2443"}。

输入信息：%s
%s
指令：%s
问题：%s
请根据上述信息，给出计算结果（答案保留四位小数）。请以JSON格式返回最终结果，例如，{"PROB":"0.1234"}：""",
    'explicit-function':
    """You are a helpful assistant for math probability.
Input Info: %s
%s
Instruction: %s
Question: %s
Provide the calculation result to four decimal places in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}:""",
    'explicit-function-CN':
    """你是一个用于计算数学概率的得力助手。
输入信息：%s
%s
指令：%s
问题：%s
请根据上述信息，给出计算结果（答案保留四位小数）。请以JSON格式返回最终结果，例如，{"ANSWER":"是","PROB":"0.1234"}：""",
}


def get_prompt(task_name, prompt_style, item, prompt_style_str=''):
    base = base_prompt_dict[prompt_style]

    prompt = prompt_style_str + base % (item['given_info'],
                                        item['Background']['data_info'],
                                        item['Instruction'], item['Question'])
    return prompt
