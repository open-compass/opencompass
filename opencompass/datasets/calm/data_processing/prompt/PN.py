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
    """Answer questions about the Probability of Necessity (PN). Calculating the Probability of Necessity involves examining the outcomes of individuals who received the treatment and experienced the desired effect. The Probability of Necessity is the proportion of these individuals for whom the treatment was essential to achieve the outcome, meaning they would not have achieved the outcome without the treatment.
Input Info: %s
%s
Instruction: %s
Question: %s
Provide the calculation result to four decimal places in JSON format, like {"PROB": "0.1234"}:""",
    'zero-shot-IcL-CN':
    """回答有关必要性概率 (PN) 的问题。必要概率的计算涉及对接受治疗并取得预期效果的个体的结果进行检查。必要概率是指在这些人中，治疗对取得疗效至关重要的比例，也就是说，如果没有治疗，他们就不会取得疗效。
输入信息：%s
%s
指令：%s
问题：%s
请根据上述信息，给出计算结果（答案保留四位小数）。请以JSON格式返回最终结果，例如，{"PROB":"0.1234"}：
""",
    'one-shot-IcL':
    """Answer questions about the Probability of Necessity (PN). Calculating the Probability of Necessity involves examining the outcomes of individuals who received the treatment and experienced the desired effect. The Probability of Necessity is the proportion of these individuals for whom the treatment was essential to achieve the outcome, meaning they would not have achieved the outcome without the treatment.

Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: A company has a direct effect on company revenue. A company has a direct effect on company expenses. A company has a direct effect on company profit. Company revenue has a direct effect on company expenses.
For those with a company being inefficient, the probability of company revenue being low is 0.3878. The probability of a company being inefficient and company revenue being low is 0.1900. The probability of a company being efficient and company revenue being high is 0.3871.
Instruction: Consider the probability of necessity (PN) of a company on company revenue.
Question: Given that a company was efficient and company revenue was high, what is the upper bound of the probability of the company revenue would have been low if the a company had been inefficient?
Provide the calculation result to four decimal places in JSON format, like {"PROB": "0.1234"}: {"PROB": "0.5110"}

Input Info: %s
%s
Instruction: %s
Question: %s
Provide the calculation result to four decimal places in JSON format, like {"PROB": "0.1234"}:""",
    'one-shot-IcL-CN':
    """回答有关必要性概率 (PN) 的问题。必要概率的计算涉及对接受治疗并取得预期效果的个体的结果进行检查。必要概率是指在这些人中，治疗对取得疗效至关重要的比例，也就是说，如果没有治疗，他们就不会取得疗效。

输入信息：设想一个只有以下条件，而没有其他因素或因果关系的假设世界：一个公司对公司收入有直接影响。一个公司对公司费用有直接影响。一个公司对公司利润有直接影响。公司收入对公司费用
有直接影响。
在一个公司为低效的条件下, 公司收入为低的概率为0.3878。一个公司为低效且公司收入为低的概率为0.1900。一个公司为高效且公司收入为高的概率为0.3871。
指令：考虑一个公司作用于公司收入的必要性概率(probability of necessity, PN)。
问题：给定一个公司为高效且公司收入为高, 那么假如一个公司为低效，此时公司收入为低的概率的上界是多少？
请根据上述信息，给出计算结果（答案保留四位小数），并给出最终答案“是“或”否“。请以JSON格式返回最终结果，例如，{"PROB":"0.1234"}： {"PROB":"0.5110"}

输入信息：%s
%s
指令：%s
问题：%s
请根据上述信息，给出计算结果（答案保留四位小数）。请以JSON格式返回最终结果，例如，{"PROB":"0.1234"}：""",
    'two-shot-IcL':
    """Answer questions about the Probability of Necessity (PN). Calculating the Probability of Necessity involves examining the outcomes of individuals who received the treatment and experienced the desired effect. The Probability of Necessity is the proportion of these individuals for whom the treatment was essential to achieve the outcome, meaning they would not have achieved the outcome without the treatment.

Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: A company has a direct effect on company revenue. A company has a direct effect on company expenses. A company has a direct effect on company profit. Company revenue has a direct effect on company expenses.
For those with a company being inefficient, the probability of company revenue being low is 0.3878. The probability of a company being inefficient and company revenue being low is 0.1900. The probability of a company being efficient and company revenue being high is 0.3871.
Instruction: Consider the probability of necessity (PN) of a company on company revenue.
Question: Given that a company was efficient and company revenue was high, what is the upper bound of the probability of the company revenue would have been low if the a company had been inefficient?
Provide the calculation result to four decimal places in JSON format, like {"PROB": "0.1234"}: {"PROB": "0.5110"}

Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: Czvl has a direct effect on bsaz. Vimz has a direct effect on bsaz. Bsaz has a direct effect on tava.
For those with vimz being low, the probability of bsaz being low is 0.4591. The probability of vimz being low and bsaz being low is 0.1278. The probability of vimz being high and bsaz being high is 0.1813.
Instruction: Consider the probability of necessity (PN) of vimz on bsaz.
Question: Given that vimz was high and bsaz was high, what is the upper bound of the probability of the bsaz would have been low if the vimz had been low?
Provide the calculation result to four decimal places in JSON format, like {"PROB": "0.1234"}: {"PROB": "1.0000"}

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
    """Here are two examples for math problems about probability of necessity (PN) task with chain of thought.

Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: Market demand has a direct effect on amount of exercise. Market demand has a direct effect on weather condition. Market demand has a direct effect on sales performance. Amount of exercise has a direct effect on sales performance. Weather condition has a direct effect on sales performance.
For those with market demand being low, the probability of sales performance being high is 0.3144. The probability of sales performance being high is 0.3216. The probability of market demand being high and sales performance being high is 0.1890.
Instruction: Consider the probability of necessity (PN) of market demand on sales performance.
Question: Given that market demand was high and sales performance was high, what is the lower bound of the probability of the sales performance would have been low if the market demand had been low?
Provide the calculation result to four decimal places in JSON format, like {"PROB": "0.1234"}: With A represents market demand, C represents weather condition and D represents sales performance, we have P(D=1|C=0,A=0)=0.4246; P(A=0)=0.4216; P(D=1|C=0,A=1)=0.4459; P(A=1)=0.5784; P(D=1)=0.3216; P(C=1,D=1)=0.1293; Calculate P(D=1|do(A=0))=P(D=1|A=0)=0.3144, then lower bound of PN is max{0, [P(D=1)-P(D=1|do(A=0))]/P(A=1,D=1)}\n=max{0, (0.3216-0.3144)/0.1890}\n=max{0, 0.0380}\n=0.0380. The answer is:  {"PROB": "0.0380"}.

Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: Mktt has a direct effect on oroo. Mktt has a direct effect on tlxp. Mktt has a direct effect on enck. Oroo has a direct effect on tlxp.
For those with oroo being low and mktt being low, the probability of tlxp being low is 0.5355. The probability of mktt being low is 0.6363. For those with oroo being low and mktt being high, the probability of tlxp being low is 0.2443. The probability of mktt being high is 0.3637. The probability of oroo being low and tlxp being low is 0.3148. The probability of oroo being high and tlxp being high is 0.1731.
Instruction: Consider the probability of necessity (PN) of oroo on tlxp.
Question: Given that oroo was high and tlxp was high, what is the upper bound of the probability of the tlxp would have been low if the oroo had been low?
Provide the calculation result to four decimal places in JSON format, like {"PROB": "0.1234"}: With A represents mktt, B represents oroo and C represents tlxp, we have: P(C=0|B=0,A=0)=0.5355; P(A=0)=0.6363; P(C=0|B=0,A=1)=0.2443; P(A=1)=0.3637; P(B=0,C=0)=0.3148; P(B=1,C=1)=0.1731; Calculate P(C=0|do(B=0))=sum_{A} P(C=0|B=0,A)*P(A)=P(C=0|B=0,A=0)*P(A=0)+P(C=0|B=0,A=1)*P(A=1), then the upper bound of PN is min{1, [P(C=0)|do(B=0)-P(B=0,C=0)]/P(B=1,C=1)}\n=min{1, (0.4296-0.3148)/0.1731}\n=min{1, 0.6632}\n=0.6632. The answer is:  {"PROB": "0.6632"}.

Input Info: %s
%s
Instruction: %s
Question: %s
Provide the calculation result to four decimal places in JSON format, like {"PROB": "0.1234"}:""",
    'manual-CoT-CN':
    """如下为一个使用思维链进行推理的关于必要性概率(probability of necessity, PN)的数学问题：

输入信息：设想一个只有以下条件，而没有其他因素或因果关系的假设世界：Cegl对mwcg有直接影响。Cegl对jeie有直接影响。Mwcg对jeie有直接影响。
在cegl为低的条件下, mwcg为高的概率为0.6879。mwcg为高的概率为0.8162。cegl为高且mwcg为高的概率为0.6351。
指令：考虑cegl作用于mwcg的必要性概率(probability of necessity, PN)。
问题：给定cegl为高且mwcg为高, 那么假如cegl为低，此时mwcg为低的概率的下界是多少？
请根据上述信息，给出计算结果（答案保留四位小数）。请以JSON格式返回最终结果，例如，{"PROB":"0.1234"}：用A代表cegl, B代表mwcg，所以P(B=1|A=0)=0.6879; P=0.8162; P(A=1,B=1)=0.6351; 计算PN的下界为max{0, [P-P(B=1|do)]/P(A=1,B=1)}\n=max{0, (0.8162-0.6879)/0.6351}\n=max{0, 0.2020}\n=0.2020。因此答案为{"PROB":"0.2020"}。

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
