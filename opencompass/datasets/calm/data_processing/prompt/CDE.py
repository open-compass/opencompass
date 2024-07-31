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
    """Answer questions about the Controlled Direct Effect (CDE). Computing the Controlled Direct Effect involves comparing the outcomes of individuals under two scenarios: receiving the treatment and not receiving the treatment, while holding a third variable (the mediator) constant.
Input Info: %s
%s
Instruction: %s
Question: %s
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}:""",
    'zero-shot-IcL-CN':
    """回答有关受控直接效应（CDE）的问题。计算受控直接效应包括比较两种情况下的个人结果：接受治疗和不接受治疗，同时保持第三个变量（中介变量）不变。
输入信息：%s
%s
指令：%s
问题：%s
请根据上述信息，给出计算结果（答案保留四位小数），并给出最终答案“是“或”否“。请以JSON格式返回最终结果，例如，{"ANSWER":"是","PROB":"0.1234"}：""",
    'one-shot-IcL':
    """Answer questions about the Controlled Direct Effect (CDE). Computing the Controlled Direct Effect involves comparing the outcomes of individuals under two scenarios: receiving the treatment and not receiving the treatment, while holding a third variable (the mediator) constant.

Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: Weather conditions has a direct effect on amount of rainfall. Weather conditions has a direct effect on crop yield. Amount of rainfall has a direct effect on crop yield.
For those with weather conditions being good and amount of rainfall being small, the probability of crop yield being high is 0.3510. For those with weather conditions being bad and amount of rainfall being small, the probability of crop yield being high is 0.1420.
Instruction: Consider the controlled direct effect (CDE) of weather conditions on crop yield.
Question: Conditioned on amount of rainfall being small, if the weather conditions had been good, would the crop yield have been more likely to be high?
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}: {"ANSWER": "Yes", "PROB": "0.2090"}

Input Info: %s
%s
Instruction: %s
Question: %s
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}:""",
    'one-shot-IcL-CN':
    """回答有关受控直接效应（CDE）的问题。计算受控直接效应包括比较两种情况下的个人结果：接受治疗和不接受治疗，同时保持第三个变量（中介变量）不变。

输入信息：设想一个只有以下条件，而没有其他因素或因果关系的假设世界：天气状况对降雨量有直接影响。天气状况对农作物产量有直接影响。降雨量对农作物产量有直接影响。
在天气状况为好且降雨量为小的条件下, 农作物产量为高的概率为0.3510。在天气状况为不好且降雨量为小的条件下, 农作物产量为高的概率为0.1420。
指令：考虑天气状况作用于农作物产量的“受控直接效果”(controlled direct effect, CDE)。
问题：在降雨量为小的条件下，假如天气状况为好，那么农作物产量更有可能为高吗？
请根据上述信息，给出计算结果（答案保留四位小数），并给出最终答案“是“或”否“。请以JSON格式返回最终结果，例如，{"ANSWER":"是","PROB":"0.1234"}： {"ANSWER":"是","PROB":"0.2090"}

输入信息：%s
%s
指令：%s
问题：%s
请根据上述信息，给出计算结果（答案保留四位小数），并给出最终答案“是“或”否“。请以JSON格式返回最终结果，例如，{"ANSWER":"是","PROB":"0.1234"}：""",
    'three-shot-IcL':
    """Answer questions about the Controlled Direct Effect (CDE). Computing the Controlled Direct Effect involves comparing the outcomes of individuals under two scenarios: receiving the treatment and not receiving the treatment, while holding a third variable (the mediator) constant.

Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: Weather conditions has a direct effect on amount of rainfall. Weather conditions has a direct effect on crop yield. Amount of rainfall has a direct effect on crop yield.
For those with weather conditions being good and amount of rainfall being small, the probability of crop yield being high is 0.3510. For those with weather conditions being bad and amount of rainfall being small, the probability of crop yield being high is 0.1420.
Instruction: Consider the controlled direct effect (CDE) of weather conditions on crop yield.
Question: Conditioned on amount of rainfall being small, if the weather conditions had been good, would the crop yield have been more likely to be high?
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}: {"ANSWER": "Yes", "PROB": "0.2090"}

Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: Nlta has a direct effect on vhuj. Nlta has a direct effect on huit. Vhuj has a direct effect on xyrs. Vhuj has a direct effect on nxur. Xyrs has a direct effect on huit. Xyrs has a direct effect on nxur. Huit has a direct effect on nxur.

Instruction: Consider the controlled direct effect (CDE) of nlta on nxur.
Question: Conditioned on xyrs being low, vhuj being low and huit being low, if the nlta had been low, would the nxur have been more likely to be high?
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}: {"ANSWER": "No", "PROB": "0.0000"}

Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: Work-life balance has a direct effect on amount of exercise. Work-life balance has a direct effect on appearance. Amount of exercise has a direct effect on appearance.
For those with work-life balance being low and amount of exercise being low, the probability of appearance being low is 0.2287. For those with work-life balance being high and amount of exercise being low, the probability of appearance being low is 0.1287.
Instruction: Consider the controlled direct effect (CDE) of work-life balance on appearance.
Question: Conditioned on amount of exercise being low, if the work-life balance had been low, would the appearance have been more likely to be low?
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}: {"ANSWER": "Yes", "PROB": "0.1000"}

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
    """Here are three examples for math problems about controlled direct effect (CDE) task with chain of thought.

Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: Etsq has a direct effect on ahcp. Etsq has a direct effect on pcit. Ahcp has a direct effect on pcit. Fqyq has a direct effect on pcit.
For those with etsq being low and ahcp being low, the probability of pcit being low is 0.7081. For those with etsq being high and ahcp being low, the probability of pcit being low is 0.5410.
Instruction: Consider the controlled direct effect (CDE) of etsq on pcit.
Question: Conditioned on ahcp being low, if the etsq had been low, would the pcit have been
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}: With A represents etsq, B represents ahcp, and D represents pcit, we have P(D=0|A=0,B=0)=0.7081; P(D=0|A=1,B=0)=0.5410; Considering the edge A->D, and in this situation, empty is a valid backdoor adjustment set, we calculate CDE=P(D=0|do(A=0,B=0))-P(D=0|do(A=1,B=0))=P(D=0|A=0,B=0)-P(D=0|A=1,B=0)=0.7081-0.5410=0.1671>0. The answer is {"ANSWER": "Yes", "PROB": "0.1671"}.

Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: Mental health has a direct effect on temperature. Temperature has a direct effect on government policies.

Instruction: Consider the controlled direct effect (CDE) of mental health on government policies.
Question: Conditioned on temperature being low, if the mental health had been high, would the government policies have been more likely to be high?
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}: With A represents mental health and C represents government policies, the edge A->C does not exist. The answer is {"ANSWER": "No", "PROB": "0.0000"}.

Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: Bobg has a direct effect on viah. Bobg has a direct effect on afgs. Viah has a direct effect on afgs.
For those with bobg being low and viah being high, the probability of afgs being low is 0.2091. For those with bobg being high and viah being high, the probability of afgs being low is 0.5622.
Instruction: Consider the controlled direct effect
Question: Conditioned on viah being high, if the bobg had been low, would the afgs have been more likely to be low?
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}: With A represents bobg, B represents viah and C represents afgs, we find P(C=0|A=0,B=1)=0.2091; P(C=0|A=1,B=1)=0.5622; Considering the edge A->C exists, and in this situation, empty set is a valid backdoor adjustment set, we calculate CDE=P(C=0|do(A=0,B=1))-P(C=0|do(A=1,B=1))=P(C=0|A=0,B=1)-P(C=0|A=1,B=1)=0.2091-0.5622=-0.3531<0. The answer is {"ANSWER": "No", "PROB": "-0.3531"}.

Input Info: %s
%s
Instruction: %s
Question: %s
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}:""",
    'manual-CoT-CN':
    """如下为一个使用思维链进行推理的关于“受控直接效果”(controlled direct effect, CDE)任务的数学问题：

输入信息：设想一个只有以下条件，而没有其他因素或因果关系的假设世界：降雨量对土壤湿度水平有直接影响。降雨量对农作物产量有直接影响。土壤湿度水平对农作物产量有直接影响。
在降雨量为大且土壤湿度水平为湿润的条件下, 农作物产量为高的概率为0.9092。在降雨量为小且土壤湿度水平为湿润的条件下, 农作物产量为高的概率为0.8062。
指令：考虑降雨量作用于农作物产量的“受控直接效果”(controlled direct effect, CDE)。
问题：在土壤湿度水平为湿润的条件下，假如降雨量为大，那么农作物产量更有可能为高吗？
请根据上述信息，给出计算结果（答案保留四位小数）。请以JSON格式返回最终结果，例如，{"ANSWER":"是","PROB":"0.1234"}：用A代表降雨量, B代表土壤湿度水平, C代表农作物产量，边A->C存在。考虑到P(C=1|A=1,B=1)=0.9092，P(C=1|A=0,B=1)=0.8062，且该问题中有一个合法的后门调整集合：空集，所以CDE=P(C=1|do(A=1,B=1))-P(C=1|do(A=0,B=1))=P(C=1|A=1,B=1)-P(C=1|A=0,B=1)=0.9092-0.8062=0.1030>0。因此答案为{"ANSWER":"是","PROB":"0.1030"}。

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
