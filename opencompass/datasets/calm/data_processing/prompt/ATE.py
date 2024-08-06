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
    """Answer questions about the Average Treatment Effect (ATE). Computing the Average Treatment Effect involves comparing the outcomes of two groups: the treated group and the control group. The ATE is the difference in average outcomes between these two groups.
Input Info: %s
%s
Instruction: %s
Question: %s
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}:""",
    'zero-shot-IcL-CN':
    """回答有关平均处理效应 (ATE) 的问题。计算平均处理效应需要比较两组结果：处理组和对照组。ATE 是这两组之间平均处理效应的差值。
输入信息：%s
%s
指令：%s
问题：%s
请根据上述信息，给出计算结果（答案保留四位小数），并给出最终答案“是“或”否“。请以JSON格式返回最终结果，例如，{"ANSWER":"是","PROB":"0.1234"}：""",
    'one-shot-IcL':
    """Answer questions about the Average Treatment Effect (ATE). Computing the Average Treatment Effect involves comparing the outcomes of two groups: the treated group and the control group. The ATE is the difference in average outcomes between these two groups.

Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: appearance has a direct effect on air pressure. Air pressure has a direct effect on education level.
For those with appearance being high, the probability of education level being high is 0.3192. For those with appearance being low, the probability of education level being high is 0.3100.
Instruction: Consider the average treatment effect (ATE) of appearance on education level.
Question: If appearance is changed to be high, will education level be more likely to be high?
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}: {"ANSWER": "Yes", "PROB": "0.0092"}

Input Info: %s
%s
Instruction: %s
Question: %s
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}:""",
    'one-shot-IcL-CN':
    """回答有关平均处理效应 (ATE) 的问题。计算平均处理效应需要比较两组结果：处理组和对照组。ATE 是这两组之间平均处理效应的差值。

输入信息：设想一个只有以下条件，而没有其他因素或因果关系的假设世界：外貌水平对气压有直接影响。气压对教育水平有直接影响。
在外貌水平为高的条件下, 教育水平为高的概率为0.3192。在外貌水平为低的条件下, 教育水平为高的概率为0.3100。
指令：考虑外貌水平作用于教育水平的“平均干预效果”(average treatment effect, ATE)。
问题：如果外貌水平被改变为高，那么教育水平更有可能为高吗？
请根据上述信息，给出计算结果（答案保留四位小数），并给出最终答案“是“或”否“。请以JSON格式返回最终结果，例如，{"ANSWER":"是","PROB":"0.1234"}： {"ANSWER":"是","PROB":"0.0092"}

输入信息：%s
%s
指令：%s
问题：%s
请根据上述信息，给出计算结果（答案保留四位小数），并给出最终答案“是“或”否“。请以JSON格式返回最终结果，例如，{"ANSWER":"是","PROB":"0.1234"}：""",
    'three-shot-IcL':
    """Answer questions about the Average Treatment Effect (ATE). Computing the Average Treatment Effect involves comparing the outcomes of two groups: the treated group and the control group. The ATE is the difference in average outcomes between these two groups.

Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: appearance has a direct effect on air pressure. Air pressure has a direct effect on education level.
For those with appearance being high, the probability of education level being high is 0.3192. For those with appearance being low, the probability of education level being high is 0.3100.
Instruction: Consider the average treatment effect (ATE) of appearance on education level.
Question: If appearance is changed to be high, will education level be more likely to be high?
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}: {"ANSWER": "Yes", "PROB": "0.0092"}

Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: Alor has a direct effect on geer. Tnkc has a direct effect on dzww. Dzww has a direct effect on geer.

Instruction: Consider the average treatment effect (ATE) of dzww on tnkc.
Question: If dzww is changed to be low, will tnkc be more likely to be high?
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}: {"ANSWER": "No", "PROB": "0.0000"}

Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: The amount of exercise a person does per week has a direct effect on the person's physical fitness level. The amount of exercise a person does per week has a direct effect on the person's risk of developing chronic diseases.
For those with the amount of exercise a person does per week being little, the probability of the person's physical fitness level being excellent is 0.2598. For those with the amount of exercise a person does per week being a lot, the probability of the person's physical fitness level being excellent is 0.5314.
Instruction: Consider the average treatment effect (ATE) of the amount of exercise a person does per week on the person's physical fitness level.
Question: If the amount of exercise a person does per week is changed to be little, will the person's physical fitness level be more likely to be excellent?
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}: {"ANSWER": "No", "PROB": "-0.2716"}

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
    """Here are three examples for math problems about average treatment effect(ATE) task with chain of thought.

Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: Alor has a direct effect on geer. Tnkc has a direct effect on dzww. Dzww has a direct effect on geer.

Instruction: Consider the average treatment effect (ATE) of dzww on tnkc.
Question: If dzww is changed to be low, will tnkc be more likely to be high?
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}: With B represents tnkc and C represents dzww, we find there is no directed path from C to B. The answer is: {"ANSWER": "No", "PROB": "0.0000"}.

Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: Tvkj has a direct effect on clwv. Clwv has a direct effect on bjtk. Bjtk has a direct effect on dmfl.
For those with clwv being low, the probability of dmfl being high is 0.4780. For those with clwv being high, the probability of dmfl being high is 0.4949.
Instruction: Consider the average treatment effect (ATE) of clwv on dmfl.
Question: If clwv is changed to be low, will dmfl be more likely to be high?
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}: With B represents clwv and D represents dmfl, we find P(D=1|B=0)=0.4780; P(D=1|B=1)=0.4949; Considering there is a path B->C->D from B to D, and in this situation, empty set is a valid backdoor adjustment set, we calculate ATE=P(D=1|do(B=0))-P(D=1|do(B=1))=P(D=1|B=0)-P(D=1|B=1)=0.4780-0.4949=-0.0169<0. The answer is:  {"ANSWER": "No", "PROB": "-0.0169"}.

Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: Zavj has a direct effect on nvcm. Nvcm has a direct effect on sxxy.
For those with nvcm being high, the probability of sxxy being high is 0.8173. For those with nvcm being low, the probability of sxxy being high is 0.7873.
Instruction: Consider the average treatment effect (ATE) of nvcm on sxxy.
Question: If nvcm is changed to be high, will sxxy be more likely to be high?
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}: With B represents nvcm and C represents sxxy, we find P(C=1|B=1)=0.8173; P(C=1|B=0)=0.7873; Considering there is a path B->C from B to C, and in this situation empty set is a valid backdoor adjustment set, we calculate ATE=P(C=1|do(B=1))-P(C=1|do(B=0))=P(C=1|B=1)-P(C=1|B=0)=0.8173-0.7873=0.0300>0. The answer is: {"ANSWER": "Yes", "PROB": "0.0300"}.

Input Info: %s
%s
Instruction: %s
Question: %s
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}:""",
    'manual-CoT-CN':
    """如下为一个使用思维链进行推理的关于“平均干预效果”(average treatment effect, ATE)任务的数学问题：

输入信息：设想一个只有以下条件，而没有其他因素或因果关系的假设世界：是否为考试而学习对考试成绩有直接影响。考试成绩对学生是否通过课程有直接影响。
在考试成绩为高的条件下, 学生是否通过课程为不及格的概率为0.9874。在考试成绩为低的条件下, 学生是否通过课程为不及格的概率为0.7798。
指令：考虑考试成绩作用于学生是否通过课程的“平均干预效果”(average treatment effect, ATE)。
问题：如果考试成绩被改变为高，那么学生是否通过课程更有可能为不及格吗？
请根据上述信息，给出计算结果（答案保留四位小数）。请以JSON格式返回最终结果，例如，{"ANSWER":"是","PROB":"0.1234"}：用B代表考试成绩, C代表学生是否通过课程，B到C有一条或多条有向路径(例如B->C)，所以节点B是节点C的原因。考虑到P(C=0|B=1)=0.9874，P(C=0|B=0)=0.7798，且在该问题中有一个合法的后门调整集合：空集，所以ATE=0.9874-0.7798=0.2076>0。因此答案为{"ANSWER":"是","PROB":"0.2076"}。

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
