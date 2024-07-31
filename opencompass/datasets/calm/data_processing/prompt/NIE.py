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
    """Answer questions about the Natural Indirect Effect (NIE). Computing the Natural Indirect Effect involves looking at the outcomes for individuals when the treatment is fixed but the mediator is allowed to change as it naturally would due to the treatment.
Input Info: %s
%s
Instruction: %s
Question: %s
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}:""",
    'zero-shot-IcL-CN':
    """回答有关自然间接效应（NIE）的问题。自然间接效应的计算方法是，当治疗方法固定不变，但允许中介因子因治疗方法而自然发生变化时，研究个体的结果。
输入信息：%s
%s
指令：%s
问题：%s
请根据上述信息，给出计算结果（答案保留四位小数），并给出最终答案“是“或”否“。请以JSON格式返回最终结果，例如，{"ANSWER":"是","PROB":"0.1234"}：""",
    'one-shot-IcL':
    """Answer questions about the Natural Indirect Effect (NIE). Computing the Natural Indirect Effect involves looking at the outcomes for individuals when the treatment is fixed but the mediator is allowed to change as it naturally would due to the treatment.

Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: Income level has a direct effect on job satisfaction. Income level has a direct effect on life satisfaction. Education level has a direct effect on job satisfaction. Education level has a direct effect on happiness level. Job satisfaction has a direct effect on happiness level. Job satisfaction has a direct effect on life satisfaction. Happiness level has a direct effect on life satisfaction.
For those with job satisfaction being not satisfied and education level being high, the probability of happiness level being low is 0.2180. For those with education level being low, the probability of job satisfaction being not satisfied is 0.5969. For those with education level being high, the probability of job satisfaction being not satisfied is 0.4075. For those with job satisfaction being satisfied and education level being high, the probability of happiness level being low is 0.1982.
Instruction: Consider the natural indirect effect (NIE) of education level on happiness level.
Question: Suppose education level is held constant and the mediator changes to whatever value it would have attained under education level changing to be low, would happiness level have been more likely to be low?
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}: {"ANSWER": "Yes", "PROB": "0.0038"}

Input Info: %s
%s
Instruction: %s
Question: %s
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}:""",
    'one-shot-IcL-CN':
    """回答有关自然间接效应（NIE）的问题。自然间接效应的计算方法是，当治疗方法固定不变，但允许中介因子因治疗方法而自然发生变化时，研究个体的结果。

输入信息：设想一个只有以下条件，而没有其他因素或因果关系的假设世界：收入水平对工作是否满意有直接影响。收入水平对生活是否满意有直接影响。教育水平对工作是否满意有直接影响。教育水平对幸福水平有直接影响。工作是否满意对幸福水平有直接影响。工作是否满意对生活是否满意有直接影响。幸福水平对生活是否满意有直接影响。
在工作是否满意为不满意且教育水平为高的条件下, 幸福水平为低的概率为0.2180。在教育水平为低的条件下, 工作是否满意为不满意的概率为0.5969。在教育水平为高的条件下, 工作是否满意为不满意的概率为0.4075。在工作是否满意为满意且教育水平为高的条件下, 幸福水平为低的概率为0.1982。
指令：考虑教育水平作用于幸福水平的“自然间接效果”(natural indirect effect, NIE)。
问题：假如教育水平保持不变，而所有中间变量被改变为当它们在教育水平变化为低下的取值，那么幸福水平更有可能为低吗？
请根据上述信息，给出计算结果（答案保留四位小数），并给出最终答案“是“或”否“。请以JSON格式返回最终结果，例如，{"ANSWER":"是","PROB":"0.1234"}： {"ANSWER":"是","PROB":"0.0038"}

输入信息：%s
%s
指令：%s
问题：%s
请根据上述信息，给出计算结果（答案保留四位小数），并给出最终答案“是“或”否“。请以JSON格式返回最终结果，例如，{"ANSWER":"是","PROB":"0.1234"}：""",
    'two-shot-IcL':
    """Answer questions about the Natural Indirect Effect (NIE). Computing the Natural Indirect Effect involves looking at the outcomes for individuals when the treatment is fixed but the mediator is allowed to change as it naturally would due to the treatment.

Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: Income level has a direct effect on job satisfaction. Income level has a direct effect on life satisfaction. Education level has a direct effect on job satisfaction. Education level has a direct effect on happiness level. Job satisfaction has a direct effect on happiness level. Job satisfaction has a direct effect on life satisfaction. Happiness level has a direct effect on life satisfaction.
For those with job satisfaction being not satisfied and education level being high, the probability of happiness level being low is 0.2180. For those with education level being low, the probability of job satisfaction being not satisfied is 0.5969. For those with education level being high, the probability of job satisfaction being not satisfied is 0.4075. For those with job satisfaction being satisfied and education level being high, the probability of happiness level being low is 0.1982.
Instruction: Consider the natural indirect effect (NIE) of education level on happiness level.
Question: Suppose education level is held constant and the mediator changes to whatever value it would have attained under education level changing to be low, would happiness level have been more likely to be low?
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}: {"ANSWER": "Yes", "PROB": "0.0038"}

Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: Sslg has a direct effect on gjot. Sslg has a direct effect on hlky. Etat has a direct effect on gjot. Etat has a direct effect on hlky. Gjot has a direct effect on hlky.

Instruction: Consider the natural indirect effect (NIE) of sslg on gjot.
Question: Suppose sslg is held constant and the mediator changes to whatever value it would have attained under sslg changing to be low, would gjot have been more likely to be high?
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}: {"ANSWER": "No", "PROB": "0.0000"}

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
    """Here are three examples for math problems about natural indirect effect (NIE) task with chain of thought.

Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: Alor has a direct effect on geer. Tnkc has a direct effect on dzww. Dzww has a direct effect on geer.
For those with dzww being low and tnkc being low, the probability of geer being high is 0.2261. For those with tnkc being high, the probability of dzww being low is 0.9090. For those with tnkc being low, the probability of dzww being low is 0.4752. For those with dzww being high and tnkc being low, the probability of geer being high is 0.0652.
Instruction: Consider the natural indirect effect (NIE) of tnkc on geer.
Question: Suppose tnkc is held constant and the mediator changes to whatever value it would have attained under tnkc changing to be high, would geer have been more likely to be high?
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}: With B represents tnkc, C represents dzww and D represents geer, we find: P(D=1|C=0,B=0)=0.2261; P(C=0|B=1)=0.9090; P(C=0|B=0)=0.4752; P(D=1|C=1,B=0)=0.0652; Considering there is an indirect connect between B and D(B->C->D), and in this situation, we find a valid mediator set: {C}, we calculate NIE=sum_{C} P(D=1|B=0,C)*[P(C|B=1)-P(C|B=0)]=P(D=1|B=0,C=0)*[P(C=0|B=1)-P(C=0|B=0)]+P(D=1|B=0,C=1)*[P(C=1|B=1)-P(C=1|B=0)]=0.2261*(0.9090-0.4752)+0.0652*(0.0910-0.5248)=0.0698>0. The answer is: {"ANSWER": "Yes", "PROB": "0.0698"}.

Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: Vfvq has a direct effect on dupa. Vfvq has a direct effect on fbzr. Xizv has a direct effect on dupa.

Instruction: Consider the natural indirect effect (NIE) of vfvq on fbzr.
Question: Suppose vfvq is held constant and the mediator changes to whatever value it would have attained under vfvq changing to be high, would fbzr have been more likely to be high?
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}: With A represents vfvq and D represents fbzr, there is no indirect connect between A and D. The answer is: {"ANSWER": "No", "PROB": "0.0000"}.

Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: Number of hours of studying for a test has a direct effect on test score. Test score has a direct effect on final grade in the class.
For those with test score being low and number of hours of studying for a test being few, the probability of final grade in the class being low is 0.9320. For those with number of hours of studying for a test being many, the probability of test score being low is 0.2929. For those with number of hours of studying for a test being few, the probability of test score being low is 0.4453. For those with test score being high and number of hours of studying for a test being few, the probability of final grade in the class being low is 0.6552.
Instruction: Consider the natural indirect effect (NIE) of number of hours of studying for a test on final grade in the class.
Question: Suppose number of hours of studying for a test is held constant and the mediator changes to whatever value it would have attained under number of hours of studying for a test changing to be many, would final grade in the class have been more likely to be low?
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}: With A represents number of hours of studying for a test and C represents final grade in the class, we find P(C=0|B=0,A=0)=0.9320; P(B=0|A=1)=0.2929; P(B=0|A=0)=0.4453; P(C=0|B=1,A=0)=0.6552; Considering there is an indirect connect between A and C(A->B->C), and in this situation, we find a valid mediator set: {B}, we calculate NIE=sum_{B} P(C=0|A=0,B)*[P(B|A=1)-P(B|A=0)]=P(C=0|A=0,B=0)*[P(B=0|A=1)-P(B=0|A=0)]+P(C=0|A=0,B=1)*[P(B=1|A=1)-P(B=1|A=0)]=0.9320*(0.2929-0.4453)+0.6552*(0.7071-0.5547)=-0.0422<0. The answer is:  {"ANSWER": "No", "PROB": "-0.0422"}.

Input Info: %s
%s
Instruction: %s
Question: %s
Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}:""",
    'manual-CoT-CN':
    """如下为一个使用思维链进行推理的关于“自然间接效果”(natural indirect effect, NIE)任务的数学问题：

输入信息：设想一个只有以下条件，而没有其他因素或因果关系的假设世界：Zild对vean有直接影响。Zild对dhib有直接影响。Vean对dhib有直接影响。Dhib对maiw有直接影响。
在vean为低且zild为高的条件下, dhib为高的概率为0.5548。在zild为低的条件下, vean为低的概率为0.6871。在zild为高的条件下, vean为低的概率为0.7006。在vean为高且zild为高的条件下, dhib为高的概率为0.9182。
指令：考虑zild作用于dhib的“自然间接效果”(natural indirect effect, NIE)。
问题：假如zild保持不变，而所有中间变量被改变为当它们在zild变化为低下的取值，那么dhib更有可能为高吗？
请根据上述信息，给出计算结果（答案保留四位小数）。请以JSON格式返回最终结果，例如，{"ANSWER":"是","PROB":"0.1234"}：用A代表zild, B代表vean, C代表dhib,所以P(C=1|B=0,A=1)=0.5548; P(B=0|A=0)=0.6871; P(B=0|A=1)=0.7006; P(C=1|B=1,A=1)=0.9182; 考虑到从A到C存在节点数大于等于3的有向路径(例如 A->B->C)，且该问题中有一个合法的中间变量集合: {B}，所以NIE=sum_{B} P(C=1|A=1,B)*[P(B|A=0)-P(B|A=1)]=P(C=1|A=1,B=0)*[P(B=0|A=0)-P(B=0|A=1)]+P(C=1|A=1,B=1)*[P(B=1|A=0)-P(B=1|A=1)]=0.5548*(0.6871-0.7006)+0.9182*(0.3129-0.2994)=0.0049>0。因此答案为{"ANSWER":"是",”PROB":"0.0049"}。

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
