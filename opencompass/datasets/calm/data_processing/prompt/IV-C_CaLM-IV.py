# flake8: noqa: E501
base_prompt_dict = {
    'basic':
    """You will be presented with a causal graph in the following form: %s.
Question: Which set of variables is the instrumental variables relative to an ordered pair of variables (%s, %s) in the above causal graph?
Option 1: %s
Option 2: %s
Option 3: %s
Answer (Option 1 or Option 2 or Option 3 ?):""",
    'basic-CN':
    """给定如下因果图：%s。
问题：对于上述因果图中的有序变量对 (%s, %s)，工具变量是哪个？
选项一：%s
选项二：%s
选项三：%s
答案（选项一或选项二或选项三？）：""",
    'adversarial-ignore':
    """You will be presented with a causal graph in the following form: %s.
Question: Which set of variables is the instrumental variables relative to an ordered pair of variables (%s, %s) in the above causal graph?
Option 1: %s
Option 2: %s
Option 3: %s
Answer (Option 1 or Option 2 or Option 3 ?):""",
    'adversarial-ignore-CN':
    """给定如下因果图：%s。
问题：对于上述因果图中的有序变量对 (%s, %s)，工具变量是哪个？
选项一：%s
选项二：%s
选项三：%s
答案（选项一或选项二或选项三？）：""",
    'adversarial-doubt':
    """You will be presented with a causal graph in the following form: %s.
Question: Which set of variables is the instrumental variables relative to an ordered pair of variables (%s, %s) in the above causal graph?
Option 1: %s
Option 2: %s
Option 3: %s
Answer (Option 1 or Option 2 or Option 3 ?):""",
    'adversarial-doubt-CN':
    """给定如下因果图：%s。
问题：对于上述因果图中的有序变量对 (%s, %s)，工具变量是哪个？
选项一：%s
选项二：%s
选项三：%s
答案（选项一或选项二或选项三？）：""",
    'zero-shot-IcL':
    """Objective:
Your task is to identify the set of variables that serve as instrumental variables for a given ordered pair of treatment and outcome variables (X, Y) in a specified causal graph.
Background Information:
An instrumental variable Z must meet the following criteria:
- (i) Z has a causal effect on X (treatment variable).
- (ii) Z affects Y (outcome variable) only through its effect on X, and not directly (exclusion restriction).
- (iii) There is no confounding between the effect of Z on Y, meaning that all common causes of Z and Y (if any) are controlled for in the study or are non-existent.
Input:
- Description of the causal graph, denoting which variables have causal relationships with each other.
- A question that specifies the identification of instrumental variables with respect to an ordered pair of variables (X, Y).
- Multiple-choice options representing sets of variables that could be instrumental.
Output:
- Answer in the format "Option N," where N is either 1, 2, or 3 based on the provided options.

You will be presented with a causal graph in the following form: %s.
Question: Which set of variables is the instrument variables relative to an ordered pair of variables (%s, %s) in the above causal graph?
Option 1: %s
Option 2: %s
Option 3: %s
Answer (Option 1 or Option 2 or Option 3 ?):""",
    'zero-shot-IcL-CN':
    """目标
您的任务是在指定的因果关系图中，找出一组变量，作为给定的一对有序处理变量和结果变量（X，Y）的工具变量。
背景信息：
工具变量 Z 必须符合以下标准：
- (i) Z 对 X（治疗变量）有因果效应。
- (ii) Z 仅通过对 X 的影响而非直接影响 Y（结果变量）（排除限制）。
- (iii) Z 对 Y 的影响之间不存在混杂因素，即 Z 和 Y 的所有共同原因（如有）都在研究中得到控制或不存在。
输入：
- 因果图描述，表示哪些变量之间存在因果关系。
- 一个问题，指明如何确定与一对有序变量（X，Y）相关的工具变量。
- 代表可能是工具变量的多选选项。
输出：
- 答案格式为 "选项 N"，根据所提供的选项，N 为 一、二 或 三。

给定如下因果图：%s。
问题：对于上述因果图中的有序变量对 (%s, %s)，工具变量是哪个？
选项一：%s
选项二：%s
选项三：%s
答案（选项一或选项二或选项三？）：""",
    'one-shot-IcL':
    """Objective:
Your task is to identify the set of variables that serve as instrumental variables for a given ordered pair of treatment and outcome variables (X, Y) in a specified causal graph.
Background Information:
An instrumental variable Z must meet the following criteria:
- (i) Z has a causal effect on X (treatment variable).
- (ii) Z affects Y (outcome variable) only through its effect on X, and not directly (exclusion restriction).
- (iii) There is no confounding between the effect of Z on Y, meaning that all common causes of Z and Y (if any) are controlled for in the study or are non-existent.
Input:
- Description of the causal graph, denoting which variables have causal relationships with each other.
- A question that specifies the identification of instrumental variables with respect to an ordered pair of variables (X, Y).
- Multiple-choice options representing sets of variables that could be instrumental.
Output:
- Answer in the format "Option N," where N is either 1, 2, or 3 based on the provided options.

Example:
You will be presented with a causal graph in the following form: A causes D, A causes E, B causes E, C causes D, and D causes E.
Question: Which set of variables is the instrument variables relative to an ordered pair of variables (D, E) in the above causal graph?
Option 1: A
Option 2: C
Option 3: E
Answer (Option 1 or Option 2 or Option 3 ?): Option 2

New Input:
You will be presented with a causal graph in the following form: %s.
Question: Which set of variables is the instrument variables relative to an ordered pair of variables (%s, %s) in the above causal graph?
Option 1: %s
Option 2: %s
Option 3: %s
Answer (Option 1 or Option 2 or Option 3 ?):""",
    'one-shot-IcL-CN':
    """目标
您的任务是在指定的因果关系图中，找出一组变量，作为给定的一对有序处理变量和结果变量（X，Y）的工具变量。
背景信息：
工具变量 Z 必须符合以下标准：
- (i) Z 对 X（治疗变量）有因果效应。
- (ii) Z 仅通过对 X 的影响而非直接影响 Y（结果变量）（排除限制）。
- (iii) Z 对 Y 的影响之间不存在混杂因素，即 Z 和 Y 的所有共同原因（如有）都在研究中得到控制或不存在。
输入：
- 因果图描述，表示哪些变量之间存在因果关系。
- 一个问题，指明如何确定与一对有序变量（X，Y）相关的工具变量。
- 代表可能是工具变量的多选选项。
输出：
- 答案格式为 "选项 N"，根据所提供的选项，N 为 一、二 或 三。

例子：
给定如下因果图：A导致D, A导致E, B导致E, C导致D, 以及D导致E。
问题：对于上述因果图中的有序变量对 (D, E)，工具变量是哪个？
选项一：A
选项二：C
选项三：E
答案（选项一或选项二或选项三？）：选项二

新输入：
给定如下因果图：%s。
问题：对于上述因果图中的有序变量对 (%s, %s)，工具变量是哪个？
选项一：%s
选项二：%s
选项三：%s
答案（选项一或选项二或选项三？）：""",
    'three-shot-IcL':
    """Objective:
Your task is to identify the set of variables that serve as instrumental variables for a given ordered pair of treatment and outcome variables (X, Y) in a specified causal graph.
Background Information:
An instrumental variable Z must meet the following criteria:
- (i) Z has a causal effect on X (treatment variable).
- (ii) Z affects Y (outcome variable) only through its effect on X, and not directly (exclusion restriction).
- (iii) There is no confounding between the effect of Z on Y, meaning that all common causes of Z and Y (if any) are controlled for in the study or are non-existent.
Input:
- Description of the causal graph, denoting which variables have causal relationships with each other.
- A question that specifies the identification of instrumental variables with respect to an ordered pair of variables (X, Y).
- Multiple-choice options representing sets of variables that could be instrumental.
Output:
- Answer in the format "Option N," where N is either 1, 2, or 3 based on the provided options.

Example:
You will be presented with a causal graph in the following form: A causes D, A causes E, B causes E, C causes D, and D causes E.
Question: Which set of variables is the instrument variables relative to an ordered pair of variables (D, E) in the above causal graph?
Option 1: A
Option 2: C
Option 3: E
Answer (Option 1 or Option 2 or Option 3 ?): Option 2

You will be presented with a causal graph in the following form: A causes B, A causes E, B causes E, B causes D, C causes E, and C causes D.
Question: Which set of variables is the instrument variables relative to an ordered pair of variables (B, D) in the above causal graph?
Option 1: A
Option 2: B
Option 3: D
Answer (Option 1 or Option 2 or Option 3 ?): Option 1

You will be presented with a causal graph in the following form: A causes D, A causes B, C causes E, and D causes E.
Question: Which set of variables is the instrument variables relative to an ordered pair of variables (D, E) in the above causal graph?
Option 1: E
Option 2: C
Option 3: A
Answer (Option 1 or Option 2 or Option 3 ?): Option 3

New Input:
You will be presented with a causal graph in the following form: %s.
Question: Which set of variables is the instrument variables relative to an ordered pair of variables (%s, %s) in the above causal graph?
Option 1: %s
Option 2: %s
Option 3: %s
Answer (Option 1 or Option 2 or Option 3 ?):""",
    'three-shot-IcL-CN':
    """目标
您的任务是在指定的因果关系图中，找出一组变量，作为给定的一对有序处理变量和结果变量（X，Y）的工具变量。
背景信息：
工具变量 Z 必须符合以下标准：
- (i) Z 对 X（治疗变量）有因果效应。
- (ii) Z 仅通过对 X 的影响而非直接影响 Y（结果变量）（排除限制）。
- (iii) Z 对 Y 的影响之间不存在混杂因素，即 Z 和 Y 的所有共同原因（如有）都在研究中得到控制或不存在。
输入：
- 因果图描述，表示哪些变量之间存在因果关系。
- 一个问题，指明如何确定与一对有序变量（X，Y）相关的工具变量。
- 代表可能是工具变量的多选选项。
输出：
- 答案格式为 "选项 N"，根据所提供的选项，N 为 一、二 或 三。

例子：
给定如下因果图：A导致D, A导致E, B导致E, C导致D, 以及D导致E。
问题：对于上述因果图中的有序变量对 (D, E)，工具变量是哪个？
选项一：A
选项二：C
选项三：E
答案（选项一或选项二或选项三？）：选项二

给定如下因果图：A导致B, A导致E, B导致E, B导致D, C导致E, 以及C导致D。
问题：对于上述因果图中的有序变量对 (B, D)，工具变量是哪个？
选项一：A
选项二：B
选项三：D
答案（选项一或选项二或选项三？）：选项一

给定如下因果图：A导致D, A导致B, C导致E, 以及D导致E。
问题：对于上述因果图中的有序变量对 (D, E)，工具变量是哪个？
选项一：E
选项二：C
选项三：A
答案（选项一或选项二或选项三？）：选项三

新输入：
给定如下因果图：%s。
问题：对于上述因果图中的有序变量对 (%s, %s)，工具变量是哪个？
选项一：%s
选项二：%s
选项三：%s
答案（选项一或选项二或选项三？）：""",
    'zero-shot-CoT':
    """You will be presented with a causal graph in the following form: %s.
Question: Which set of variables is the instrument variables relative to an ordered pair of variables (%s, %s) in the above causal graph? Let's think step by step.
Option 1: %s
Option 2: %s
Option 3: %s
Answer (Option 1 or Option 2 or Option 3 ?):""",
    'zero-shot-CoT-CN':
    """给定如下因果图：%s。
问题：对于上述因果图中的有序变量对 (%s, %s)，工具变量是哪个？请逐步思考。
选项一：%s
选项二：%s
选项三：%s
答案（选项一或选项二或选项三？）：""",
    'manual-CoT':
    """Here are three examples of identifying instrumental variables using chain of thought, and a question to answer.

You will be presented with a causal graph in the following form: A causes D, A causes C, B causes E, C causes D, and D causes E.
Question: Which set of variables is the instrument variables relative to an ordered pair of variables (D, E) in the above causal graph?
Option 1: A, C
Option 2: B
Option 3: D
Answer (Option 1 or Option 2 or Option 3 ?): A causes D, and C causes D, meaning that A, C, and D are d-connected. Additionally, A and C are not directly related to E, indicating that A, C, and E are d-separated. Therefore, the answer is Option 1.

You will be presented with a causal graph in the following form: A causes B, A causes E, A causes C, B causes C, B causes D, B causes E, and D causes E.
Question: Which set of variables is the instrument variables relative to an ordered pair of variables (B, D) in the above causal graph?
Option 1: B
Option 2: A
Option 3: D
Answer (Option 1 or Option 2 or Option 3 ?): A causes B, meaning that A and B are d-connected. Also, A is not directly related to D, thus A and D are d-separated. Therefore, the answer is Option 2.

You will be presented with a causal graph in the following form: A causes C, A causes D, B causes C, B causes D, and C causes E.
Question: Which set of variables is the instrument variables relative to an ordered pair of variables (C, E) in the above causal graph?
Option 1: D
Option 2: C
Option 3: B, A
Answer (Option 1 or Option 2 or Option 3 ?): A causes C, and B causes C, meaning that B, A, and C are d-connected. Additionally, both B and A are not directly related to E, indicating that B, A, and E are d-separated. Therefore, the answer is Option 3.

You will be presented with a causal graph in the following form: %s.
Question: Which set of variables is the instrument variables relative to an ordered pair of variables (%s, %s) in the above causal graph?
Option 1: %s
Option 2: %s
Option 3: %s
Answer (Option 1 or Option 2 or Option 3 ?):""",
    'manual-CoT-CN':
    """如下为两个使用思维链进行推理的识别工具变量的示例，和一个需要回答的问题。

给定如下因果图：A导致D, A导致C, B导致E, C导致D, 以及D导致E。
问题：对于上述因果图中的有序变量对 (D, E)，工具变量是哪个？
选项一：A, C
选项二：B
选项三：D
答案（选项一或选项二或选项三？）：A导致D，C导致D，即A ,C和D是d-连通的，且A和C不与E直接相关，即A, C与E是d-分离的。因此答案为选项一。

给定如下因果图：A导致B, A导致E, A导致C, B导致C, B导致D, B导致E, 以及D导致E。
问题：对于上述因果图中的有序变量对 (B, D)，工具变量是哪个？
选项一：B
选项二：A
选项三：D
答案（选项一或选项二或选项三？）：A导致B，即A和B是d-连通的，且A与D不直接相关，即A和D是d-分离的。因此答案为选项二。

给定如下因果图：%s。
问题：对于上述因果图中的有序变量对 (%s, %s)，工具变量是哪个？
选项一：%s
选项二：%s
选项三：%s
答案（选项一或选项二或选项三？）：""",
    'explicit-function':
    """You are a helpful assistant for adjustment set analysis (instrument variables).
You will be presented with a causal graph in the following form: %s.
Question: Which set of variables is the instrument variables relative to an ordered pair of variables (%s, %s) in the above causal graph?
Option 1: %s
Option 2: %s
Option 3: %s
Answer (Option 1 or Option 2 or Option 3 ?):""",
    'explicit-function-CN':
    """你是一个用于调整集分析(工具变量)的得力助手。
给定如下因果图：%s。
问题：对于上述因果图中的有序变量对 (%s, %s)，工具变量是哪个？
选项一：%s
选项二：%s
选项三：%s
答案（选项一或选项二或选项三？）""",
}


def get_prompt(task_name, prompt_style, item, prompt_style_str=''):
    base = base_prompt_dict[prompt_style]

    prompt = prompt_style_str + base % (item['edges'], item['treatment'],
                                        item['outcome'], item['option1'],
                                        item['option2'], item['option3'])
    return prompt
