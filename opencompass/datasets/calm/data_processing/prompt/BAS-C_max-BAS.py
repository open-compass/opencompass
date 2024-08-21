# flake8: noqa: E501
base_prompt_dict = {
    'basic':
    """You will be presented with a causal graph in the following form: %s.
Question: Which is the maximal set of variables that satisfies the back-door criterion relative to an ordered pair of variables (%s, %s) in the above causal graph?
Option 1: %s
Option 2: %s
Option 3: %s
Answer (Option 1 or Option 2 or Option 3 ?):""",
    'basic-CN':
    """给定如下因果图：%s。
问题：对于上述因果图中的有序变量对 (%s, %s)，满足后门准则的最大变量集是哪个？
选项一：%s
选项二：%s
选项三：%s
答案（选项一或选项二或选项三？）：
""",
    'adversarial-ignore':
    """You will be presented with a causal graph in the following form: %s.
Question: Which is the maximal set of variables that satisfies the back-door criterion relative to an ordered pair of variables (%s, %s) in the above causal graph?
Option 1: %s
Option 2: %s
Option 3: %s
Answer (Option 1 or Option 2 or Option 3 ?):""",
    'adversarial-ignore-CN':
    """给定如下因果图：%s。
问题：对于上述因果图中的有序变量对 (%s, %s)，满足后门准则的最大变量集是哪个？
选项一：%s
选项二：%s
选项三：%s
答案（选项一或选项二或选项三？）：
""",
    'adversarial-doubt':
    """You will be presented with a causal graph in the following form: %s.
Question: Which is the maximal set of variables that satisfies the back-door criterion relative to an ordered pair of variables (%s, %s) in the above causal graph?
Option 1: %s
Option 2: %s
Option 3: %s
Answer (Option 1 or Option 2 or Option 3 ?):""",
    'adversarial-doubt-CN':
    """给定如下因果图：%s。
问题：对于上述因果图中的有序变量对 (%s, %s)，满足后门准则的最大变量集是哪个？
选项一：%s
选项二：%s
选项三：%s
答案（选项一或选项二或选项三？）：
""",
    'zero-shot-IcL':
    """Objective:
Given a causal graph with specified relationships between variables, your task is to identify the set of variables that satisfy the back-door criterion relative to an ordered pair of variables (X, Y) in that graph.
Background Information:
The back-door criterion is defined as follows:
1. The variable set Z must not contain any descendants of X.
2. The variable set Z must block every path from X to Y that has an arrow pointing to X.
Input:
- A textual description of the causal graph, detailing which variables cause which other variables.
- A question asking for the maximal set of variables that satisfy the back-door criterion for a specified ordered pair of variables (X, Y).
- Multiple choice options representing possible sets of variables that could satisfy the criterion.
Output:
- Answer to the question in the format "Option N", where N is 1, 2, or 3 based on the given options.

You will be presented with a causal graph in the following form: %s.
Question: Which is the maximal set of variables that satisfies the back-door criterion relative to an ordered pair of variables (%s, %s) in the above causal graph?
Option 1: %s
Option 2: %s
Option 3: %s
Answer (Option 1 or Option 2 or Option 3 ?):""",
    'zero-shot-IcL-CN':
    """给定一个具有指定变量间关系的因果图，你的任务是找出相对于该图中有序的一对变量（X，Y），满足后门标准的一组变量。
背景信息：
后门准则的定义如下：
1. 变量集 Z 不能包含任何 X 的后代。
2. 变量集 Z 必须阻止每一条从 X 到 Y 的路径，这条路径必须有一个箭头指向 X。
输入
- 因果图的文字描述，详细说明哪些变量会导致哪些其他变量。
- 一个问题，要求找出满足指定有序变量对（X，Y）的后门标准的最大变量集。
- 代表可能满足该标准的变量集的多选选项。
输出：
- 问题答案，格式为 "选项 N"，根据给定选项，N 为 一、二或 三。

给定如下因果图：%s。
问题：对于上述因果图中的有序变量对 (%s, %s)，满足后门准则的最大变量集是哪个？
选项一：%s
选项二：%s
选项三：%s
答案（选项一或选项二或选项三？）：""",
    'one-shot-IcL':
    """Objective:
Given a causal graph with specified relationships between variables, your task is to identify the set of variables that satisfy the back-door criterion relative to an ordered pair of variables (X, Y) in that graph.
Background Information:
The back-door criterion is defined as follows:
1. The variable set Z must not contain any descendants of X.
2. The variable set Z must block every path from X to Y that has an arrow pointing to X.
Input:
- A textual description of the causal graph, detailing which variables cause which other variables.
- A question asking for the maximal set of variables that satisfy the back-door criterion for a specified ordered pair of variables (X, Y).
- Multiple choice options representing possible sets of variables that could satisfy the criterion.
Output:
- Answer to the question in the format "Option N", where N is 1, 2, or 3 based on the given options.

Example:
You will be presented with a causal graph in the following form: A causes D, A causes E, B causes E, C causes D, and D causes E.
Question: Which is the maximal set of variables that satisfies the back-door criterion relative to an ordered pair of variables (A, E) in the above causal graph?
Option 1: E, B
Option 2: B, C
Option 3: E, D
Answer (Option 1 or Option 2 or Option 3 ?): Option 2

New Input:
You will be presented with a causal graph in the following form: %s.
Question: Which is the maximal set of variables that satisfies the back-door criterion relative to an ordered pair of variables (%s, %s) in the above causal graph?
Option 1: %s
Option 2: %s
Option 3: %s
Answer (Option 1 or Option 2 or Option 3 ?):""",
    'one-shot-IcL-CN':
    """给定一个具有指定变量间关系的因果图，你的任务是找出相对于该图中有序的一对变量（X，Y），满足后门标准的一组变量。
背景信息：
后门准则的定义如下：
1. 变量集 Z 不能包含任何 X 的后代。
2. 变量集 Z 必须阻止每一条从 X 到 Y 的路径，这条路径必须有一个箭头指向 X。
输入
- 因果图的文字描述，详细说明哪些变量会导致哪些其他变量。
- 一个问题，要求找出满足指定有序变量对（X，Y）的后门标准的最大变量集。
- 代表可能满足该标准的变量集的多选选项。
输出：
- 问题答案，格式为 "选项 N"，根据给定选项，N 为 一、二或 三。

例子：
给定如下因果图：A导致D, A导致E, B导致E, C导致D, 以及D导致E。
问题：对于上述因果图中的有序变量对 (A, E)，满足后门准则的最大变量集是哪个？
选项一：E, B
选项二：B, C
选项三：E, D
答案（选项一或选项二或选项三？）：选项二

新输入：
给定如下因果图：%s。
问题：对于上述因果图中的有序变量对 (%s, %s)，满足后门准则的最小变量集是哪个？
选项一：%s
选项二：%s
选项三：%s
答案（选项一或选项二或选项三？）：""",
    'three-shot-IcL':
    """Objective:
Given a causal graph with specified relationships between variables, your task is to identify the set of variables that satisfy the back-door criterion relative to an ordered pair of variables (X, Y) in that graph.
Background Information:
The back-door criterion is defined as follows:
1. The variable set Z must not contain any descendants of X.
2. The variable set Z must block every path from X to Y that has an arrow pointing to X.
Input:
- A textual description of the causal graph, detailing which variables cause which other variables.
- A question asking for the maximal set of variables that satisfy the back-door criterion for a specified ordered pair of variables (X, Y).
- Multiple choice options representing possible sets of variables that could satisfy the criterion.
Output:
- Answer to the question in the format "Option N", where N is 1, 2, or 3 based on the given options.

Example:
You will be presented with a causal graph in the following form: A causes D, A causes E, B causes E, C causes D, and D causes E.
Question: Which is the maximal set of variables that satisfies the back-door criterion relative to an ordered pair of variables (A, E) in the above causal graph?
Option 1: E, B
Option 2: B, C
Option 3: E, D
Answer (Option 1 or Option 2 or Option 3 ?): Option 2

You will be presented with a causal graph in the following form: A causes D, A causes C, B causes E, C causes D, and D causes E.
Question: Which is the maximal set of variables that satisfies the back-door criterion relative to an ordered pair of variables (B, E) in the above causal graph?
Option 1: D, C
Option 2: E, A
Option 3: A, C
Answer (Option 1 or Option 2 or Option 3 ?): Option 1

You will be presented with a causal graph in the following form: A causes E, A causes D, B causes D, B causes E, C causes E, and D causes E.
Question: Which is the maximal set of variables that satisfies the back-door criterion relative to an ordered pair of variables (A, E) in the above causal graph?
Option 1: B, D
Option 2: A, C
Option 3: B, C
Answer (Option 1 or Option 2 or Option 3 ?): Option 3

New Input:
You will be presented with a causal graph in the following form: %s.
Question: Which is the maximal set of variables that satisfies the back-door criterion relative to an ordered pair of variables (%s, %s) in the above causal graph?
Option 1: %s
Option 2: %s
Option 3: %s
Answer (Option 1 or Option 2 or Option 3 ?):""",
    'three-shot-IcL-CN':
    """给定一个具有指定变量间关系的因果图，你的任务是找出相对于该图中有序的一对变量（X，Y），满足后门标准的一组变量。
背景信息：
后门准则的定义如下：
1. 变量集 Z 不能包含任何 X 的后代。
2. 变量集 Z 必须阻止每一条从 X 到 Y 的路径，这条路径必须有一个箭头指向 X。
输入
- 因果图的文字描述，详细说明哪些变量会导致哪些其他变量。
- 一个问题，要求找出满足指定有序变量对（X，Y）的后门标准的最大变量集。
- 代表可能满足该标准的变量集的多选选项。
输出：
- 问题答案，格式为 "选项 N"，根据给定选项，N 为 一、二或 三。

例子：
给定如下因果图：A导致D, A导致E, B导致E, C导致D, 以及D导致E。
问题：对于上述因果图中的有序变量对 (A, E)，满足后门准则的最大变量集是哪个？
选项一：E, B
选项二：B, C
选项三：E, D
答案（选项一或选项二或选项三？）：选项二


给定如下因果图：A导致D, A导致C, B导致E, C导致D, 以及D导致E。
问题：对于上述因果图中的有序变量对 (B, E)，满足后门准则的最小变量集是哪个？
选项一：D, C
选项二：E, A
选项三：A, C
答案（选项一或选项二或选项三？）：选项一

给定如下因果图：A导致E, A导致D, B导致D, B导致E, C导致E, 以及D导致E。
问题：对于上述因果图中的有序变量对 (A, E)，满足后门准则的最小变量集是哪个？
选项一：B, D
选项二：A, C
选项三：B, C
答案（选项一或选项二或选项三？）：选项三

新输入：
给定如下因果图：%s。
问题：对于上述因果图中的有序变量对 (%s, %s)，满足后门准则的最小变量集是哪个？
选项一：%s
选项二：%s
选项三：%s
答案（选项一或选项二或选项三？）：""",
    'zero-shot-CoT':
    """You will be presented with a causal graph in the following form: %s.
Question: Which is the maximal set of variables that satisfies the back-door criterion relative to an ordered pair of variables (%s, %s) in the above causal graph? Let's think step by step.
Option 1: %s
Option 2: %s
Option 3: %s
Answer (Option 1 or Option 2 or Option 3 ?):""",
    'zero-shot-CoT-CN':
    """给定如下因果图：%s。
问题：对于上述因果图中的有序变量对 (%s, %s)，满足后门准则的最大变量集是哪个？请逐步思考。
选项一：%s
选项二：%s
选项三：%s
答案（选项一或选项二或选项三？）：""",
    'manual-CoT':
    """Here are three examples of finding the maximal backdoor adjustment set using chain of thought, and a question to answer. Note A is unobserved in the following questions.

You will be presented with a causal graph in the following form: A causes B, A causes D, A causes E, B causes E, B causes D, B causes C, C causes D, C causes E, D causes F, and D causes E.
Question: Which is the maximal set of variables that satisfies the back-door criterion relative to an ordered pair of variables (D, F) in the above causal graph?
Option 1: D, A
Option 2: D, F
Option 3: B, C
Answer (Option 1 or Option 2 or Option 3 ?): Since E is a descendant of D and A is unobserved, and there is no path between D and F containing an arrow pointing into D, thus the maximal backdoor adjust set is {B, C}. Therefore, the answer is Option 3.

You will be presented with a causal graph in the following form: A causes G, A causes C, B causes F, B causes D, B causes E, B causes G, C causes E, D causes F, D causes G, E causes G, and E causes F.
Question: Which is the maximal set of variables that satisfies the back-door criterion relative to an ordered pair of variables (A, E) in the above causal graph?
Option 1: E, F
Option 2: F, A
Option 3: B, D
Answer (Option 1 or Option 2 or Option 3 ?): Since C, G, E and F are all the descendants of A, and there is no path between A and E with an arrow pointing into A, thus the maximal backdoor adjustment set is {B, D}. Therefore, the answer is Option 3.

You will be presented with a causal graph in the following form: A causes H, A causes C, A causes E, A causes G, A causes B, B causes D, C causes D, C causes E, D causes F, D causes G, and F causes H.
Question: Which is the maximal set of variables that satisfies the back-door criterion relative to an ordered pair of variables (B, D) in the above causal graph?
Option 1: C, E
Option 2: E, D
Option 3: C, H
Answer (Option 1 or Option 2 or Option 3 ?): Since F, G and H are all the descendants of B, and C and E block all the paths from B to D with an arrow pointing into B, thus the maximal backdoor adjustment set is {C,E}. Therefore, the answer is Option 1.

You will be presented with a causal graph in the following form: %s.
Question: Which is the maximal set of variables that satisfies the back-door criterion relative to an ordered pair of variables (%s, %s) in the above causal graph?
Option 1: %s
Option 2: %s
Option 3: %s
Answer (Option 1 or Option 2 or Option 3 ?):""",
    'manual-CoT-CN':
    """如下为两个使用思维链进行推理的判断最大后门变量集合的示例，和一个需要回答的问题。注意在以下的问题设定中，A为未观测到的变量。

给定如下因果图：A导致B, A导致D, A导致E, B导致E, B导致D, B导致C, C导致D, C导致E, D导致F, 以及D导致E。
问题：对于上述因果图中的有序变量对 (D, F)，满足后门准则的最大变量集是哪个？
选项一：D, A
选项二：D, F
选项三：B, C
答案（选项一或选项二或选项三？）：由于E是D的后代，A为未观测到的变量，D和F之间没有箭头指向D的路径，所以最大后门调整集是 {B，C}。因此答案为选项三。

给定如下因果图：A导致G, A导致C, B导致F, B导致D, B导致E, B导致G, C导致E, D导致F, D导致G, E导致G, 以及E导致F。
问题：对于上述因果图中的有序变量对 (A, E)，满足后门准则的最大变量集是哪个？
选项一：E, F
选项二：F, A
选项三：B, D
答案（选项一或选项二或选项三？）：由于C、G、E和F都是A的后代，A和E之间没有箭头指向A的路径，所以最大后门调整集是 {B，D}。因此答案为选项三。

给定如下因果图：%s。
问题：对于上述因果图中的有序变量对 (%s, %s)，满足后门准则的最大变量集是哪个？
选项一：%s
选项二：%s
选项三：%s
答案（选项一或选项二或选项三？）：""",
    'explicit-function':
    """You are a helpful assistant for adjustment set analysis (back-door criterion).
You will be presented with a causal graph in the following form: %s.
Question: Which is the maximal set of variables that satisfies the back-door criterion relative to an ordered pair of variables (%s, %s) in the above causal graph?
Option 1: %s
Option 2: %s
Option 3: %s
Answer (Option 1 or Option 2 or Option 3 ?):""",
    'explicit-function-CN':
    """你是一个用于调整集分析(后门准则)的得力助手。
给定如下因果图：%s。
问题：对于上述因果图中的有序变量对 (%s, %s)，满足后门准则的最大变量集是哪个？
选项一：%s
选项二：%s
选项三：%s
答案（选项一或选项二或选项三？）：
""",
}


def get_prompt(task_name, prompt_style, item, prompt_style_str=''):
    base = base_prompt_dict[prompt_style]
    prompt = prompt_style_str + base % (item['edges'], item['treatment'],
                                        item['outcome'], item['option1'],
                                        item['option2'], item['option3'])
    return prompt
