# flake8: noqa: E501
base_prompt_dict = {
    'basic':
    """You will be presented with a causal graph in the following form: %s.
Question: Which set of variables that satisfies the back-door criterion relative to an ordered pair of variables (%s, %s) in the above causal graph?
Option 1: %s
Option 2: %s
Option 3: %s
Answer (Option 1 or Option 2 or Option 3 ?):""",
    'basic-CN':
    """给定如下因果图：%s。
问题：对于上述因果图中的有序变量对 (%s, %s)，满足后门准则的变量集是哪个？
选项一：%s
选项二：%s
选项三：%s
答案（选项一或选项二或选项三？）：
""",
    'adversarial-ignore':
    """You will be presented with a causal graph in the following form: %s.
Question: Which set of variables that satisfies the back-door criterion relative to an ordered pair of variables (%s, %s) in the above causal graph?
Option 1: %s
Option 2: %s
Option 3: %s
Answer (Option 1 or Option 2 or Option 3 ?):""",
    'adversarial-ignore-CN':
    """给定如下因果图：%s。
问题：对于上述因果图中的有序变量对 (%s, %s)，满足后门准则的变量集是哪个？
选项一：%s
选项二：%s
选项三：%s
答案（选项一或选项二或选项三？）：
""",
    'adversarial-doubt':
    """You will be presented with a causal graph in the following form: %s.
Question: Which set of variables that satisfies the back-door criterion relative to an ordered pair of variables (%s, %s) in the above causal graph?
Option 1: %s
Option 2: %s
Option 3: %s
Answer (Option 1 or Option 2 or Option 3 ?):""",
    'adversarial-doubt-CN':
    """给定如下因果图：%s。
问题：对于上述因果图中的有序变量对 (%s, %s)，满足后门准则的变量集是哪个？
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
- A question asking for the set of variables that satisfy the back-door criterion for a specified ordered pair of variables (X, Y).
- Multiple choice options representing possible sets of variables that could satisfy the criterion.
Output:
- Answer to the question in the format "Option N", where N is 1, 2, or 3 based on the given options.

You will be presented with a causal graph in the following form: %s.
Question: Which set of variables that satisfies the back-door criterion relative to an ordered pair of variables (%s, %s) in the above causal graph?
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
- 一个问题，要求找出满足指定有序变量对（X，Y）的后门标准的变量集。
- 代表可能满足该标准的变量集的多选选项。
输出：
- 问题答案，格式为 "选项 N"，根据给定选项，N 为 一、二或 三。

给定如下因果图：%s。
问题：对于上述因果图中的有序变量对 (%s, %s)，满足后门准则的变量集是哪个？
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
- A question asking for the set of variables that satisfy the back-door criterion for a specified ordered pair of variables (X, Y).
- Multiple choice options representing possible sets of variables that could satisfy the criterion.
Output:
- Answer to the question in the format "Option N", where N is 1, 2, or 3 based on the given options.

Example:
You will be presented with a causal graph in the following form: A causes D, A causes E, B causes E, C causes D, and D causes E.
Question: Which set of variables that satisfies the back-door criterion relative to an ordered pair of variables (B, B) in the above causal graph?
Option 1:
Option 2: C
Option 3: E
Answer (Option 1 or Option 2 or Option 3 ?): Option 1

New Input:
You will be presented with a causal graph in the following form: %s.
Question: Which set of variables that satisfies the back-door criterion relative to an ordered pair of variables (%s, %s) in the above causal graph?
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
- 一个问题，要求找出满足指定有序变量对（X，Y）的后门标准的变量集。
- 代表可能满足该标准的变量集的多选选项。
输出：
- 问题答案，格式为 "选项 N"，根据给定选项，N 为 一、二或 三。

例子：
给定如下因果图：A导致D, A导致E, B导致E, C导致D, 以及D导致E。
问题：对于上述因果图中的有序变量对 (B, B)，满足后门准则的变量集是哪个？
选项一：
选项二：C
选项三：E
答案（选项一或选项二或选项三？）：选项一

新输入：
给定如下因果图：%s。
问题：对于上述因果图中的有序变量对 (%s, %s)，满足后门准则的变量集是哪个？
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
- A question asking for the set of variables that satisfy the back-door criterion for a specified ordered pair of variables (X, Y).
- Multiple choice options representing possible sets of variables that could satisfy the criterion.
Output:
- Answer to the question in the format "Option N", where N is 1, 2, or 3 based on the given options.

Example:
You will be presented with a causal graph in the following form: A causes D, A causes E, B causes E, C causes D, and D causes E.
Question: Which set of variables that satisfies the back-door criterion relative to an ordered pair of variables (B, B) in the above causal graph?
Option 1:
Option 2: C
Option 3: E
Answer (Option 1 or Option 2 or Option 3 ?): Option 1

You will be presented with a causal graph in the following form: A causes C, A causes D, A causes E, B causes D, B causes E, C causes D, and D causes E.
Question: Which set of variables that satisfies the back-door criterion relative to an ordered pair of variables (A, D) in the above causal graph?
Option 1: C
Option 2: B
Option 3: E
Answer (Option 1 or Option 2 or Option 3 ?): Option 2

You will be presented with a causal graph in the following form: A causes E, A causes D, B causes D, B causes E, C causes E, and D causes E.
Question: Which set of variables that satisfies the back-door criterion relative to an ordered pair of variables (A, B) in the above causal graph?
Option 1: A
Option 2: D
Option 3:
Answer (Option 1 or Option 2 or Option 3 ?): Option 3

New Input:
You will be presented with a causal graph in the following form: %s.
Question: Which set of variables that satisfies the back-door criterion relative to an ordered pair of variables (%s, %s) in the above causal graph?
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
- 一个问题，要求找出满足指定有序变量对（X，Y）的后门标准的变量集。
- 代表可能满足该标准的变量集的多选选项。
输出：
- 问题答案，格式为 "选项 N"，根据给定选项，N 为 一、二或 三。

例子：
给定如下因果图：A导致D, A导致E, B导致E, C导致D, 以及D导致E。
问题：对于上述因果图中的有序变量对 (B, B)，满足后门准则的变量集是哪个？
选项一：
选项二：C
选项三：E
答案（选项一或选项二或选项三？）：选项一

给定如下因果图：A导致C, A导致D, A导致E, B导致D, B导致E, C导致D, 以及D导致E。
问题：对于上述因果图中的有序变量对 (A, D)，满足后门准则的变量集是哪个？
选项一：C
选项二：B
选项三：E
答案（选项一或选项二或选项三？）：选项二

给定如下因果图：A导致E, A导致D, B导致D, B导致E, C导致E, 以及D导致E。
问题：对于上述因果图中的有序变量对 (A, B)，满足后门准则的变量集是哪个？
选项一：A
选项二：D
选项三：
答案（选项一或选项二或选项三？）：选项三

新输入：
给定如下因果图：%s。
问题：对于上述因果图中的有序变量对 (%s, %s)，满足后门准则的变量集是哪个？
选项一：%s
选项二：%s
选项三：%s
答案（选项一或选项二或选项三？）：""",
    'zero-shot-CoT':
    """You will be presented with a causal graph in the following form: %s.
Question: Which set of variables that satisfies the back-door criterion relative to an ordered pair of variables (%s, %s) in the above causal graph? Let's think step by step.
Option 1: %s
Option 2: %s
Option 3: %s
Answer (Option 1 or Option 2 or Option 3 ?):""",
    'zero-shot-CoT-CN':
    """给定如下因果图：%s。
问题：对于上述因果图中的有序变量对 (%s, %s)，满足后门准则的变量集是哪个？请逐步思考。
选项一：%s
选项二：%s
选项三：%s
答案（选项一或选项二或选项三？）：""",
    'manual-CoT':
    """Here are eight examples for problems about finding mix backdoor adjustment set with chain of thought. Note A is unobserved in the following questions.

You will be presented with a causal graph in the following form: "A causes D, A causes B, C causes E, and D causes E.
Question: Which set of variables that satisfies the back-door criterion relative to an ordered pair of variables (C, B) in the above causal graph?
Option 1: B
Option 2: D
Option 3:
Answer (Option 1 or Option 2 or Option 3 ?): Since there is no path between C and B with an arrow pointing into C, thus no variable satisfies the back-door criterion. The answer is Option 3.

You will be presented with a causal graph in the following form: A causes B, A causes E, B causes C, B causes E, C causes E, and D causes E.
Question: Which set of variables that satisfies the back-door criterion relative to an ordered pair of variables (A, E) in the above causal graph?
Option 1: C
Option 2: A
Option 3: D
Answer (Option 1 or Option 2 or Option 3 ?): Since B and C are both the descendants of A, and no path between A and E contains an arrow pointing into A, thus only D satisfies the back-door criterion. The answer is Option 3.

You will be presented with a causal graph in the following form: A causes F, A causes D, A causes E, A causes B, B causes E, B causes F, C causes F, D causes E, and E causes F.
Question: Which set of variables that satisfies the back-door criterion relative to an ordered pair of variables (A, F) in the above causal graph?
Option 1: D
Option 2: C
Option 3: E
Answer (Option 1 or Option 2 or Option 3 ?): Since B, D and E are all descendants of A, and no path between A and F contains an arrow pointing into A, thus only C satisfies the back-door criterion. The answer is Option 2.

You will be presented with a causal graph in the following form: A causes C, B causes E, C causes D, and C causes E.
Question: Which set of variables that satisfies the back-door criterion relative to an ordered pair of variables (C, A) in the above causal graph?
Option 1:
Option 2: C
Option 3: B
Answer (Option 1 or Option 2 or Option 3 ?): Since D and E are both the descendants of C, and no path between C and A contains an arrow pointing into C, thus no variable satisfies the back-door criterion. The answer is Option 1.

You will be presented with a causal graph in the following form: A", "B", "C", "D", "E", "F"], "edges": "A causes E, A causes D, A causes C, B causes F, C causes D, C causes E, D causes F, D causes E, and E causes F.
Question: Which set of variables that satisfies the back-door criterion relative to an ordered pair of variables (E, F) in the above causal graph?
Option 1: A
Option 2: B
Option 3: D
Answer (Option 1 or Option 2 or Option 3 ?): Since D is not a descendant of A, and D blocks every path between E and F containing an arrow pointing into E, D satisfies the back-door criterion. The answer is Option 3.

You will be presented with a causal graph in the following form: A causes E, A causes B, A causes C, B causes F, C causes D, and C causes E.
Question: Which set of variables that satisfies the back-door criterion relative to an ordered pair of variables (B, F) in the above causal graph?
Option 1: E, D, C
Option 2: C, E, B
Option 3: F, D, C
Answer (Option 1 or Option 2 or Option 3 ?): Since E, C and D are not descendants of B, and no path between B and F contains an arrow pointing into B, then E, C and D satisfy the back-door criterion. The answer is Option 1.

You will be presented with a causal graph in the following form: A causes E, A causes B, A causes C, B causes D, B causes C, C causes D, and D causes E.
Question: Which set of variables that satisfies the back-door criterion relative to an ordered pair of variables (C, D) in the above causal graph?
Option 1: C
Option 2: B
Option 3: E
Answer (Option 1 or Option 2 or Option 3 ?): Since E is a descendant of C, A is unobserved and B blocks every path between C and D containing an arrow pointing into C, B satisfies the back-door criterion. The answer is Option 2.

You will be presented with a causal graph in the following form: A causes C, A causes B, A causes E, B causes D, C causes D, and D causes E.
Question: Which set of variables that satisfies the back-door criterion relative to an ordered pair of variables (B, D) in the above causal graph?
Option 1: D
Option 2: C
Option 3: B
Answer (Option 1 or Option 2 or Option 3 ?): Since E is a descendant of B, A is unobserved and C blocks every path between B and D with an arrow pointing into B, C satisfies the back-door criterion. The answer is Option 2.

You will be presented with a causal graph in the following form: %s.
Question: Which set of variables that satisfies the back-door criterion relative to an ordered pair of variables (%s, %s) in the above causal graph?
Option 1: %s
Option 2: %s
Option 3: %s
Answer (Option 1 or Option 2 or Option 3 ?):""",
    'manual-CoT-CN':
    """如下为两个使用思维链进行推理的判断后门变量集合的示例，和一个需要回答的问题。

给定如下因果图：A导致D, A导致B, C导致E, 以及D导致E。
问题：对于上述因果图中的有序变量对 (C, B)，满足后门准则的变量集是哪个？
选项一：B
选项二：D
选项三：
答案（选项一或选项二或选项三？）：因为C和B之间没有含有指向C的边的路径，所以后门集合为空集。因此答案为选项三。

给定如下因果图：A导致F, A导致D, A导致E, A导致B, B导致E, B导致F, C导致F, D导致E, 以及E导致F。
问题：对于上述因果图中的有序变量对 (A, F)，满足后门准则的变量集是哪个？
选项一：D
选项二：C
选项三：E
答案（选项一或选项二或选项三？）：由于B, D和E都是A的后代，A和F之间没有箭头指向A的路径，因此C满足后门准则。答案为选项二。

给定如下因果图：%s。
问题：对于上述因果图中的有序变量对 (%s, %s)，满足后门准则的变量集是哪个？
选项一：%s
选项二：%s
选项三：%s
答案（选项一或选项二或选项三？）：
""",
    'explicit-function':
    """You are a helpful assistant for adjustment set analysis (back-door criterion).
You will be presented with a causal graph in the following form: %s.
Question: Which set of variables that satisfies the back-door criterion relative to an ordered pair of variables (%s, %s) in the above causal graph?
Option 1: %s
Option 2: %s
Option 3: %s
Answer (Option 1 or Option 2 or Option 3 ?):""",
    'explicit-function-CN':
    """你是一个用于调整集分析(后门准则)的得力助手。
给定如下因果图：%s。
问题：对于上述因果图中的有序变量对 (%s, %s)，满足后门准则的变量集是哪个？
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
