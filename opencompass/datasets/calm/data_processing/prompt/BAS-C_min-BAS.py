# flake8: noqa: E501
base_prompt_dict = {
    'basic':
    """You will be presented with a causal graph in the following form: %s.
Question: Which is the minimal set of variables that satisfies the back-door criterion relative to an ordered pair of variables (%s, %s) in the above causal graph?
Option 1: %s
Option 2: %s
Option 3: %s
Answer (Option 1 or Option 2 or Option 3 ?):""",
    'basic-CN':
    """给定如下因果图：%s。
问题：对于上述因果图中的有序变量对 (%s, %s)，满足后门准则的最小变量集是哪个？
选项一：%s
选项二：%s
选项三：%s
答案（选项一或选项二或选项三？）：""",
    'adversarial-ignore':
    """You will be presented with a causal graph in the following form: %s.
Question: Which is the minimal set of variables that satisfies the back-door criterion relative to an ordered pair of variables (%s, %s) in the above causal graph?
Option 1: %s
Option 2: %s
Option 3: %s
Answer (Option 1 or Option 2 or Option 3 ?):""",
    'adversarial-ignore-CN':
    """给定如下因果图：%s。
问题：对于上述因果图中的有序变量对 (%s, %s)，满足后门准则的最小变量集是哪个？
选项一：%s
选项二：%s
选项三：%s
答案（选项一或选项二或选项三？）：""",
    'adversarial-doubt':
    """You will be presented with a causal graph in the following form: %s.
Question: Which is the minimal set of variables that satisfies the back-door criterion relative to an ordered pair of variables (%s, %s) in the above causal graph?
Option 1: %s
Option 2: %s
Option 3: %s
Answer (Option 1 or Option 2 or Option 3 ?):""",
    'adversarial-doubt-CN':
    """给定如下因果图：%s。
问题：对于上述因果图中的有序变量对 (%s, %s)，满足后门准则的最小变量集是哪个？
选项一：%s
选项二：%s
选项三：%s
答案（选项一或选项二或选项三？）：""",
    'zero-shot-IcL':
    """Objective:
Given a causal graph with specified relationships between variables, your task is to identify the set of variables that satisfy the back-door criterion relative to an ordered pair of variables (X, Y) in that graph.
Background Information:
The back-door criterion is defined as follows:
1. The variable set Z must not contain any descendants of X.
2. The variable set Z must block every path from X to Y that has an arrow pointing to X.
Input:
- A textual description of the causal graph, detailing which variables cause which other variables.
- A question asking for the minimal set of variables that satisfy the back-door criterion for a specified ordered pair of variables (X, Y).
- Multiple choice options representing possible sets of variables that could satisfy the criterion.
Output:
- Answer to the question in the format "Option N", where N is 1, 2, or 3 based on the given options.

You will be presented with a causal graph in the following form: %s.
Question: Which is the minimal set of variables that satisfies the back-door criterion relative to an ordered pair of variables (%s, %s) in the above causal graph?
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
- 一个问题，要求找出满足指定有序变量对（X，Y）的后门标准的最小变量集。
- 代表可能满足该标准的变量集的多选选项。
输出：
- 问题答案，格式为 "选项 N"，根据给定选项，N 为 一、二或 三。

给定如下因果图：%s。
问题：对于上述因果图中的有序变量对 (%s, %s)，满足后门准则的最小变量集是哪个？
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
- A question asking for the minimal set of variables that satisfy the back-door criterion for a specified ordered pair of variables (X, Y).
- Multiple choice options representing possible sets of variables that could satisfy the criterion.
Output:
- Answer to the question in the format "Option N", where N is 1, 2, or 3 based on the given options.

Example:
You will be presented with a causal graph in the following form: A causes D, A causes E, B causes E, C causes D, and D causes E.
Question: Which is the minimal set of variables that satisfies the back-door criterion relative to an ordered pair of variables (A, C) in the above causal graph?
Option 1:
Option 2: A
Option 3: E
Answer (Option 1 or Option 2 or Option 3 ?): Option 1

New Input:
You will be presented with a causal graph in the following form: %s.
Question: Which is the minimal set of variables that satisfies the back-door criterion relative to an ordered pair of variables (%s, %s) in the above causal graph?
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
- 一个问题，要求找出满足指定有序变量对（X，Y）的后门标准的最小变量集。
- 代表可能满足该标准的变量集的多选选项。
输出：
- 问题答案，格式为 "选项 N"，根据给定选项，N 为 一、二或 三。

例子：
给定如下因果图：A导致D, A导致E, B导致E, C导致D, 以及D导致E。
问题：对于上述因果图中的有序变量对 (A, C)，满足后门准则的最小变量集是哪个？
选项一：
选项二：A
选项三：E
答案（选项一或选项二或选项三？）：选项一

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
- A question asking for the minimal set of variables that satisfy the back-door criterion for a specified ordered pair of variables (X, Y).
- Multiple choice options representing possible sets of variables that could satisfy the criterion.
Output:
- Answer to the question in the format "Option N", where N is 1, 2, or 3 based on the given options.

Example:
You will be presented with a causal graph in the following form: A causes D, A causes E, B causes E, C causes D, and D causes E.
Question: Which is the minimal set of variables that satisfies the back-door criterion relative to an ordered pair of variables (A, C) in the above causal graph?
Option 1:
Option 2: A
Option 3: E
Answer (Option 1 or Option 2 or Option 3 ?): Option 1

You will be presented with a causal graph in the following form: A causes B, A causes C, B causes E, B causes C, C causes D, and C causes E.
Question: Which is the minimal set of variables that satisfies the back-door criterion relative to an ordered pair of variables (C, E) in the above causal graph?
Option 1: E
Option 2: D
Option 3: B
Answer (Option 1 or Option 2 or Option 3 ?): Option 3

You will be presented with a causal graph in the following form: A causes C, A causes D, A causes E, B causes D, B causes E, C causes D, and D causes E.
Question: Which is the minimal set of variables that satisfies the back-door criterion relative to an ordered pair of variables (D, D) in the above causal graph?
Option 1: B
Option 2:
Option 3: C
Answer (Option 1 or Option 2 or Option 3 ?): Option 2

New Input:
You will be presented with a causal graph in the following form: %s.
Question: Which is the minimal set of variables that satisfies the back-door criterion relative to an ordered pair of variables (%s, %s) in the above causal graph?
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
- 一个问题，要求找出满足指定有序变量对（X，Y）的后门标准的最小变量集。
- 代表可能满足该标准的变量集的多选选项。
输出：
- 问题答案，格式为 "选项 N"，根据给定选项，N 为 一、二或 三。

例子：
给定如下因果图：A导致D, A导致E, B导致E, C导致D, 以及D导致E。
问题：对于上述因果图中的有序变量对 (A, C)，满足后门准则的最小变量集是哪个？
选项一：
选项二：A
选项三：E
答案（选项一或选项二或选项三？）：选项一

给定如下因果图：A导致B, A导致C, B导致E, B导致C, C导致D, 以及C导致E。
问题：对于上述因果图中的有序变量对 (C, E)，满足后门准则的最小变量集是哪个？
选项一：E
选项二：D
选项三：B
答案（选项一或选项二或选项三？）：选项三

给定如下因果图：A导致C, A导致D, A导致E, B导致D, B导致E, C导致D, 以及D导致E。
问题：对于上述因果图中的有序变量对 (D, D)，满足后门准则的最小变量集是哪个？
选项一：B
选项二：
选项三：C
答案（选项一或选项二或选项三？）：选项二

新输入：
给定如下因果图：%s。
问题：对于上述因果图中的有序变量对 (%s, %s)，满足后门准则的最小变量集是哪个？
选项一：%s
选项二：%s
选项三：%s
答案（选项一或选项二或选项三？）：""",
    'zero-shot-CoT':
    """You will be presented with a causal graph in the following form: %s.
Question: Which is the minimal set of variables that satisfies the back-door criterion relative to an ordered pair of variables (%s, %s) in the above causal graph? Let's think step by step.
Option 1: %s
Option 2: %s
Option 3: %s
Answer (Option 1 or Option 2 or Option 3 ?):""",
    'zero-shot-CoT-CN':
    """给定如下因果图：%s。
问题：对于上述因果图中的有序变量对 (%s, %s)，满足后门准则的最小变量集是哪个？请逐步思考。
选项一：%s
选项二：%s
选项三：%s
答案（选项一或选项二或选项三？）：""",
    'manual-CoT':
    """Here are eight examples for problems about finding minimal backdoor adjustment set with chain of thought.

You will be presented with a causal graph in the following form: A causes C, A causes D, B causes C, B causes D, and C causes E.
Question: Which is the minimal set of variables that satisfies the back-door criterion relative to an ordered pair of variables (B, E) in the above causal graph?
Option 1: D
Option 2:
Option 3: C
Answer (Option 1 or Option 2 or Option 3 ?): B does not have any ancestors. C and D are descendants of B, thus the minimal adjustment set is empty. The answer is Option 2.

You will be presented with a causal graph in the following form: A causes F, A causes D, B causes D, B causes C, B causes E, B causes F, C causes D, C causes F, D causes E, and D causes F.
Question: Which is the minimal set of variables that satisfies the back-door criterion relative to an ordered pair of variables (C, D) in the above causal graph?
Option 1: D
Option 2: C
Option 3: B
Answer (Option 1 or Option 2 or Option 3 ?): B causes C and B causes D, so B blocks the path C<-B->D. C does not have any other ancestor, thus B blocks every path from C to D with an arrow pointing into C. The answer is Option 3.

You will be presented with a causal graph in the following form: A causes D, A causes F, A causes E, B causes D, B causes F, B causes C, C causes D, C causes E, and E causes F.
Question: Which is the minimal set of variables that satisfies the back-door criterion relative to an ordered pair of variables (C, D) in the above causal graph?
Option 1: B
Option 2: C
Option 3: E
Answer (Option 1 or Option 2 or Option 3 ?): B causes C and B causes D, so B blocks the path C<-B->D. C does not have any other ancestor, thus B blocks every path from C to D with an arrow pointing into C. The answer is Option 1.

You will be presented with a causal graph in the following form: A causes D, B causes C, B causes D, C causes E, and C causes D.
Question: Which is the minimal set of variables that satisfies the back-door criterion relative to an ordered pair of variables (C, D) in the above causal graph?
Option 1: D
Option 2: B
Option 3: A
Answer (Option 1 or Option 2 or Option 3 ?): B causes C and D, then B blocks path C<-B->D. C does not have any other ancestor, thus B blocks every path from C to D with an arrow pointing into C. The answer is Option 2.

You will be presented with a causal graph in the following form: A causes G, A causes F, A causes C, B causes G, B causes E, B causes D, B causes F, C causes E, C causes G, D causes G, E causes G, and F causes G.
Question: Which is the minimal set of variables that satisfies the back-door criterion relative to an ordered pair of variables (D, G) in the above causal graph?
Option 1: A
Option 2: D
Option 3: B
Answer (Option 1 or Option 2 or Option 3 ?): B is the only parent of D, and B blocks all the paths from D to G with an arrow pointing into D, like D<-B->G and D<-B->E->G. The answer is Option 3.

You will be presented with a causal graph in the following form: A causes B, A causes F, B causes D, B causes F, B causes E, C causes E, C causes F, D causes E, and E causes F.
Question: Which is the minimal set of variables that satisfies the back-door criterion relative to an ordered pair of variables (D, E) in the above causal graph?
Option 1: B
Option 2: E
Option 3: C
Answer (Option 1 or Option 2 or Option 3 ?): B is the only parent of D, and B blocks all the paths from D to E with an arrow pointing into D, like D<-B->E. The answer is Option 1.

You will be presented with a causal graph in the following form: A causes D, A causes C, A causes E, A causes B, B causes C, B causes E, B causes D, C causes F, C causes D, and E causes F.
Question: Which is the minimal set of variables that satisfies the back-door criterion relative to an ordered pair of variables (C, F) in the above causal graph?
Option 1: A
Option 2: B
Option 3: E
Answer (Option 1 or Option 2 or Option 3 ?): E is not a descendant of C, and E blocks every path from C to F with an arrow pointing into C, like C<-B->E->F. The answer is Option 3.

You will be presented with a causal graph in the following form: A causes B, A causes F, A causes C, B causes F, C causes E, C causes F, D causes F, D causes E, and E causes F.
Question: Which is the minimal set of variables that satisfies the back-door criterion relative to an ordered pair of variables (E, F) in the above causal graph?
Option 1: A
Option 2: D, C
Option 3: F
Answer (Option 1 or Option 2 or Option 3 ?): D and C are two parents of E. C and D block every path from E to F with an arrow pointing into E, and any smaller variable set fails to achieve it. The answer is Option 2.

You will be presented with a causal graph in the following form: %s.
Question: Which is the minimal set of variables that satisfies the back-door criterion relative to an ordered pair of variables (%s, %s) in the above causal graph?
Option 1: %s
Option 2: %s
Option 3: %s
Answer (Option 1 or Option 2 or Option 3 ?):
""",
    'manual-CoT-CN':
    """如下为两个使用思维链进行推理的判断最小后门变量集合的示例，和一个需要回答的问题。

给定如下因果图：A导致C, A导致D, B导致C, B导致D, 以及C导致E。
问题：对于上述因果图中的有序变量对 (B, E)，满足后门准则的最小变量集是哪个？
选项一：D
选项二：
选项三：C
答案（选项一或选项二或选项三？）：由于D和C是B的后代，B和E之间没有箭头指向B的路径，所以最小后门集合为空集。因此答案选项二。

给定如下因果图：A导致D, A导致B, A导致E, B导致C, B导致E, C导致D, C导致E, 以及D导致E。
问题：对于上述因果图中的有序变量对 (C, D)，满足后门准则的最小变量集是哪个？
选项一：C
选项二：B
选项三：D
答案（选项一或选项二或选项三？）：B阻断了C和D之间所有含有指向C的边的路径，例如C<-B<-A->D，所以最小后门集合是{B}。因此答案为选项二。

给定如下因果图：%s。
问题：对于上述因果图中的有序变量对 (%s, %s)，满足后门准则的最小变量集是哪个？
选项一：%s
选项二：%s
选项三：%s
答案（选项一或选项二或选项三？）：""",
    'explicit-function':
    """You are a helpful assistant for adjustment set analysis (back-door criterion).
You will be presented with a causal graph in the following form: %s.
Question: Which is the minimal set of variables that satisfies the back-door criterion relative to an ordered pair of variables (%s, %s) in the above causal graph?
Option 1: %s
Option 2: %s
Option 3: %s
Answer (Option 1 or Option 2 or Option 3 ?):""",
    'explicit-function-CN':
    """你是一个用于调整集分析(后门准则)的得力助手。
给定如下因果图：%s。
问题：对于上述因果图中的有序变量对 (%s, %s)，满足后门准则的最小变量集是哪个？
选项一：%s
选项二：%s
选项三：%s
答案（选项一或选项二或选项三？）：""",
}


def get_prompt(task_name, prompt_style, item, prompt_style_str=''):
    base = base_prompt_dict[prompt_style]
    prompt = prompt_style_str + base % (item['edges'], item['treatment'],
                                        item['outcome'], item['option1'],
                                        item['option2'], item['option3'])
    return prompt
