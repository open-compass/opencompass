# flake8: noqa: E501
base_prompt_dict = {
    'basic':
    """You will be presented with a causal graph in the following form: %s.
Question: Which set of variables that satisfies the front-door criterion relative to an ordered pair of variables (%s, %s) in the above causal graph?
Option 1: %s
Option 2: %s
Option 3: %s
Answer (Option 1 or Option 2 or Option 3 ?):""",
    'basic-CN':
    """给定如下因果图：%s。
问题：对于上述因果图中的有序变量对 (%s, %s)，满足前门准则的变量集是哪个？
选项一：%s
选项二：%s
选项三：%s
答案（选项一或选项二或选项三？）：""",
    'adversarial-ignore':
    """You will be presented with a causal graph in the following form: %s.
Question: Which set of variables that satisfies the front-door criterion relative to an ordered pair of variables (%s, %s) in the above causal graph?
Option 1: %s
Option 2: %s
Option 3: %s
Answer (Option 1 or Option 2 or Option 3 ?):""",
    'adversarial-ignore-CN':
    """给定如下因果图：%s。
问题：对于上述因果图中的有序变量对 (%s, %s)，满足前门准则的变量集是哪个？
选项一：%s
选项二：%s
选项三：%s
答案（选项一或选项二或选项三？）：""",
    'adversarial-doubt':
    """You will be presented with a causal graph in the following form: %s.
Question: Which set of variables that satisfies the front-door criterion relative to an ordered pair of variables (%s, %s) in the above causal graph?
Option 1: %s
Option 2: %s
Option 3: %s
Answer (Option 1 or Option 2 or Option 3 ?):""",
    'adversarial-doubt-CN':
    """给定如下因果图：%s。
问题：对于上述因果图中的有序变量对 (%s, %s)，满足前门准则的变量集是哪个？
选项一：%s
选项二：%s
选项三：%s
答案（选项一或选项二或选项三？）：""",
    'zero-shot-IcL':
    """Objective:
Your task is to identify the set of variables that satisfy the front-door criterion relative to an ordered pair of variables (X, Y) in a given causal graph.
Background Information:
- Back-door criterion is defined as follows:
  1. The variable set Z must not contain any descendants of X.
  2. Z blocks every path from X to Y that has an arrow pointing to X.
- Front-door criterion is defined as follows:
  1. The variable set Z blocks all directed paths from X to Y.
  2. There are no back-door paths from X to Z.
  3. All back-door paths from Z to Y are blocked by X.
Input:
- Description of the causal graph detailing the relationships between variables.
- A question asking for the set of variables that satisfy the front-door criterion for a specified ordered pair of variables (X, Y).
- Multiple-choice options for the set of variables that could satisfy the specified criterion.
Output:
- Answer to the question, provided in the format "Option N", where N is either 1, 2, or 3 based on the provided options.

You will be presented with a causal graph in the following form: %s.
Question: Which set of variables that satisfies the front-door criterion relative to an ordered pair of variables (%s, %s) in the above causal graph?
Option 1: %s
Option 2: %s
Option 3: %s
Answer (Option 1 or Option 2 or Option 3 ?):""",
    'zero-shot-IcL-CN':
    """目标
你的任务是在给定的因果图中，找出相对于有序变量对（X，Y）满足前门准则的变量集。
背景信息：
- 后门准则定义如下：
  1. 变量集 Z 不能包含任何 X 的后代。
  2. Z 阻止了从 X 到 Y 的每一条有箭头指向 X 的路径。
- 前门准则的定义如下
  1. 变量集 Z 阻止所有从 X 到 Y 的有向路径。
  2. 没有从 X 到 Z 的后门路径。
  3. 从 Z 到 Y 的所有后门路径都被 X 堵塞。
输入
- 详细描述变量之间关系的因果图。
- 一个问题，询问符合指定有序变量对（X，Y）的前门标准的变量集。
- 可满足指定标准的变量集合的多选选项。
输出：
- 问题答案，格式为 "选项 N"，根据所提供的选项，N 为 一、二 或 三。

给定如下因果图：%s。
问题：对于上述因果图中的有序变量对 (%s, %s)，满足前门准则的变量集是哪个？
选项一：%s
选项二：%s
选项三：%s
答案（选项一或选项二或选项三？）：""",
    'one-shot-IcL':
    """Objective:
Your task is to identify the set of variables that satisfy the front-door criterion relative to an ordered pair of variables (X, Y) in a given causal graph.
Background Information:
- Back-door criterion is defined as follows:
  1. The variable set Z must not contain any descendants of X.
  2. Z blocks every path from X to Y that has an arrow pointing to X.
- Front-door criterion is defined as follows:
  1. The variable set Z blocks all directed paths from X to Y.
  2. There are no back-door paths from X to Z.
  3. All back-door paths from Z to Y are blocked by X.
Input:
- Description of the causal graph detailing the relationships between variables.
- A question asking for the set of variables that satisfy the front-door criterion for a specified ordered pair of variables (X, Y).
- Multiple-choice options for the set of variables that could satisfy the specified criterion.
Output:
- Answer to the question, provided in the format "Option N", where N is either 1, 2, or 3 based on the provided options.

Example:
You will be presented with a causal graph in the following form: A causes D, A causes E, B causes E, C causes D, and D causes E.
Question: Which set of variables that satisfies the front-door criterion relative to an ordered pair of variables (E, A) in the above causal graph?
Option 1: D
Option 2: C
Option 3:
Answer (Option 1 or Option 2 or Option 3 ?): Option 3

You will be presented with a causal graph in the following form: %s.
Question: Which set of variables that satisfies the front-door criterion relative to an ordered pair of variables (%s, %s) in the above causal graph?
Option 1: %s
Option 2: %s
Option 3: %s
Answer (Option 1 or Option 2 or Option 3 ?):""",
    'one-shot-IcL-CN':
    """目标
你的任务是在给定的因果图中，找出相对于有序变量对（X，Y）满足前门准则的变量集。
背景信息：
- 后门准则定义如下：
  1. 变量集 Z 不能包含任何 X 的后代。
  2. Z 阻止了从 X 到 Y 的每一条有箭头指向 X 的路径。
- 前门准则的定义如下
  1. 变量集 Z 阻止所有从 X 到 Y 的有向路径。
  2. 没有从 X 到 Z 的后门路径。
  3. 从 Z 到 Y 的所有后门路径都被 X 堵塞。
输入
- 详细描述变量之间关系的因果图。
- 一个问题，询问符合指定有序变量对（X，Y）的前门标准的变量集。
- 可满足指定标准的变量集合的多选选项。
输出：
- 问题答案，格式为 "选项 N"，根据所提供的选项，N 为 一、二 或 三。

给定如下因果图：A导致D, A导致E, B导致E, C导致D, 以及D导致E。
问题：对于上述因果图中的有序变量对 (E, A)，满足前门准则的变量集是哪个？
选项一：D
选项二：C
选项三：
答案（选项一或选项二或选项三？）：选项三

给定如下因果图：%s。
问题：对于上述因果图中的有序变量对 (%s, %s)，满足前门准则的变量集是哪个？
选项一：%s
选项二：%s
选项三：%s
答案（选项一或选项二或选项三？）：""",
    'three-shot-IcL':
    """Objective:
Your task is to identify the set of variables that satisfy the front-door criterion relative to an ordered pair of variables (X, Y) in a given causal graph.
Background Information:
- Back-door criterion is defined as follows:
  1. The variable set Z must not contain any descendants of X.
  2. Z blocks every path from X to Y that has an arrow pointing to X.
- Front-door criterion is defined as follows:
  1. The variable set Z blocks all directed paths from X to Y.
  2. There are no back-door paths from X to Z.
  3. All back-door paths from Z to Y are blocked by X.
Input:
- Description of the causal graph detailing the relationships between variables.
- A question asking for the set of variables that satisfy the front-door criterion for a specified ordered pair of variables (X, Y).
- Multiple-choice options for the set of variables that could satisfy the specified criterion.
Output:
- Answer to the question, provided in the format "Option N", where N is either 1, 2, or 3 based on the provided options.

Example:
You will be presented with a causal graph in the following form: A causes D, A causes E, B causes E, C causes D, and D causes E.
Question: Which set of variables that satisfies the front-door criterion relative to an ordered pair of variables (E, A) in the above causal graph?
Option 1: D
Option 2: C
Option 3:
Answer (Option 1 or Option 2 or Option 3 ?): Option 3

You will be presented with a causal graph in the following form: A causes B, A causes E, B causes E, B causes D, C causes E, and C causes D.
Question: Which set of variables that satisfies the front-door criterion relative to an ordered pair of variables (A, D) in the above causal graph?
Option 1: D
Option 2: B
Option 3: A
Answer (Option 1 or Option 2 or Option 3 ?): Option 2

You will be presented with a causal graph in the following form: A causes D, A causes C, B causes E, C causes D, and D causes E.
Question: Which set of variables that satisfies the front-door criterion relative to an ordered pair of variables (A, E) in the above causal graph?
Option 1: D
Option 2: C
Option 3: B
Answer (Option 1 or Option 2 or Option 3 ?): Option 1

You will be presented with a causal graph in the following form: %s.
Question: Which set of variables that satisfies the front-door criterion relative to an ordered pair of variables (%s, %s) in the above causal graph?
Option 1: %s
Option 2: %s
Option 3: %s
Answer (Option 1 or Option 2 or Option 3 ?):""",
    'three-shot-IcL-CN':
    """目标
你的任务是在给定的因果图中，找出相对于有序变量对（X，Y）满足前门准则的变量集。
背景信息：
- 后门准则定义如下：
  1. 变量集 Z 不能包含任何 X 的后代。
  2. Z 阻止了从 X 到 Y 的每一条有箭头指向 X 的路径。
- 前门准则的定义如下
  1. 变量集 Z 阻止所有从 X 到 Y 的有向路径。
  2. 没有从 X 到 Z 的后门路径。
  3. 从 Z 到 Y 的所有后门路径都被 X 堵塞。
输入
- 详细描述变量之间关系的因果图。
- 一个问题，询问符合指定有序变量对（X，Y）的前门标准的变量集。
- 可满足指定标准的变量集合的多选选项。
输出：
- 问题答案，格式为 "选项 N"，根据所提供的选项，N 为 一、二 或 三。

给定如下因果图：A导致D, A导致E, B导致E, C导致D, 以及D导致E。
问题：对于上述因果图中的有序变量对 (E, A)，满足前门准则的变量集是哪个？
选项一：D
选项二：C
选项三：
答案（选项一或选项二或选项三？）：选项三

给定如下因果图：A导致B, A导致E, B导致E, B导致D, C导致E, 以及C导致D。
问题：对于上述因果图中的有序变量对 (A, D)，满足前门准则的变量集是哪个？
选项一：D
选项二：B
选项三：A
答案（选项一或选项二或选项三？）：选项二

给定如下因果图：A导致D, A导致C, B导致E, C导致D, 以及D导致E。
问题：对于上述因果图中的有序变量对 (A, E)，满足前门准则的变量集是哪个？
选项一：D
选项二：C
选项三：B
答案（选项一或选项二或选项三？）：选项一

给定如下因果图：%s。
问题：对于上述因果图中的有序变量对 (%s, %s)，满足前门准则的变量集是哪个？
选项一：%s
选项二：%s
选项三：%s
答案（选项一或选项二或选项三？）：""",
    'zero-shot-CoT':
    """You will be presented with a causal graph in the following form: %s.
Question: Which set of variables that satisfies the front-door criterion relative to an ordered pair of variables (%s, %s) in the above causal graph? Let's think step by step.
Option 1: %s
Option 2: %s
Option 3: %s
Answer (Option 1 or Option 2 or Option 3 ?):""",
    'zero-shot-CoT-CN':
    """给定如下因果图：%s。
问题：对于上述因果图中的有序变量对 (%s, %s)，满足前门准则的变量集是哪个？请逐步思考。
选项一：%s
选项二：%s
选项三：%s
答案（选项一或选项二或选项三？）：""",
    'manual-CoT':
    """Here are eight examples for problems about finding front-door adjustment set with chain of thought. Note A is unobserved in the following questions.

You will be presented with a causal graph in the following form: A causes D, A causes C, B causes E, C causes D, and D causes E.
Question: Which set of variables that satisfies the front-door criterion relative to an ordered pair of variables (A, E) in the above causal graph?
Option 1: D
Option 2: C
Option 3: B
Answer (Option 1 or Option 2 or Option 3 ?): D intercepts the paths from A to E (A->D->E and A->C->D->E). There is no backdoor path from A to D or from D to E. Thus, D satisfies all three conditions of the front-door criterion for the relationship between A (treatment) and E (outcome). The answer is Option 1.

You will be presented with a causal graph in the following form: A causes B, A causes E, B causes C, B causes E, C causes E, and D causes E.
Question: Which set of variables that satisfies the front-door criterion relative to an ordered pair of variables (A, C) in the above causal graph?
Option 1: D
Option 2: B
Option 3: C
Answer (Option 1 or Option 2 or Option 3 ?): B intercepts the path from A to C (A→B→C). There is no back-door path from A to B or from B to C. Thus, B satisfies all three conditions of the front-door criterion for the relationship between A (treatment) and C (outcome). The answer is Option 2.

You will be presented with a causal graph in the following form: A causes B, B causes C, C causes F, D causes F, and D causes E.
Question: Which set of variables that satisfies the front-door criterion relative to an ordered pair of variables (A, C) in the above causal graph?
Option 1: F
Option 2: E
Option 3: B
Answer (Option 1 or Option 2 or Option 3 ?): B intercepts the path from A to C (A->B->C). There is no backdoor path from A to B or from B to C. Thus, B satisfies all three conditions of the front-door criterion for the relationship between A (treatment) and C (outcome). The answer is Option3.

You will be presented with a causal graph in the following form: A causes D, A causes C, B causes D, B causes E, and C causes D.
Question: Which set of variables that satisfies the front-door criterion relative to an ordered pair of variables (C, E) in the above causal graph?
Option 1: A
Option 2:
Option 3: C
Answer (Option 1 or Option 2 or Option 3 ?): A is unobserved. There is a backdoor path from C to D (C<-A->D) and a backdoor path from C to B (C<-A->D<-B). And path C->D<-B->E is a backdoor path from C to E. Thus, A, B, C and E do not satisfy front-door criterion. The answer is Option 2.

You will be presented with a causal graph in the following form: A causes D, B causes C, C causes E, and D causes E.
Question: Which set of variables that satisfies the front-door criterion relative to an ordered pair of variables (A, E) in the above causal graph?
Option 1: D
Option 2: A
Option 3: B
Answer (Option 1 or Option 2 or Option 3 ?): D intercepts the path from A to E (A->D->E). There is no back-door path from A to D or from D to E. Thus, D satisfies all three conditions of the front-door criterion for the relationship between A (treatment) and E (outcome). The answer is Option 1.

You will be presented with a causal graph in the following form: A causes D, B causes C, B causes D, C causes E, and C causes D.
Question: Which set of variables that satisfies the front-door criterion relative to an ordered pair of variables (B, E) in the above causal graph?
Option 1: B
Option 2: A
Option 3: C
Answer (Option 1 or Option 2 or Option 3 ?): C intercepts the path from B to E (B->C->E). There is no back-door path from B to C or from C to E. Thus, C satisfies all three conditions of the front-door criterion for the relationship between B (treatment) and E (outcome). The answer is Option 3.

You will be presented with a causal graph in the following form: A causes E, A causes B, B causes C, and B causes D.
Question: Which set of variables that satisfies the front-door criterion relative to an ordered pair of variables (A, C) in the above causal graph?
Option 1: C
Option 2: B
Option 3: A
Answer (Option 1 or Option 2 or Option 3 ?): B intercepts the path from A to C (A->B->C). There is no back-door path from A to B or from B to C. Thus, B satisfies all three conditions of the front-door criterion for the relationship between A (treatment) and C (outcome). The answer is Option 2.

You will be presented with a causal graph in the following form:
A causes B, B causes C, B causes E, B causes D, and D causes E.
Question: Which set of variables that satisfies the front-door criterion relative to an ordered pair of variables (A, C) in the above causal graph?
Option 1: B
Option 2: E
Option 3: C
Answer (Option 1 or Option 2 or Option 3 ?): B intercepts the path from A and C (A->B->C). There is no back-door path from A to B or from B to C. Thus, B satisfies all three conditions of the front-door criterion for the relationship between A (treatment) and C (outcome). The answer is Option 1.

You will be presented with a causal graph in the following form: %s.
Question: Which set of variables that satisfies the front-door criterion relative to an ordered pair of variables (%s, %s) in the above causal graph?
Option 1: %s
Option 2: %s
Option 3: %s
Answer (Option 1 or Option 2 or Option 3 ?):""",
    'manual-CoT-CN':
    """如下为两个使用思维链进行推理的判断前门变量集合的示例，和一个需要回答的问题。

给定如下因果图：A导致D, A导致C, B导致E, C导致D, 以及D导致E。
问题：对于上述因果图中的有序变量对 (A, E)，满足前门准则的变量集是哪个？
选项一：D
选项二：C
选项三：B
答案（选项一或选项二或选项三？）：D截断了从A到E的路径 (A->D->E以及A->C->D->E)。从A到D和从D到E都没有后门路径。所以D满足从A到E的前门准则的三个条件。因此答案为选项一。

给定如下因果图：A导致B, A导致E, B导致C, B导致E, C导致E, 以及D导致E。
问题：对于上述因果图中的有序变量对 (A, C)，满足前门准则的变量集是哪个？
选项一：D
选项二：B
选项三：C
答案（选项一或选项二或选项三？）：B截断了从A到C的路径(A→B→C)。从A到B和从B到C都没有后门路径。所以B满足从A到C的前门准则的三个条件。因此答案为选项二。

给定如下因果图：%s。
问题：对于上述因果图中的有序变量对 (%s, %s)，满足前门准则的变量集是哪个？
选项一：%s
选项二：%s
选项三：%s
答案（选项一或选项二或选项三？）：""",
    'explicit-function':
    """You are a helpful assistant for adjustment set analysis (front-door criterion).
You will be presented with a causal graph in the following form: %s.
Question: Which set of variables that satisfies the front-door criterion relative to an ordered pair of variables (%s, %s) in the above causal graph?
Option 1: %s
Option 2: %s
Option 3: %s
Answer (Option 1 or Option 2 or Option 3 ?):""",
    'explicit-function-CN':
    """你是一个用于调整集分析(前门准则)的得力助手。
给定如下因果图：%s。
问题：对于上述因果图中的有序变量对 (%s, %s)，满足前门准则的变量集是哪个？
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
