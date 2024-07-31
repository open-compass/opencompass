# flake8: noqa: E501
base_prompt_dict = {
    'basic':
    """You will be presented with a causal graph in the following form: %s.
Question: Does %s serve as the parent node of %s?
Answer (Yes or No ?):""",
    'basic-CN':
    """给定如下因果图：%s。
问题：%s是%s的父节点吗？
答案（是或否？）：""",
    'adversarial-ignore':
    """You will be presented with a causal graph in the following form: %s.
Question: Does %s serve as the parent node of %s?
Answer (Yes or No ?):""",
    'adversarial-ignore-CN':
    """给定如下因果图：%s。
问题：%s是%s的父节点吗？
答案（是或否？）：""",
    'adversarial-doubt':
    """You will be presented with a causal graph in the following form: %s.
Question: Does %s serve as the parent node of %s?
Answer (Yes or No ?):""",
    'adversarial-doubt-CN':
    """给定如下因果图：%s。
问题：%s是%s的父节点吗？
答案（是或否？）：""",
    'zero-shot-IcL':
    """Determine whether or not a variable can serve as the parent of another variable in a given causal graph.
You will be presented with a causal graph in the following form: %s.
Question: Does %s serve as the parent node of %s?
Answer (Yes or No ?):""",
    'zero-shot-IcL-CN':
    """确定在给定的因果图中，一个变量是否可以作为另一个变量的父变量。
给定如下因果图：%s。
问题：%s是%s的父节点吗？
答案（是或否？）：""",
    'one-shot-IcL':
    """Determine whether or not a variable can serve as the parent of another variable in a given causal graph.
You will be presented with a causal graph in the following form: A causes D, A causes E, B causes E, C causes D, and D causes E.
Question: Does E serve as the parent node of E?
Answer (Yes or No ?): No

You will be presented with a causal graph in the following form: %s.
Question: Does %s serve as the parent node of %s?
Answer (Yes or No ?):""",
    'one-shot-IcL-CN':
    """确定在给定的因果图中，一个变量是否可以作为另一个变量的父变量。
给定如下因果图：A导致D, A导致E, B导致E, C导致D, 以及D导致E。
问题：E是E的父节点吗？
答案（是或否？）：否

给定如下因果图：%s。
问题：%s是%s的父节点吗？
答案（是或否？）：""",
    'three-shot-IcL':
    """Determine whether or not a variable can serve as the parent of another variable in a given causal graph.
You will be presented with a causal graph in the following form: A causes D, A causes E, B causes E, C causes D, and D causes E.
Question: Does E serve as the parent node of E?
Answer (Yes or No ?): No

You will be presented with a causal graph in the following form: A causes C, A causes D, A causes E, B causes D, B causes E, C causes D, and D causes E.
Question: Does B serve as the parent node of E?
Answer (Yes or No ?): Yes

You will be presented with a causal graph in the following form: A causes B, A causes D, A causes E, B causes C, B causes E, C causes D, and D causes E.
Question: Does E serve as the parent node of E?
Answer (Yes or No ?): No

You will be presented with a causal graph in the following form: %s.
Question: Does %s serve as the parent node of %s?
Answer (Yes or No ?):""",
    'three-shot-IcL-CN':
    """确定在给定的因果图中，一个变量是否可以作为另一个变量的父变量。
给定如下因果图：A导致D, A导致E, B导致E, C导致D, 以及D导致E。
问题：E是E的父节点吗？
答案（是或否？）：否

给定如下因果图：A导致C, A导致D, A导致E, B导致D, B导致E, C导致D, 以及D导致E。
问题：B是E的父节点吗？
答案（是或否？）：是

给定如下因果图：A导致B, A导致D, A导致E, B导致C, B导致E, C导致D, 以及D导致E。
问题：E是E的父节点吗？
答案（是或否？）：否

给定如下因果图：%s。
问题：%s是%s的父节点吗？
答案（是或否？）：""",
    'zero-shot-CoT':
    """You will be presented with a causal graph in the following form: %s.
Question: Does %s serve as the parent node of %s? Let's think step by step.
Answer (Yes or No ?):""",
    'zero-shot-CoT-CN':
    """给定如下因果图：%s。
问题：%s是%s的父节点吗？请逐步思考。
答案（是或否？）：""",
    'manual-CoT':
    """Here are eight examples for symbol causal attribution task of parents with chain of thought.

You will be presented with a causal graph in the following form: A causes D, A causes B, C causes E, and D causes E.
Question: Does D serve as the parent node of E?
Answer (Yes or No ?): D causes E, so D is the parent node of E. Thus the answer is Yes.

You will be presented with a causal graph in the following form: A causes B, B causes C, B causes D, and D causes E.
Question: Does E serve as the parent node of E?
Answer (Yes or No ?): E does not cause itself so E is not a parent node of E. Thus the answer is No.

You will be presented with a causal graph in the following form: A causes F, A causes B, B causes E, C causes F, C causes D, C causes E, and D causes E.
Question: Does C serve as the parent node of F?
Answer (Yes or No ?): C causes F, so C is the parent node of F. Thus the answer is Yes.

You will be presented with a causal graph in the following form: A causes E, A causes D, A causes F, B causes E, B causes C, C causes E, C causes D, C causes F, C causes G, D causes E, D causes F, E causes F, and F causes G.
Question: Does A serve as the parent node of G?
Answer (Yes or No ?): A does not cause G, so A is not a parent node of G. Thus the answer is No.

You will be presented with a causal graph in the following form: A causes H, A causes B, B causes H, B causes G, B causes D, B causes E, B causes I, B causes F, C causes F, C causes G, C causes E, D causes G, D causes I, D causes E, D causes F, F causes I, G causes H, G causes I, and H causes I.
Question: Does G serve as the parent node of I?
Answer (Yes or No ?): G causes I, so G is a parent node of I. Thus the answer is Yes.

You will be presented with a causal graph in the following form: A causes E, A causes J, A causes B, A causes G, B causes G, B causes F, B causes H, B causes D, C causes D, C causes G, C causes H, D causes F, D causes H, E causes F, E causes G, E causes K, E causes I, F causes K, F causes G, G causes J, G causes K, H causes K, and J causes K.
Question: Does D serve as the parent node of K?
Answer (Yes or No ?): D does not cause K, so D is not a parent node of K. Thus the answer is No.

You will be presented with a causal graph in the following form: A causes M, A causes C, A causes B, A causes H, A causes N, A causes E, A causes J, A causes F, B causes J, B causes C, B causes H, B causes N, B causes O, C causes H, C causes J, C causes I, C causes F, C causes L, D causes O, D causes Q, D causes I, D causes F, D causes G, D causes K, E causes Q, E causes J, E causes N, E causes G, F causes M, F causes K, F causes L, F causes I, G causes L, G causes M, G causes K, H causes Q, H causes M, H causes O, H causes K, I causes Q, I causes K, K causes P, K causes Q, L causes O, L causes P, M causes Q, N causes P, N causes O, and P causes Q.
Question: Does E serve as the parent node of Q?
Answer (Yes or No ?): E causes Q, so E is a parent node of Q. Thus the answer is Yes.

You will be presented with a causal graph in the following form: A causes G, A causes H, A causes C, A causes M, A causes K, A causes J, B causes I, B causes G, B causes E, B causes L, B causes H, B causes D, B causes J, C causes E, C causes K, C causes G, C causes L, C causes N, D causes N, D causes M, D causes G, D causes J, D causes K, D causes E, D causes I, E causes I, E causes F, E causes H, E causes N, E causes G, E causes J, F causes J, F causes I, F causes G, F causes N, F causes L, F causes K, G causes L, G causes J, G causes H, H causes M, H causes N, H causes L, I causes M, I causes K, I causes L, J causes N, J causes M, K causes M, K causes L, and M causes N.
Question: Does K serve as the parent node of N?
Answer (Yes or No ?): K does not cause N, so K cannot be a parent node of N. Thus the answer is No.

You will be presented with a causal graph in the following form: %s.
Question: Does %s serve as the parent node of %s?
Answer (Yes or No ?):""",
    'manual-CoT-CN':
    """如下为三个使用思维链进行推理的因果归因判断父节点的示例，和一个需要回答的问题。

给定如下因果图：A导致D, A导致C, A导致B, B导致E, B导致D, 以及C导致D。
问题：D是E的父节点吗？
答案（是或否？）：D没有导致E，D不是E的父节点。因此答案为“否”。

给定如下因果图：A导致B, A导致E, A导致C, B导致E, B导致C, C导致D, 以及C导致E。
问题：A是E的父节点吗？
答案（是或否？）：A导致E，A是E的父节点。因此答案为“是”。

给定如下因果图：A导致E, A导致B, B导致E, B导致C, C导致F, C导致E, 以及D导致F。
问题：A是F的父节点吗？
答案（是或否？）：A没有导致F，A不是F的父节点。因此答案为“否”。

给定如下因果图：%s。
问题：%s是%s的父节点吗？
答案（是或否？）：
""",
    'explicit-function':
    """You are a helpful assistant for causal attribution (parent node).
You will be presented with a causal graph in the following form: %s.
Question: Does %s serve as the parent node of %s?
Answer (Yes or No ?):""",
    'explicit-function-CN':
    """你是一个用于因果归因（父节点）的得力助手。
给定如下因果图：%s。
问题：%s是%s的父节点吗？
答案（是或否？）""",
}


def get_prompt(task_name, prompt_style, item, prompt_style_str=''):
    base = base_prompt_dict[prompt_style]

    prompt = prompt_style_str + base % (item['edges'], item['sampled_parent'],
                                        item['attribution'])
    return prompt
