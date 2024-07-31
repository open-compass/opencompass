# flake8: noqa: E501
base_prompt_dict = {
    'basic':
    """You will be presented with a causal graph in the following form: %s.
Question: Does %s serve as the ancestor node of %s?
Answer (Yes or No ?):""",
    'basic-CN':
    """给定如下因果图：%s。
问题：%s是%s的祖先节点吗？
答案（是或否？）：""",
    'adversarial-ignore':
    """You will be presented with a causal graph in the following form: %s.
Question: Does %s serve as the ancestor node of %s?
Answer (Yes or No ?):""",
    'adversarial-ignore-CN':
    """给定如下因果图：%s。
问题：%s是%s的祖先节点吗？
答案（是或否？）：""",
    'adversarial-doubt':
    """You will be presented with a causal graph in the following form: %s.
Question: Does %s serve as the ancestor node of %s?
Answer (Yes or No ?):""",
    'adversarial-doubt-CN':
    """给定如下因果图：%s。
问题：%s是%s的祖先节点吗？
答案（是或否？）：""",
    'zero-shot-IcL':
    """Determine whether or not a variable can serve as the ancestor of another variable in a given causal graph.
You will be presented with a causal graph in the following form: %s.
Question: Does %s serve as the ancestor node of %s?
Answer (Yes or No ?):""",
    'zero-shot-IcL-CN':
    """确定在给定的因果图中，一个变量是否可以作为另一个变量的祖先。
给定如下因果图：%s。
问题：%s是%s的祖先节点吗？
答案（是或否？）：""",
    'one-shot-IcL':
    """Determine whether or not a variable can serve as the ancestor of another variable in a given causal graph.
You will be presented with a causal graph in the following form: A causes D, A causes E, B causes E, C causes D, and D causes E.
Question: Does A serve as the ancestor node of E?
Answer (Yes or No ?): Yes

You will be presented with a causal graph in the following form: %s.
Question: Does %s serve as the ancestor node of %s?
Answer (Yes or No ?):""",
    'one-shot-IcL-CN':
    """确定在给定的因果图中，一个变量是否可以作为另一个变量的祖先。
给定如下因果图：A导致D, A导致E, B导致E, C导致D, 以及D导致E。
问题：A是E的祖先节点吗？
答案（是或否？）：是

给定如下因果图：%s。
问题：%s是%s的祖先节点吗？
答案（是或否？）：""",
    'three-shot-IcL':
    """Determine whether or not a variable can serve as the ancestor of another variable in a given causal graph.
You will be presented with a causal graph in the following form: A causes D, A causes E, B causes E, C causes D, and D causes E.
Question: Does A serve as the ancestor node of E?
Answer (Yes or No ?): Yes

You will be presented with a causal graph in the following form: A causes D, A causes B, C causes E, and D causes E.
Question: Does B serve as the ancestor node of E?
Answer (Yes or No ?): No

You will be presented with a causal graph in the following form: A causes B, A causes D, A causes E, B causes C, B causes E, C causes D, and D causes E.
Question: Does E serve as the ancestor node of E?
Answer (Yes or No ?): No

You will be presented with a causal graph in the following form: %s.
Question: Does %s serve as the ancestor node of %s?
Answer (Yes or No ?):""",
    'three-shot-IcL-CN':
    """确定在给定的因果图中，一个变量是否可以作为另一个变量的祖先。
给定如下因果图：A导致D, A导致E, B导致E, C导致D, 以及D导致E。
问题：A是E的祖先节点吗？
答案（是或否？）：是

给定如下因果图：A导致D, A导致B, C导致E, 以及D导致E。
问题：B是E的祖先节点吗？
答案（是或否？）：否

给定如下因果图：A导致B, A导致D, A导致E, B导致C, B导致E, C导致D, 以及D导致E。
问题：E是E的祖先节点吗？
答案（是或否？）：否

给定如下因果图：%s。
问题：%s是%s的祖先节点吗？
答案（是或否？）：""",
    'zero-shot-CoT':
    """You will be presented with a causal graph in the following form: %s.
Question: Does %s serve as the ancestor node of %s? Let's think step by step.
Answer (Yes or No ?):""",
    'zero-shot-CoT-CN':
    """给定如下因果图：%s。
问题：%s是%s的祖先节点吗？请逐步思考。
答案（是或否？）：""",
    'manual-CoT':
    """Here are eight examples for symbol causal attribution task of ancestors with chain of thought.

You will be presented with a causal graph in the following form: A causes D, A causes E, B causes E, C causes D, and D causes E.
Question: Does A serve as the ancestor node of E?
Answer (Yes or No ?): A causes E, so A is a direct ancestor of E. Thus the answer is Yes.

You will be presented with a causal graph in the following form: A causes D, A causes C, B causes D, B causes E, and C causes D.
Question: Does C serve as the ancestor node of E?
Answer (Yes or No ?): B is the only node causes E, and no node causes B. So E only has one ancestor B. Thus the answer is No.

You will be presented with a causal graph in the following form: A causes E, A causes B, A causes C, B causes D, B causes C, C causes D, and D causes E.
Question: Does B serve as the ancestor node of E?
Answer (Yes or No ?):Yes. B causes D and D causes E, so B is an ancestor of E. Thus the answer is Yes.

You will be presented with a causal graph in the following form: A causes E, A causes C, A causes B, B causes C, B causes E, and C causes D.
Question: Does E serve as the ancestor node of E?
Answer (Yes or No ?): E does not cause any node, so E cannot be the ancestor of itself. Thus the answer is No.

You will be presented with a causal graph in the following form: A causes B, A causes F, A causes E, A causes D, B causes C, B causes F, B causes D, B causes G, B causes H, C causes F, C causes H, C causes E, D causes H, E causes H, and F causes G.
Question: Does D serve as the ancestor node of H?
Answer (Yes or No ?): D causes H so D is the ancestor node of H. Thus the answer is Yes.

You will be presented with a causal graph in the following form: A causes G, B causes C, B causes H, B causes J, B causes D, C causes D, C causes E, C causes J, C causes G, D causes G, D causes I, E causes I, E causes F, E causes J, F causes H, F causes I, F causes J, F causes G, G causes I, and H causes I.
Question: Does A serve as the ancestor node of J?
Answer (Yes or No ?): A only causes G, G only causes I, and I causes none. So A cannot be the ancestor of J. Thus the answer is No.

You will be presented with a causal graph in the following form: A causes H, A causes G, A causes D, A causes F, B causes D, B causes H, B causes J, C causes K, C causes G, C causes E, C causes I, C causes H, C causes F, D causes J, D causes E, D causes G, D causes I, E causes J, E causes F, F causes I, F causes G, F causes J, G causes J, G causes I, G causes H, H causes I, H causes K, and J causes K.
Question: Does D serve as the ancestor node of K?
Answer (Yes or No ?): D causes J and J causes K. So D is an ancestor node of K. Thus the answer is Yes.

You will be presented with a causal graph in the following form: A causes I, A causes B, A causes F, A causes K, A causes G, B causes C, C causes G, D causes E, D causes H, D causes G, E causes J, E causes F, F causes I, F causes K, and H causes I.
Question: Does G serve as the ancestor node of K?
Answer (Yes or No ?): G causes none, so G cannot be the ancestor of K. Thus the answer is No.

You will be presented with a causal graph in the following form: %s.
Question: Does %s serve as the ancestor node of %s?
Answer (Yes or No ?):""",
    'manual-CoT-CN':
    """如下为三个使用思维链进行推理的因果归因判断祖先节点的示例，和一个需要回答的问题。

给定如下因果图：A导致E, A导致C, A导致D, B导致C, B导致D, C导致D, 以及D导致E。
问题：C是E的祖先节点吗？
答案（是或否？）：C导致D，D导致E，所以C是E的祖先节点。因此答案为“是”。

给定如下因果图：A导致E, A导致B, B导致E, B导致C, C导致F, C导致E, 以及D导致F。
问题：E是F的祖先节点吗？
答案（是或否？）：E没有导致任何节点，E不是任何节点的祖先节点。因此答案为“否”。

给定如下因果图：A导致C, A导致D, A导致F, B导致D, B导致E, C导致D, D导致E, 以及D导致F。
问题：A是F的祖先节点吗？
答案（是或否？）：A导致C，C导致D，D导致F，所以A是F的祖先节点。因此答案为“是”。

给定如下因果图：%s。
问题：%s是%s的祖先节点吗？
答案（是或否？）：
""",
    'explicit-function':
    """You are a helpful assistant for causal attribution (ancestor node).
You will be presented with a causal graph in the following form: %s.
Question: Does %s serve as the ancestor node of %s?
Answer (Yes or No ?):""",
    'explicit-function-CN':
    """你是一个用于因果归因（祖先节点）的得力助手。
给定如下因果图：%s。
问题：%s是%s的祖先节点吗？
答案（是或否？）：""",
}


def get_prompt(task_name, prompt_style, item, prompt_style_str=''):
    base = base_prompt_dict[prompt_style]

    prompt = prompt_style_str + base % (
        item['edges'], item['sampled_ancestor'], item['attribution'])
    return prompt
