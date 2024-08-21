# flake8: noqa: E501
base_prompt_dict = {
    'basic':
    """You will be presented with a causal graph in the following form: %s.
There exist unobserved confounders between: %s.
Question: Whether the causal effect of %s on %s is identified or not?
Answer (Yes or No ?):""",
    'basic-CN':
    """给定如下因果图：%s。
在这些变量间存在着不可观察的混淆变量：%s。
问题：%s对%s的因果效应是否可以被识别？
答案（是或否？）：""",
    'adversarial-ignore':
    """You will be presented with a causal graph in the following form: %s.
There exist unobserved confounders between: %s.
Question: Whether the causal effect of %s on %s is identified or not?
Answer (Yes or No ?):""",
    'adversarial-ignore-CN':
    """给定如下因果图：%s。
在这些变量间存在着不可观察的混淆变量：%s。
问题：%s对%s的因果效应是否可以被识别？
答案（是或否？）：""",
    'adversarial-doubt':
    """You will be presented with a causal graph in the following form: %s.
There exist unobserved confounders between: %s.
Question: Whether the causal effect of %s on %s is identified or not?
Answer (Yes or No ?):""",
    'adversarial-doubt-CN':
    """给定如下因果图：%s。
在这些变量间存在着不可观察的混淆变量：%s。
问题：%s对%s的因果效应是否可以被识别？
答案（是或否？）：""",
    'zero-shot-IcL':
    """Determine whether the causal effect can be identified given two variables on a causal graph.
You will be presented with a causal graph in the following form: %s.
There exist unobserved confounders between: %s.
Question: Whether the causal effect of %s on %s is identified or not?
Answer (Yes or No ?):""",
    'zero-shot-IcL-CN':
    """确定在因果图中给定两个变量的情况下，因果效应是否可以被识别。
给定如下因果图：%s。
在这些变量间存在着不可观察的混淆变量：%s。
问题：%s对%s的因果效应是否可以被识别？
答案（是或否？）：""",
    'one-shot-IcL':
    """Determine whether the causal effect can be identified given two variables on a causal graph.
You will be presented with a causal graph in the following form: A causes E, A causes C, A causes B, B causes D, B causes E, and D causes E.
There exist unobserved confounders between: B and E.
Question: Whether the causal effect of B on E is identified or not?
Answer (Yes or No ?): No

You will be presented with a causal graph in the following form: %s.
There exist unobserved confounders between: %s.
Question: Whether the causal effect of %s on %s is identified or not?
Answer (Yes or No ?):""",
    'one-shot-IcL-CN':
    """确定在因果图中给定两个变量的情况下，因果效应是否可以被识别。
给定如下因果图：A导致E, A导致C, A导致B, B导致D, B导致E, 以及D导致E。
在这些变量间存在着不可观察的混淆变量：B和E。
问题：B对E的因果效应是否可以被识别？
答案（是或否？）：否

给定如下因果图：%s。
在这些变量间存在着不可观察的混淆变量：%s。
问题：%s对%s的因果效应是否可以被识别？
答案（是或否？）：""",
    'three-shot-IcL':
    """Determine whether the causal effect can be identified given two variables on a causal graph.
You will be presented with a causal graph in the following form: A causes E, A causes C, A causes B, B causes D, B causes E, and D causes E.
There exist unobserved confounders between: B and E.
Question: Whether the causal effect of B on E is identified or not?
Answer (Yes or No ?): No

You will be presented with a causal graph in the following form: A causes D, A causes E, B causes E, C causes D, and D causes E.
There exist unobserved confounders between: C and D, and A and E.
Question: Whether the causal effect of C on D is identified or not?
Answer (Yes or No ?): No

You will be presented with a causal graph in the following form: A causes D, A causes C, A causes B, B causes E, B causes D, and C causes D.
There exist unobserved confounders between: B and D, C and D, and A and B.
Question: Whether the causal effect of D on C is identified or not?
Answer (Yes or No ?): Yes

You will be presented with a causal graph in the following form: %s.
There exist unobserved confounders between: %s.
Question: Whether the causal effect of %s on %s is identified or not?
Answer (Yes or No ?):""",
    'three-shot-IcL-CN':
    """确定在因果图中给定两个变量的情况下，因果效应是否可以被识别。
给定如下因果图：A导致E, A导致C, A导致B, B导致D, B导致E, 以及D导致E。
在这些变量间存在着不可观察的混淆变量：B和E。
问题：B对E的因果效应是否可以被识别？
答案（是或否？）：否

给定如下因果图：A导致D, A导致E, B导致E, C导致D, 以及D导致E。
在这些变量间存在着不可观察的混淆变量：C和D, 以及A和E。
问题：C对D的因果效应是否可以被识别？
答案（是或否？）：否

给定如下因果图：A导致D, A导致C, A导致B, B导致E, B导致D, 以及C导致D。
在这些变量间存在着不可观察的混淆变量：B和D, C和D, 以及A和B。
问题：D对C的因果效应是否可以被识别？
答案（是或否？）：是

给定如下因果图：%s。
在这些变量间存在着不可观察的混淆变量：%s。
问题：%s对%s的因果效应是否可以被识别？
答案（是或否？）：""",
    'zero-shot-CoT':
    """You will be presented with a causal graph in the following form: %s.
There exist unobserved confounders between: %s.
Question: Whether the causal effect of %s on %s is identified or not? Let's think step by step.
Answer (Yes or No ?):""",
    'zero-shot-CoT-CN':
    """给定如下因果图：%s。
在这些变量间存在着不可观察的混淆变量：%s。
问题：%s对%s的因果效应是否可以被识别？请逐步思考。
答案（是或否？）：""",
    'manual-CoT':
    """Here are three examples of causal effect identification using chain of thought, and a question to answer.

You will be presented with a causal graph in the following form: A causes E, A causes D, B causes D, B causes E, C causes E, and D causes E.
There exist unobserved confounders between: B and D.
Question: Whether the causal effect of B on E is identified or not?
Answer (Yes or No ?): The unobserved confounders between B and D suggests there might be a causal path from the confounder to B. Therefore, there may be an unblocked back-door path from B to E, making the causal effect of B on E not identified. Therefore, the answer is No.

You will be presented with a causal graph in the following form: A causes B, B causes C, B causes D, and D causes E.
There exist unobserved confounders between: .
Question: Whether the causal effect of A on B is identified or not?
Answer (Yes or No ?): There are no unobserved confounders, and there is no unblocked back-door path from A to B, so the causal effect of A on B can be identified. Therefore, the answer is Yes.

You will be presented with a causal graph in the following form: A causes D, A causes C, B causes D, B causes E, and C causes D.
There exist unobserved confounders between: B and D, and C and D.
Question: Whether the causal effect of A on B is identified or not?
Answer (Yes or No ?): There are no unobserved confounders between A and B, and there is no unblocked back-door path from A to B, so the causal effect of A on B can be identified. Therefore, the answer is Yes.

You will be presented with a causal graph in the following form: %s.
There exist unobserved confounders between: %s.
Question: Whether the causal effect of %s on %s is identified or not?
Answer (Yes or No ?):
""",
    'manual-CoT-CN':
    """如下为两个使用思维链进行推理的判断因果效应可否识别的示例，和一个需要回答的问题。

给定如下因果图：A导致E, A导致D, B导致D, B导致E, C导致E, 以及D导致E。
在这些变量间存在着不可观察的混淆变量：B和D。
问题：B对E的因果效应是否可以被识别？
答案（是或否？）：B和D之间存在不可观察的混淆变量说明可能存在从混淆变量指向B的因果路径。因此B到E可能存在无法被阻断的后门路径，导致B对E的因果效应不可被识别。因此答案为“否”。

给定如下因果图：A导致B, B导致C, B导致D, 以及D导致E。
在这些变量间存在着不可观察的混淆变量：。
问题：A对B的因果效应是否可以被识别？
答案（是或否？）：不存在不可观察的混淆变量，A到B不存在无法被阻断的后门路径，所以A对B的因果效应可以被识别。因此答案为“是”。

给定如下因果图：%s。
在这些变量间存在着不可观察的混淆变量：%s。
问题：%s对%s的因果效应是否可以被识别？
答案（是或否？）：
""",
    'explicit-function':
    """You are a helpful assistant for causality identification.
You will be presented with a causal graph in the following form: %s.
There exist unobserved confounders between: %s.
Question: Whether the causal effect of %s on %s is identified or not?
Answer (Yes or No ?):""",
    'explicit-function-CN':
    """你是一个用于因果识别的得力助手。
给定如下因果图：%s。
在这些变量间存在着不可观察的混淆变量：%s。
问题：%s对%s的因果效应是否可以被识别？
答案（是或否？）：""",
}


def get_prompt(task_name, prompt_style, item, prompt_style_str=''):
    base = base_prompt_dict[prompt_style]

    prompt = prompt_style_str + base % (item['di_edges'], item['bi_edges'],
                                        item['treatment'], item['outcome'])
    return prompt
