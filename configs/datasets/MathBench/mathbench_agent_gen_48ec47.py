from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import AgentInferencer
from opencompass.openicl.icl_evaluator import CircularEvaluator, AccEvaluator
from opencompass.datasets import MathBenchDataset, mathbench_postprocess
from opencompass.utils.text_postprocessors import first_option_postprocess


PROMPT_EN = {
        'FEWSHOT_INSTRUCTION_CLOZE' : [
        dict(role='HUMAN', prompt='Mark\'s basketball team scores 25 2 pointers, 8 3 pointers and 10 free throws.  Their opponents score double the 2 pointers but half the 3 pointers and free throws.  What\'s the total number of points scored by both teams added together?'),
        dict(role='BOT', prompt='Tool:PythonInterpreter\nTool Input:```python\ndef solution():\n    mark_pointers_2 = 25 * 2\n    mark_pointers_3 = 8 * 3\n    mark_free_throws = 10 * 1\n    mark_points_scored = mark_pointers_2 + mark_pointers_3 + mark_free_throws\n    opponents_pointers_2 = mark_pointers_2 * 2\n    opponents_pointers_3 = mark_pointers_3 / 2\n    opponents_free_throws = mark_free_throws / 2\n    opponents_points_scored = opponents_pointers_2 + opponents_pointers_3 + opponents_free_throws\n    total_points_scored = mark_points_scored + opponents_points_scored\n    result = total_points_scored\n    return result\n```'),
        dict(role='SYSTEM', prompt='Response:210'),
        dict(role='BOT', prompt='Thought: According to the response, I got the answer\nFinalAnswer: 210'),

        dict(role='HUMAN', prompt='Bella has two times as many marbles as frisbees. She also has 20 more frisbees than deck cards. If she buys 2/5 times more of each item, what would be the total number of the items she will have if she currently has 60 marbles?'),
        dict(role='BOT', prompt='Tool:PythonInterpreter\nTool Input:```python\ndef solution():\n    marbles = 60\n    num_increased_marbles = marbles * 2 / 5\n    num_total_marbles = marbles + num_increased_marbles\n    frisbees = marbles / 2\n    num_increased_frisbees = frisbees * 2 / 5\n    num_total_frisbees = frisbees + num_increased_frisbees\n    deck_cards = frisbees - 20\n    num_increased_deck_cards = deck_cards * 2 / 5\n    num_total_deck_cards = deck_cards + num_increased_deck_cards\n    num_total = num_total_marbles + num_total_frisbees + num_total_deck_cards\n    result = num_total\n    return result\n```'),
        dict(role='SYSTEM', prompt='Response:140'),
        dict(role='BOT', prompt='Thought: According to the response, I got the answer\nFinalAnswer: 140'),

        dict(role='HUMAN', prompt='A group of 4 fruit baskets contains 9 apples, 15 oranges, and 14 bananas in the first three baskets and 2 less of each fruit in the fourth basket. How many fruits are there?'),
        dict(role='BOT', prompt="""Tool:PythonInterpreter\nTool Input:```python\ndef solution():\n    num_fruits_per_first_three_basket = 9 + 15 + 14\n    num_fruits_first_three_basket = num_fruits_per_first_three_basket * 3\n    num_apple_fourth_basket = 9 - 2\n    num_orange_fourth_basket = 15 - 2\n    num_banana_fourth_basket = 14 - 2\n    num_fruits_fourth_basket = num_apple_fourth_basket + num_orange_fourth_basket + num_banana_fourth_basket\n    num_fruits_total = num_fruits_first_three_basket + num_fruits_fourth_basket\n    result = num_fruits_total\n    return result\n```"""),
        dict(role='SYSTEM', prompt='Response:146'),
        dict(role='BOT', prompt='Thought: According to the response, I got the answer\nFinalAnswer: 146'),

        dict(role='HUMAN', prompt='{question}'),
    ],
    'FEWSHOT_INSTRUCTION_CHOICE' : [
        dict(role='HUMAN', prompt='Given point P(-1,4) lies on the graph of the inverse proportionality function $y=\\frac{{k}}{{x}}$ (k≠0), what is the value of k? A. $-\\frac{{1}}{{4}}$ B. $\\frac{{1}}{{4}}$ C. $4$ D. $-4$'),
        dict(role='BOT', prompt='Tool:PythonInterpreter\nTool Input:```python\nfrom sympy import solve, symbols, Ne, Eq\ndef solution():\n    k = symbols(\'k\')\n    result = solve([Eq(4, k / (-1)), Ne(k, 0)], k, dict=True)\n    return result\n```'),
        dict(role='SYSTEM', prompt='Response:{{Eq(k, -4)}}'),
        dict(role='BOT', prompt='Thought: Substituting point P into the function yields the value of k as $-4$\nFinalAnswer: D'),

        dict(role='HUMAN', prompt='The graph of the power function $y=(x)$ passes through the point$ (2, \\dfrac {{1}}{{4}}) $, what is the value of $f(-3)$? A. $\\frac{{1}}{{9}}$ B. $\\frac{{1}}{{8}})=196-x$ C. $\\frac{{2}}{{9}}$ D. $\\frac{{1}}{{4}}$'),
        dict(role='BOT', prompt='Tool:PythonInterpreter\nTool Input:```python\nfrom sympy import *\ndef solution():\n    x, y, k = symbols(\'x y k\')\n    eq1 = Eq(2**k, Rational(1, 4))\n    k_value = solve(eq1, k)[0]\n    y = x**k_value\n    result = y.subs(x, -3)\n    return result\n```'),
        dict(role='SYSTEM', prompt='Response:1/9'),
        dict(role='BOT', prompt='Thought: The functional expression of the power function is $y=x^{{-2}}$. Substituting $x=-3$ yields $y=$\\frac{{1}}{{9}}$\nFinalAnswer: A'),

        dict(role='HUMAN', prompt='If $3 x-y=12$, what is the value of $\\frac{8^{x}}{2^{y}} ?$\nA. The value cannot be determined from the information given.\nB. $2^{12}$\nC. 4\nD. $8^{2}$'),
        dict(role='BOT', prompt='Tool:PythonInterpreter\nTool Input:```python\nfrom sympy import symbols, Eq, solve\n\ndef sloution():\n   x, y = symbols(\'x y\')\n   equation = Eq(3*x - y, 12)\n\n   y_in_terms_of_x = solve(equation, y)[0]\n   expression = 8**x / 2**y_in_terms_of_x\n  result = expression.simplify()\n return result\n```'),
        dict(role='SYSTEM', prompt='Response:2**12'),
        dict(role='BOT', prompt='Thought: The value of $\\frac{8^{x}}{2^{y}}$ is $2^{12}$\nFinalAnswer: B'),

        dict(role='HUMAN', prompt='{question}'),
    ]
}

PROMPT_CN = {
    'FEWSHOT_INSTRUCTION_CLOZE' : [
        dict(role='HUMAN', prompt='Mark的篮球队得到25个2分球，8个3分球和10个罚球。他们的对手得到2分球的两倍，但3分球和罚球的一半。两队得分的总和是多少？'),
        dict(role='BOT', prompt='Tool:PythonInterpreter\nTool Input:```python\ndef solution():\n    mark_pointers_2 = 25 * 2\n    mark_pointers_3 = 8 * 3\n    mark_free_throws = 10 * 1\n    mark_points_scored = mark_pointers_2 + mark_pointers_3 + mark_free_throws\n    opponents_pointers_2 = mark_pointers_2 * 2\n    opponents_pointers_3 = mark_pointers_3 / 2\n    opponents_free_throws = mark_free_throws / 2\n    opponents_points_scored = opponents_pointers_2 + opponents_pointers_3 + opponents_free_throws\n    total_points_scored = mark_points_scored + opponents_points_scored\n    result = total_points_scored\n    return result\n```'),
        dict(role='SYSTEM', prompt='Response:210'),
        dict(role='BOT', prompt='Thought: 根据回答，我得到了答案\nFinalAnswer: 210'),

        dict(role='HUMAN', prompt='Bella有两倍于飞盘的弹珠。她还比卡片多20个飞盘。如果她买每种物品多2/5，她会有多少总数的物品，如果她现在有60颗弹珠？'),
        dict(role='BOT', prompt='Tool:PythonInterpreter\nTool Input:```python\ndef solution():\n    marbles = 60\n    num_increased_marbles = marbles * 2 / 5\n    num_total_marbles = marbles + num_increased_marbles\n    frisbees = marbles / 2\n    num_increased_frisbees = frisbees * 2 / 5\n    num_total_frisbees = frisbees + num_increased_frisbees\n    deck_cards = frisbees - 20\n    num_increased_deck_cards = deck_cards * 2 / 5\n    num_total_deck_cards = deck_cards + num_increased_deck_cards\n    num_total = num_total_marbles + num_total_frisbees + num_total_deck_cards\n    result = num_total\n    return result\n```'),
        dict(role='SYSTEM', prompt='Response:140'),
        dict(role='BOT', prompt='Thought: 根据回答，我得到了答案\nFinalAnswer: 140'),

        dict(role='HUMAN', prompt='一个有4个水果篮子，前三个篮子里有9个苹果、15个橙子和14个香蕉，第四个篮子里每种水果都少2个。总共有多少水果？'),
        dict(role='BOT', prompt="""Tool:PythonInterpreter\nTool Input:```python\ndef solution():\n    num_fruits_per_first_three_basket = 9 + 15 + 14\n    num_fruits_first_three_basket = num_fruits_per_first_three_basket * 3\n    num_apple_fourth_basket = 9 - 2\n    num_orange_fourth_basket = 15 - 2\n    num_banana_fourth_basket = 14 - 2\n    num_fruits_fourth_basket = num_apple_fourth_basket + num_orange_fourth_basket + num_banana_fourth_basket\n    num_fruits_total = num_fruits_first_three_basket + num_fruits_fourth_basket\n    result = num_fruits_total\n    return result\n```"""),
        dict(role='SYSTEM', prompt='Response:146'),
        dict(role='BOT', prompt='Thought: 根据回答，我得到了答案\nFinalAnswer: 146'),

        dict(role='HUMAN', prompt='{question}'),
    ],
    'FEWSHOT_INSTRUCTION_CHOICE' : [
        dict(role='HUMAN', prompt='已知点P（-1，4）在反比例函数$y=\\frac{{k}}{{x}}$ (k≠0)的图象上，则k的值是____'),
        dict(role='BOT', prompt='Tool:PythonInterpreter\nTool Input:```python\nfrom sympy import solve, symbols, Ne, Eq\ndef solution():\n    k = symbols(\'k\')\n    result = solve([Eq(4, k / (-1)), Ne(k, 0)], k, dict=True)\n    return result\n```'),
        dict(role='SYSTEM', prompt='Response:{{Eq(k, -4)}}'),
        dict(role='BOT', prompt='Thought: 将点 P 带入函数解出 k 的值为 $-4$\nFinalAnswer: D'),

        dict(role='HUMAN', prompt='幂函数$ y=(x) $的图象经过点$ (2, \\dfrac {{1}}{{4}}) $，则$ f(-3) $的值为 ______ ．'),
        dict(role='BOT', prompt='Tool:PythonInterpreter\nTool Input:```python\nfrom sympy import *\ndef solution():\n    x, y, k = symbols(\'x y k\')\n    eq1 = Eq(2**k, Rational(1, 4))\n    k_value = solve(eq1, k)[0]\n    y = x**k_value\n    result = y.subs(x, -3)\n    return result\n```'),
        dict(role='SYSTEM', prompt='Response:1/9'),
        dict(role='BOT', prompt='Thought: 求出幂函数的函数表达式为 $y=x^{{-2}}$，代入 $x=-3$ 得到 $y=$\\frac{{1}}{{9}}$\nFinalAnswer: A'),

        dict(role='HUMAN', prompt='如果$3 x-y=12$，则$\\frac{8^{x}}{2^{y}}$的值是多少？\nA. 无法从给定的信息中确定值。\nB. $2^{12}$\nC. 4\nD. $8^{2}$'),
        dict(role='BOT', prompt='Tool:PythonInterpreter\nTool Input:```python\nfrom sympy import symbols, Eq, solve\n\ndef sloution():\n   x, y = symbols(\'x y\')\n   equation = Eq(3*x - y, 12)\n\n   y_in_terms_of_x = solve(equation, y)[0]\n   expression = 8**x / 2**y_in_terms_of_x\n  result = expression.simplify()\n return result\n```'),
        dict(role='SYSTEM', prompt='Response:2**12'),
        dict(role='BOT', prompt='Thought: $\\frac{8^{x}}{2^{y}}$的值是$2^{12}$\nFinalAnswer: B'),

        dict(role='HUMAN', prompt='{question}'),
    ]
}

mathbench_sets = {
    'college': ['single_choice_cn', 'cloze_en'],
    'high': ['single_choice_cn', 'single_choice_en'],
    'middle': ['single_choice_cn'],
    'primary': ['cloze_cn'],
    'primary_refine': ['refine_cloze_cn']
}

# Use circular evaluation or not
with_circular_eval = True

mathbench_agent_datasets = []

for _split in list(mathbench_sets.keys()):
    for _name in mathbench_sets[_split]:
        prompt_example = PROMPT_CN if '_cn' in _name else PROMPT_EN
        mathbench_infer_cfg = dict(
            prompt_template=dict(type=PromptTemplate,
                                 template=dict(
                                     round = prompt_example['FEWSHOT_INSTRUCTION_CLOZE'] if 'cloze' in _name else prompt_example['FEWSHOT_INSTRUCTION_CHOICE'])),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=AgentInferencer)
        )

        mathbench_eval_cfg = dict(
            evaluator=dict(type=CircularEvaluator if 'choice' in _name and with_circular_eval else AccEvaluator),
            pred_postprocessor=dict(type=first_option_postprocess, options='ABCD') if 'single_choice' in _name else dict(type=mathbench_postprocess, name=_name))

        mathbench_agent_datasets.append(
            dict(
                abbr='mathbench-' + _split + '-' + _name + '-agent',
                type=MathBenchDataset,
                path=f'./data/mathbench/{_split}',
                name=_name,
                with_circular=with_circular_eval,
                reader_cfg=dict(
                    input_columns=['question'],
                    output_column='answer'
                    ),
                infer_cfg=mathbench_infer_cfg,
                eval_cfg=mathbench_eval_cfg,
            ))
