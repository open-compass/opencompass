OPTION_NAVIE_PROMPT_TEMPLATE = """
There is a detailed explanation of the final answer you should extract:
1. You should extract the final answer option like 'A', 'B', 'C', 'D' ... from the given output sentences.
2. The question is a single choice question, so the final answer option should be one of the options, not a combination of options.
""" # noqa

MATH_NAVIE_PROMPT_TEMPLATE = """
This is a detailed explanation of the final answer you should extract:
1. The question type is a math question, so the final answer should be a number, set, vector, matrix, interval, expression, function, equation, or inequality and any combination of them.
2. If the final answer includes additional symbols, such as units, you should exclude them and only extract the pure final answer.
""" # noqa
