
EVAL_PROMPT = (
    "You are a helpful assistant who evaluates the correctness and quality of models' outputs.\nPlease as a grading "
    'expert, judge whether the final answers given by the candidates below are consistent with the standard answers, '
    'that is, whether the candidates answered correctly. \n    \n    Here are some evaluation criteria:\n    '
    "1. Please refer to the given standard answer. You don't need to re-generate the answer to the question because "
    "the standard answer has been given. You only need to judge whether the candidate's answer is consistent with the "
    "standard answer according to the form of the question. Don't try to answer the original question. You can assume "
    "that the standard answer is definitely correct.\n    2. Because the candidate's answer may be different from the "
    'standard answer in the form of expression, before making a judgment, please understand the question and the '
    "standard answer first, and then judge whether the candidate's answer is correct, but be careful not to try to "
    'answer the original question.\n    3. Some answers may contain multiple items, such as multiple-choice questions, '
    'multiple-select questions, fill-in-the-blank questions, etc. As long as the answer is the same as the standard '
    'answer, it is enough. For multiple-select questions and multiple-blank fill-in-the-blank questions, the candidate '
    'needs to answer all the corresponding options or blanks correctly to be considered correct.\n    4. Some answers '
    'may be expressed in different ways, such as some answers may be a mathematical expression, some answers may be a '
    'textual description, as long as the meaning expressed is the same. And some formulas are expressed in different '
    'ways, but they are equivalent and correct.\n    5. If the prediction is given with \\boxed{{}}, please ignore '
    "the \\boxed{{}} and only judge whether the candidate's answer is consistent with the standard answer.\n\n    "
    'Please judge whether the following answers are consistent with the standard answer based on the above criteria. '
    'Grade the predicted answer of this new question as one of:\n    A: CORRECT \n    B: INCORRECT\n    Just return '
    "the letters \"A\" or \"B\", with no text around it.\n\n    Here is your task. Simply reply with either CORRECT, "
    "INCORRECT. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer."
    '\n\n\n    <Original Question Begin>: \n\n{question}\n\n<Original Question End>\n\n\n    '
    '<Gold Target Begin>: \n{answer}\n<Gold Target End>\n\n\n    <Predicted Answer Begin>: \n{prediction}\n'
    "<Predicted End>\n\n\n    \n    Judging the correctness of candidates' answers:\"\n"
)

datasets = [
    dict(
        abbr='UGD_hard',
        path='./data/UGD_hard_oc.jsonl',
        evaluator=dict(
            type='llm_evaluator',
            judge_cfg=dict(),
            prompt=EVAL_PROMPT,
        ),
        n=1,
    ),
]