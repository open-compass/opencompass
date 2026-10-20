from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.evaluator import GenericLLMEvaluator
from opencompass.datasets import generic_llmjudge_postprocess
from opencompass.datasets import RBenchDataset
from opencompass.utils.text_postprocessors import first_option_postprocess

RBench_reader_cfg = dict(
    input_columns=[
        'RBench_Question_Input',
        'RBench_Option_A',
        'RBench_Option_B',
        'RBench_Option_C',
        'RBench_Option_D',
        'RBench_Option_E',
        'RBench_Option_F',
    ],
    output_column='target',
)

RBench_datasets = []

systemp_prompt = "Answer the following single choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of Options(e.g. one of ABCDEF). Think step by step before answering."

GRADER_TEMPLATE = """
    Please as a grading expert, judge whether the final answers given by the candidates below are consistent with the standard answers, that is, whether the candidates answered correctly. 
    
    Here are some evaluation criteria:
    1. Please refer to the given standard answer. You don't need to re-generate the answer to the question because the standard answer has been given. You only need to judge whether the candidate's answer is consistent with the standard answer according to the form of the question. Don't try to answer the original question. You can assume that the standard answer is definitely correct.
    2. Because the candidate's answer may be different from the standard answer in the form of expression, before making a judgment, please understand the question and the standard answer first, and then judge whether the candidate's answer is correct, but be careful not to try to answer the original question.
    3. Some answers may contain multiple items, such as multiple-choice questions, multiple-select questions, fill-in-the-blank questions, etc. As long as the answer is the same as the standard answer, it is enough. For multiple-select questions and multiple-blank fill-in-the-blank questions, the candidate needs to answer all the corresponding options or blanks correctly to be considered correct.
    4. Some answers may be expressed in different ways, such as some answers may be a mathematical expression, some answers may be a textual description, as long as the meaning expressed is the same. And some formulas are expressed in different ways, but they are equivalent and correct.

    Please judge whether the following answers are consistent with the standard answer based on the above criteria. Grade the predicted answer of this new question as one of:
    A: CORRECT 
    B: INCORRECT
    Just return the letters "A" or "B", with no text around it.

    Here is your task. Simply reply with either CORRECT, INCORRECT. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.

    <Original Question Begin>: {systemp_prompt}\nQuestion: {{RBench_Question_Input}}\nA. {{RBench_Option_A}}\nB. {{RBench_Option_B}}\nC. {{RBench_Option_C}}\nD. {{RBench_Option_D}}\nE. {{RBench_Option_E}}\nF. {{RBench_Option_F}}\n<Original Question End>\n\n
    <Gold Target Begin>: \n{target}\n<Gold Target End>\n\n
    <Predicted Answer Begin>: \n{prediction}\n<Predicted End>\n\n
    Judging the correctness of candidates' answers:
""".strip()

RBench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt=f'{systemp_prompt}\nQuestion: {{RBench_Question_Input}}\nA. {{RBench_Option_A}}\nB. {{RBench_Option_B}}\nC. {{RBench_Option_C}}\nD. {{RBench_Option_D}}\nE. {{RBench_Option_E}}\nF. {{RBench_Option_F}}\nAnswer: ',
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)
RBench_datasets = []

for subset in ['en', 'zh']:
    RBench_eval_cfg = dict(
        evaluator=dict(
            type=GenericLLMEvaluator,
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    begin=[
                        dict(
                            role='SYSTEM',
                            fallback_role='HUMAN',
                            prompt="You are a helpful assistant who evaluates the correctness and quality of models' outputs.",
                        )
                    ],
                    round=[
                        dict(role='HUMAN', prompt=GRADER_TEMPLATE),
                    ],
                ),
            ),
            dataset_cfg=dict(
                type=RBenchDataset,
                path='R-Bench/R-Bench',
                name='R-Bench',
                subset=subset,
                reader_cfg=RBench_reader_cfg,
            ),
            judge_cfg=dict(),
            dict_postprocessor=dict(type=generic_llmjudge_postprocess),
        ),
    )

    RBench_datasets.append(
        dict(
            abbr=f'R-Bench_{subset}',
            type=RBenchDataset,
            path='R-Bench/R-Bench',
            name='R-Bench',
            subset=subset,
            reader_cfg=RBench_reader_cfg,
            infer_cfg=RBench_infer_cfg,
            eval_cfg=RBench_eval_cfg,
        )
    )
