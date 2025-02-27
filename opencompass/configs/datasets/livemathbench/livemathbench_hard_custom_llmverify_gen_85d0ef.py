from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.evaluator import GenericLLMEvaluator
from opencompass.datasets import CustomDataset
from opencompass.datasets import generic_llmjudge_postprocess
from itertools import product

livemathbench_reader_cfg = dict(input_columns=['question'], output_column='answer')


# Inference configuration
livemathbench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='{question}\n',
                ),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)


# Template for the LLM judge
GRADER_TEMPLATE = """
    Please as a grading expert, judge whether the final answers given by the candidates below are consistent with the standard answers, that is, whether the candidates answered correctly. 
    
    Here are some evaluation criteria:
    1. Please refer to the given standard answer. You don't need to re-generate the answer to the question because the standard answer has been given. You only need to judge whether the candidate's answer is consistent with the standard answer according to the form of the question. Don't try to answer the original question. You can assume that the standard answer is definitely correct.
    2. Because the candidate's answer may be different from the standard answer in the form of expression, before making a judgment, please understand the question and the standard answer first, and then judge whether the candidate's answer is correct, but be careful not to try to answer the original question.
    3. Some answers may contain multiple items, such as multiple-choice questions, multiple-select questions, fill-in-the-blank questions, etc. As long as the answer is the same as the standard answer, it is enough. For multiple-select questions and multiple-blank fill-in-the-blank questions, the candidate needs to answer all the corresponding options or blanks correctly to be considered correct.
    4. Some answers may be expressed in different ways, such as some answers may be a mathematical expression, some answers may be a textual description, as long as the meaning expressed is the same. And some formulas are expressed in different ways, but they are equivalent and correct.
    5. If the prediction is given with \\boxed{}, please ignore the \\boxed{} and only judge whether the candidate's answer is consistent with the standard answer.
    Please judge whether the following answers are consistent with the standard answer based on the above criteria. Grade the predicted answer of this new question as one of:
    A: CORRECT 
    B: INCORRECT
    Just return the letters "A" or "B", with no text around it.
    Here is your task. Simply reply with either CORRECT, INCORRECT. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
    <Original Question Begin>: \n{question}\n<Original Question End>\n\n
    <Gold Target Begin>: \n{answer}\n<Gold Target End>\n\n
    <Predicted Answer Begin>: \n{prediction}\n<Predicted End>\n\n
    
    Judging the correctness of candidates' answers:
""".strip()



splits = ['hard_cn', 'hard_en']
# Dataset configuration
livemathbench_datasets = [
    dict(
        type=CustomDataset,
        abbr=f'livemathbench_hard_custom_{split}_run{run_idx}',
        path='data/LiveMathBench',
        local_mode=True,
        file_name=f'202412/{split}.jsonl',
        reader_cfg=livemathbench_reader_cfg,
        infer_cfg=livemathbench_infer_cfg,
        eval_cfg=dict(
            # # Evaluation configuration using LLM as judge
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
                    type=CustomDataset,
                    path='data/LiveMathBench',
                    local_mode=True,
                    file_name=f'202412/{split}.jsonl',
                    reader_cfg=livemathbench_reader_cfg,
                ),
                judge_cfg={},
                dict_postprocessor=dict(type=generic_llmjudge_postprocess),
            ),
        ),
    ) for split, run_idx in product(splits, range(8))
]
