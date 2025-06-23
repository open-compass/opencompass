from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.matbench.matbench import MatbenchDataset
from opencompass.datasets import generic_llmjudge_postprocess
from opencompass.datasets.matbench.post_process import numerical_llmjudge_postprocess
from opencompass.evaluator.generic_llm_evaluator import GenericLLMEvaluator

JUDGE_TEMPLATE = """
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


    <Original Question Begin>: \n{problem}\n<Original Question End>\n\n
    <Gold Target Begin>: \n{answer}\n<Gold Target End>\n\n
    <Predicted Answer Begin>: \n{prediction}\n<Predicted End>\n\n
    
    Judging the correctness of candidates' answers:
""".strip()

REGRESSION_JUDGE_TEMPLATE = """
Please function as a precise data extraction engine. Your sole task is to process the provided information and return a single specific numerical value as a float.

Follow these strict output rules:

1. Your output must be a single floating-point number.
2. Do not include any surrounding text, explanations, units, or additional characters.
3. If the provided answer is incomplete or does not yield a determined result, output 0.
4. Return only the numerical value without any reasoning or additional content.
Your specific task is to extract the float result from the following:

<Original Question Begin>: \n{problem}\n<Original Question End>\n\n
<Predicted Answer Begin>: \n{prediction}\n<Predicted Answer End>\n\n

Please provide the final numerical answer as a float:

""".strip()

matbench_reader_cfg = dict(
    input_columns=['problem'], output_column='answer')


matbench_tasks =  ['matbench_steels','matbench_expt_gap', 'matbench_expt_is_metal','matbench_glass']

matbench_datasets = []

for task in matbench_tasks:
    if task in ['matbench_glass','matbench_expt_is_metal']:
        matbench_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[dict(role='HUMAN', prompt=f'{{problem}}')])),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer))

        matbench_eval_cfg = dict(
            evaluator=dict(
                type=GenericLLMEvaluator,  # Use LLM as judge
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
                            dict(role='HUMAN', prompt=JUDGE_TEMPLATE), 
                        ],
                    ),
                ),
                dataset_cfg=dict(
                    type=MatbenchDataset,
                    path='opencompass/Matbench',
                    task=task,
                    reader_cfg=matbench_reader_cfg,
                ),
                judge_cfg=dict(),
                dict_postprocessor=dict(type=generic_llmjudge_postprocess),  
            ),
        )

    elif task in ['matbench_expt_gap','matbench_steels']:
        matbench_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[dict(role='HUMAN', prompt=f'{{problem}}')])),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer))

        matbench_eval_cfg = dict(
            evaluator=dict(
                type=GenericLLMEvaluator,  # 使用LLM作为评估器
                prompt_template=dict(
                    type=PromptTemplate,
                    template=dict(
                        begin=[
                            dict(
                                role='SYSTEM',
                                fallback_role='HUMAN',
                                prompt="You are a helpful assistant who extracts the answer as float of models' outputs.",
                            )
                        ],
                        round=[
                            dict(role='HUMAN', prompt=REGRESSION_JUDGE_TEMPLATE), 
                        ],
                    ),
                ),
                dataset_cfg=dict(
                    type=MatbenchDataset,
                    path='opencompass/Matbench',
                    task=task,
                    reader_cfg=matbench_reader_cfg,
                ),
                judge_cfg=dict(),
                dict_postprocessor=dict(type=numerical_llmjudge_postprocess),
            ),
        )

    matbench_datasets.append(
        dict(
            type=MatbenchDataset,
            path='opencompass/Matbench',
            abbr=task,
            task=task,
            reader_cfg=matbench_reader_cfg,
            infer_cfg=matbench_infer_cfg,
            eval_cfg=matbench_eval_cfg))

