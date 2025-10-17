from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GPQADataset, GPQA_Simple_Eval_postprocess
from opencompass.evaluator import GenericLLMEvaluator, CascadeEvaluator
from opencompass.datasets import generic_llmjudge_postprocess
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.utils.text_postprocessors import match_answer_pattern

# openai_simple_eval prompt
align_prompt = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD.

{question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()


GRADER_TEMPLATE = """
Please as a grading expert, judge whether the final answers given by the candidates below are consistent with the standard answers, that is, whether the candidates answered correctly. 
Here are some evaluation criteria:
1. Please refer to the given standard answer. You don't need to re-generate the answer to the question because the standard answer has been given. You only need to judge whether the candidate's answer is consistent with the standard answer according to the form of the question. THE STANDARD ANSWER IS ALWAYS CORRECT AND THE QUESTION IS PERFECTLY VALID. NEVER QUESTION THEM.
2. ONLY compare the FINAL ANSWER - COMPLETELY IGNORE any potential errors in the REASONING PROCESSES.
3. Some answers may be expressed in different ways, such as some answers may be a mathematical expression, some answers may be a textual description, as long as the meaning expressed is the same. Before making a judgment, please understand the question and the standard answer first, and then judge whether the candidate's answer is correct.
4. Some answers may consist of multiple items, such as multiple-choice questions, multiple-select questions, fill-in-the-blank questions, etc. Regardless of the question type, the final answer will be considered correct as long as it matches the standard answer, regardless of whether the reasoning process is correct. For multiple-select questions and multi-blank fill-in-the-blank questions, all corresponding options or blanks must be answered correctly and match the standard answer exactly to be deemed correct.
5. If the prediction is given with \\boxed{{}}, please ignore the \\boxed{{}} and only judge whether the candidate's answer is consistent with the standard answer.
6. If the candidate's answer is invalid (e.g., incomplete (cut off mid-response), lots of unnormal repetitive content, or irrelevant to the question, saying it can't answer the question because some irresistible factors, like ethical issues, no enough information, etc.), select option C (INVALID).Please judge whether the following answers are consistent with the standard answer based on the above criteria. Grade the predicted answer of this new question as one of:
A: CORRECT 
B: INCORRECT
C: INVALID
Just return the letters "A", "B", or "C", with no text around it.
Here is your task. Simply reply with either CORRECT, INCORRECT, or INVALID. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
<Original Question Begin>:
{question}\n A) {A}\n B) {B}\n C) {C}\n D) {D}\n
<Original Question End>
<Standard Answer Begin>:
{answer}
<Standard Answer End>
<Candidate's Answer Begin>: 
{prediction}
<Candidate's Answer End>
Judging the correctness of the candidate's answer:
""".strip()


gpqa_reader_cfg = dict(
    input_columns=['question', 'A', 'B', 'C', 'D'],
    output_column='answer')

gpqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt=align_prompt),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))



gpqa_datasets = []
gpqa_subsets = {
    # 'extended': 'gpqa_extended.csv',
    # 'main': 'gpqa_main.csv',
    'diamond': 'gpqa_diamond.csv'
}

for split in list(gpqa_subsets.keys()):
    gpqa_eval_cfg = dict(
        evaluator=dict(
            type=CascadeEvaluator,
            rule_evaluator=dict(
                type=AccEvaluator,
                pred_postprocessor=dict(type=match_answer_pattern, answer_pattern=r'(?i)ANSWER\s*:\s*([A-D])'),
            ),
            llm_evaluator=dict(
                type=GenericLLMEvaluator,
                prompt_template=dict(
                    type=PromptTemplate,
                    template=dict(
                    begin=[
                        dict(
                            role='SYSTEM',
                            fallback_role='HUMAN',
                            prompt="You are a helpful assistant who evaluates the correctness and quality of models' outputs.")
                    ],
                        round=[
                        dict(
                            role='HUMAN',
                            prompt = GRADER_TEMPLATE
                        ),
                    ]),
                ),
                dataset_cfg=dict(
                    type=GPQADataset,
                    path='./data/gpqa/',
                    name=gpqa_subsets[split],
                    reader_cfg=gpqa_reader_cfg,
                ),
                judge_cfg=dict(),
                dict_postprocessor=dict(type=generic_llmjudge_postprocess),
            ),
            parallel=False,
        ),
    )
    gpqa_datasets.append(
        dict(
            abbr='GPQA_' + split + '_repeat_4',
            type=GPQADataset,
            path='./data/gpqa/',
            name=gpqa_subsets[split],
            reader_cfg=gpqa_reader_cfg,
            infer_cfg=gpqa_infer_cfg,
            eval_cfg=gpqa_eval_cfg,
            mode='singlescore',
            n=4,
        )
    )