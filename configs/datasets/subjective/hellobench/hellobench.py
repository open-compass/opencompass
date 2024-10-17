from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.datasets import HelloBenchDataset, hellobench_postprocess

system_prompt = """You are a helpful evaluator. Your task is to evaluate the checklists of the responses given by the Large Language Models (LLMs) based on user instructions. These checklists consist of yes or no questions."""

user_prompt = """Your core task is to evaluate the checklists based on the user’s instruction and LLM’s response, with each checklist item being a yes or no question indicating a specific aspect that the LLM’s response should meet. You need to judge the checklist item based on the instruction and response. The evaluation results are scored from 0 to 1, with 5 scores in total, which are:

0: The response fails to meet the checklist requirements, demonstrating substantial need for improvement across multiple areas.
0.25: The response partially meets some checklist requirements, but significant elements remain unaddressed.
0.5: The response meets several checklist requirements, yet the overall evaluation appears ambiguous or unclear.
0.75: The response aligns with most checklist requirements, though there are still minor areas that could be refined or enhanced.
1: The response fully satisfies all checklist requirements, with no identifiable issues or areas for improvement. It means this response is already perfect; you can't find any significant flaws in it.

Here is the instruction:
{{\"instruction\": {instruction}}}

Here is the response given by LLM:
{{\"response\": {prediction}}}

Since the response may be rather long, I am specifically reminding you here that the response has ended.

Here are checklists of this instruction:
{{\"checklists\": {formatted_checklists}}}

To further remind you, I will repeat my requirements:

Your core task is to evaluate the checklists based on the user’s instruction and LLM’s response, with each checklist item being a yes or no question indicating a specific aspect that the LLM’s response should meet. You need to judge the checklist item based on the instruction and response. The evaluation results are scored from 0 to 1, with 5 scores in total, which are:

0: The response fails to meet the checklist requirements, demonstrating substantial need for improvement across multiple areas.
0.25: The response partially meets some checklist requirements, but significant elements remain unaddressed.
0.5: The response meets several checklist requirements, yet the overall evaluation appears ambiguous or unclear.
0.75: The response aligns with most checklist requirements, though there are still minor areas that could be refined or enhanced.
1: The response fully satisfies all checklist requirements, with no identifiable issues or areas for improvement. It means this response is already perfect; you can't find any significant flaws in it.

Always provide the reason for your evaluation results. You should be strict but fair in your evaluation. A score of 1 means that the response perfectly meets all the checklist requirements and you think there are really no room for improvements. When giving a score of 1, you need to carefully consider whether this checklist has been perfectly satisfied.

Evaluate all the checklists and return the evaluation results of the checklists. Output a Python List consisting of the Python Dictionary formatted as follows:
[{{\"checklist_id\": \"the id of the checklist\", \"reason\": \"The reason for your evaluation results\", \"evaluation_score\": \"Your evaluation score for this checklist\"}},{{\"checklist_id\": \"the id of the checklist\",
\"reason\": \"The reason for your evaluation results\", \"evaluation_score\": \"Your evaluation score for this checklist\"}}]

There are total {num_checklist} checklists that you need to evaluate. The length of the output list is equal to the number of checklists and you should give an evaluation score for each checklist. You shoule be very very very strict to the evalution to further compare the responses from different models. Your response must be a valid Python List and should contain nothing else, as it will be directly executed in Python."""

subjective_reader_cfg = dict(
    input_columns=['instruction', 'formatted_checklists', 'num_checklist'],
    output_column='judgement',
    )

hellobench_categories = [
    'open_ended_qa',
    'summarization',
    'chat',
    'text_completion',
    'heuristic_text_generation',
]
data_path ='data/HelloBench'

hellobench_datasets = []

for category_name in hellobench_categories:
    subjective_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(
                        role='HUMAN',
                        prompt='{instruction}'
                    ),
                ]),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer, max_out_len=16384),
        )

    subjective_eval_cfg = dict(
        evaluator=dict(
            type=LMEvaluator,
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    begin=[
                        dict(
                            role='SYSTEM',
                            fallback_role='HUMAN',
                            prompt=system_prompt)
                    ],
                    round=[
                        dict(
                            role='HUMAN',
                            prompt = user_prompt
                        ),
                    ]),
            ),
            dict_postprocessor=dict(type=hellobench_postprocess,),
        ),
        pred_role='BOT',
    )

    hellobench_datasets.append(
        dict(
            abbr=f'HelloBench-{category_name}',
            type=HelloBenchDataset,
            path=data_path,
            category_name=category_name,
            reader_cfg=subjective_reader_cfg,
            infer_cfg=subjective_infer_cfg,
            eval_cfg=subjective_eval_cfg,
            mode='singlescore',
        ))
