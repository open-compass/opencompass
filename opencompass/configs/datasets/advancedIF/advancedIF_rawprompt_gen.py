from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import AdvancedIFDataset
from opencompass.evaluator import GenericLLMEvaluator

advancedif_reader_cfg = dict(
    input_columns=['conversation_history', 'full_conversation',
                    'last_user_question', 'rubrics_text'],
    output_column='rubrics_text',
)

advancedif_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            {'expand_column': 'conversation_history'},
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

RUBRIC_JUDGE_TEMPLATE = """Your job is to assess if the AI's response to the user's most recent prompt correctly follows the user's instructions
The conversation history:
--------------------------------------------------------------
{full_conversation}
--------------------------------------------------------------
User's most recent prompt:
{last_user_question}
--------------------------------------------------------------
Here's the AI's response to the user's most recent prompt:
{prediction}
--------------------------------------------------------------
Here are the rubrics:
--------------------------------------------------------------
{rubrics_text}
--------------------------------------------------------------
Your response should be a JSON blob with the following schema:
{{
    "rubrics_check": {{
        "question_1": "answer to question 1 in the rubrics",
        "question_2": "answer to question 2 in the rubrics",
        ...
    }},
    "SATISFIED_ALL_REQUIREMENTS": "YES" if the AI's response passes the rubrics check. "NO" otherwise.
}}""".strip()

advancedif_eval_cfg = dict(
    evaluator=dict(
        type=GenericLLMEvaluator,
        prompt_template=dict(
            type=RawPromptTemplate,
            messages=[
                {'role': 'user', 'content': RUBRIC_JUDGE_TEMPLATE},
            ],
        ),
        dataset_cfg=dict(
            type=AdvancedIFDataset,
            path='facebook/AdvancedIF',
            reader_cfg=advancedif_reader_cfg,
        ),
        judge_cfg=dict(),
        dict_postprocessor=dict(type='advancedif_rubric_postprocess'),
    ),
)

advancedif_datasets = [
    dict(
        type=AdvancedIFDataset,
        abbr='advancedIF',
        path='facebook/AdvancedIF',
        reader_cfg=advancedif_reader_cfg,
        infer_cfg=advancedif_infer_cfg,
        eval_cfg=advancedif_eval_cfg,
        n=1,
    )
]
