# flake8: noqa: E501
"""RebuttalBench ``R2_test``, in-domain split (Tables 2 / 8 of the paper).

All 6,000 rows are evaluated.

RebuttalBench (ICLR 2026, https://arxiv.org/abs/2601.15715): one rebuttal per
reviewer comment, scored by an LLM judge on Attitude, Clarity, Persuasiveness
and Constructiveness (0-10 each) and aggregated as the official
``eval/ag_avg.py``. See ``RebuttalBenchEvaluator`` in
``opencompass/datasets/rebuttalbench.py``.

Inference uses the prompt the official ``ag_eval.py`` applies to every
baseline model of the paper (GPT-4.1, Qwen3-8B, ...). The paper's judge is
Rebuttal-RM (https://huggingface.co/RebuttalAgent/Rebuttal-RM); any other
``judge_cfg`` keeps the rubric but changes the absolute numbers.
"""
from opencompass.datasets import RebuttalBenchDataset, RebuttalBenchEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

# ---------------------------------------------------------------------------
# Prompts, verbatim from eval/agent_prompt.py of the official repo. Trailing
# spaces of the original are kept as ``\x20``.
# ---------------------------------------------------------------------------

# ``SYS5`` in eval/agent_prompt.py
REBUTTALBENCH_BASE_PROMPT = """You are an AI authors tasked with respond to review comments on your academic paper. Based on the following input variables, generate a thoughtful and professional response to the specific review comment.
Input Variables:
1.Review_content: Complete content of the review comments.
2.Paper_fragment: Paper fragment most relevant to the comment.\x20
3.Target_comment: The target comment that currently needs to be addressed

Response Requirements:

Address the reviewer’s concerns directly and respectfully.
According to the relevant paper fragment to support your response.
Clearly explain any changes made to the manuscript or provide justification if no changes were made.
Maintain a professional and constructive tone throughout the response.
Output Format: Your response should be a well-structured paragraph that clearly adress the target comment"""

# ``SYS4`` in eval/agent_prompt.py
REBUTTALBENCH_JUDGE_PROMPT = """ You are a seasoned academic reviewer and response optimization expert. Your task is to evaluate the quality of the response based on the review comments, paper fragments, and the authors' responses. Please strictly follow the requirements below, and output only the score and score explanation.

Input variables:

review_content: Complete content of the review comments. similar_paper_fragments: Best paper fragment most relevant to the comment. comment: Specific segment of the review comments. original_response: The authors' original response text to the comment.

Your task: Based on the input information, output only a JSON object containing the following two items:
Scoring Standard
Score Range: 0 - 10
0: Wholly Ineffective
1-2: Perfunctory
3-4: Unconvincing
5-6: Addresses Some Concerns
7-8: Exceptional
9-10: Outstanding
score: Four-dimensional score breakdown, ranging from 0-10, structured as follows:
Attitude: The tone and professionalism of the response.\x20
Clarity: The logic, structure, and focus of the response.
Persuasiveness: The effectiveness of the argumentation and evidence support.\x20
Constructiveness: The commitment to revisions and specific actions taken.

score_explanation: A brief explanation of each score, specifically citing key points from the response text that reflect the scores and any shortcomings.

Output requirements:

Only output the JSON object; do not include any other characters or explanations.
The scoring must be reasonable, and the score explanation must clearly reference the original text that reflects the score.\x20
All output must be in formal, polite academic English.
Your output must be strictly JSON formal which can be dirctly loaded by json.load()
Output format example:
{ "score": { "Attitude": <int>,
              "Clarity": <int>,\x20
              "Persuasiveness": <int>,
              "Constructiveness": <int> },\x20
  "score_explanation": <explanation for your given score>}"""

# ``R_input`` of ag_eval.py
REBUTTALBENCH_GEN_INPUT = ('The whole review content is\n'
                           '{review_content}\n'
                           'The target comment content is:\n'
                           '{comment_text}\n'
                           'The best paper fragment is: {paper_fragment}\n')

rebuttalbench_reader_cfg = dict(
    input_columns=['review_content', 'comment_text', 'paper_fragment'],
    output_column=None,
)

rebuttalbench_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            dict(role='system', content=REBUTTALBENCH_BASE_PROMPT),
            dict(role='user', content=REBUTTALBENCH_GEN_INPUT),
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

rebuttalbench_eval_cfg = dict(
    evaluator=dict(
        type=RebuttalBenchEvaluator,
        prompt_template=dict(
            type=RawPromptTemplate,
            messages=[
                dict(role='system', content=REBUTTALBENCH_JUDGE_PROMPT),
                # ``RW_input1`` of ag_eval.py, built by the evaluator
                dict(role='user', content='{judge_input}'),
            ],
        ),
        judge_cfg=dict(),
        # unparsable judge outputs are judged again, 5 calls at most
        max_attempts=5,
    ),
)

rebuttalbench_datasets = [
    dict(
        type=RebuttalBenchDataset,
        abbr='rebuttalbench_r2_test_base',
        path='opencompass/rebuttalbench',
        split='R2_test',
        reader_cfg=rebuttalbench_reader_cfg,
        infer_cfg=rebuttalbench_infer_cfg,
        eval_cfg=rebuttalbench_eval_cfg,
    )
]
