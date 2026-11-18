from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import RPCBenchDataset, RPCBenchLLMEvaluator

# RPC-Bench text-only track, 8 open-ended categories (2262 questions on test).
# Inference and judge prompts are copied verbatim from the official repo
# (llm.py / prompt.py): https://github.com/zai-org/RPC-Bench
# Only placeholders are renamed: {content} -> {paper_content} (see
# RPCBenchDataset), and {reference_answer} / {predicted_answer} of the judge
# user prompt -> {answer} / {prediction}.

rpcbench_open_reader_cfg = dict(
    input_columns=['question', 'paper_content'],
    output_column='answer',
)

RPCBENCH_OPEN_INFER_TEMPLATE = """You are an expert academic assistant. Your task is to carefully read and analyze the provided complete research paper, and then answer the following question solely based on its content, arguments, and data, without using any external information or assumptions.
Response Requirements:
1. The answer must be professional, precise, concise, and clearly presented.
2. All statements in your answer must be exclusively derived from the paper's content and directly relevant to the question, avoiding any information or claims not supported by the paper.
3. The total length of your response must not exceed 3000 characters (including spaces).

Question:
{question}

Paper:
{paper_content}"""

rpcbench_open_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            {'role': 'user', 'content': RPCBENCH_OPEN_INFER_TEMPLATE},
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

# Kept as explicit concatenation: the official prompt has a trailing space
# after "system prompt." that a triple-quoted string would lose to the
# trailing-whitespace pre-commit hook.
RPCBENCH_JUDGE_USER_TEMPLATE = (
    'Evaluate and rate the quality of the following predicted answer to an '
    'academic question according to the evaluation characteristics given in '
    'the system prompt. \n\n'
    '<paper-title>{title}</paper-title>\n\n'
    '<paper-abstract>{abstract}</paper-abstract>\n\n'
    '<question>{question}</question>\n\n'
    '<reference-answer>{answer}</reference-answer>\n\n'
    '<predicted-answer>{prediction}</predicted-answer>\n\n'
)

RPCBENCH_JUDGE_CONCISENESS_SYSTEM = """<Context>
Academic question answering is the process of thoroughly reading and analyzing a scientific paper in order to generate answers to specific questions based solely on the paper’s content, arguments, and data. Unlike open-domain or general question answering, which may draw on external sources or background knowledge, academic QA is strictly limited to information contained within the source paper itself. This task demands not only accurate extraction of factual information, but also the interpretation of experimental results, logical reasoning, and careful understanding of nuanced arguments as presented by the authors. Answers in this context must faithfully and objectively reflect the ideas, evidence, and intentions of the original work, ensuring that each response is both accurate and limited to what is substantiated by the source material—without introducing personal opinions, assumptions, or information from outside the given paper.
</Context>

<Role>
You are an expert academic answer evaluator.
</Role>

<Task-Description>
The task is to evaluate the quality of a predicted answer to a given academic question. You will be provided with the following information: (1) the title of the research paper, (2) the abstract of the research paper, (3) a specific academic question about the paper, (4) a gold-standard reference answer (golden answer) generated strictly from the paper, and (5) a predicted answer to the same question, which you are to evaluate. The general objective is to determine whether the predicted answer addresses the question with accuracy, completeness, and fidelity, as exemplified by the golden answer. Please base your assessment on the evaluation characteristics listed below.
</Task-Description>

<Evaluation-Characteristics>
1. Conciseness: Evaluate whether the predicted answer is brief and to the point, avoiding unnecessary repetition or irrelevant information. The answer should deliver key content clearly, without excessive length or verbosity.
</Evaluation-Characteristics>

<Rating-Scale>
For each evaluation characteristic, assign a quality score between 0.00 (very bad) and 5.00 (very good), using decimal values precise to two decimal places (e.g., 3.73) for fine-grained assessment. Follow the guidelines specified below for each rating per evaluation characteristic.

1. Conciseness
0.00–1.00 (Very bad): The predicted answer is verbose or contains substantial irrelevant/redundant information, making it unclear or unfocused.
1.01–2.00 (Bad): The predicted answer includes some redundancy or unnecessary details, affecting clarity.
2.01–3.00 (Moderate): The predicted answer is generally clear but could benefit from further condensation to remove several minor redundancies.
3.01–4.00 (Good): The predicted answer is concise, with only minimal unnecessary information.
4.01–5.00 (Very good): The predicted answer is exceptionally concise, presenting essential information directly and clearly with no redundancy.
</Rating-Scale>

<Response-Format>
For each characteristic, rate the quality with a decimal score between 0.00 (very bad) and 5.00 (very good), precise to two decimal places (e.g., 4.21). Provide a short rationale for each rating.
Return your response in JSON format: {characteristic : {"rating": "", "rationale": ""}}

<Example-Response>
{
  "Conciseness": {
    "rating": "4.15",
    "rationale": "The answer is generally concise and focused, with only minimal redundant information."
  }
}
</Example-Response>
</Response-Format>

<Note>
Base your evaluation solely on the paper title, abstract, question, golden answer, and predicted answer provided. Do NOT use any outside knowledge or make assumptions about the paper's content beyond what is implied or demonstrated by the golden answer. Be objective and provide clear, reasoned justification for your rating.
</Note>"""

RPCBENCH_JUDGE_COMPLETENESS_SYSTEM = """<Context>
Academic question answering is the process of thoroughly reading and analyzing a scientific paper in order to generate answers to specific questions based solely on the paper’s content, arguments, and data. Unlike open-domain or general question answering, which may draw on external sources or background knowledge, academic QA is strictly limited to information contained within the source paper itself. This task demands not only accurate extraction of factual information, but also the interpretation of experimental results, logical reasoning, and careful understanding of nuanced arguments as presented by the authors. Answers in this context must faithfully and objectively reflect the ideas, evidence, and intentions of the original work, ensuring that each response is both accurate and limited to what is substantiated by the source material—without introducing personal opinions, assumptions, or information from outside the given paper.
</Context>

<Role>
You are an expert academic answer evaluator.
</Role>

<Task-Description>
The task is to evaluate the quality of a predicted answer to a given academic question. You will be provided with the following information: (1) the title of the research paper, (2) the abstract of the research paper, (3) a specific academic question about the paper, (4) a gold-standard reference answer (golden answer) generated strictly from the paper, and (5) a predicted answer to the same question, which you are to evaluate. The general objective is to determine whether the predicted answer addresses the question with accuracy, completeness, and fidelity, as exemplified by the golden answer. Please base your assessment on the evaluation characteristics listed below.
</Task-Description>

<Evaluation-Characteristics>
1. Completeness: Assess the proportion of information in the predicted answer that overlaps with the reference answer. This is analogous to recall—consider whether the predicted answer adequately covers all major points and details provided by the reference answer, and does not omit essential content.
</Evaluation-Characteristics>

<Rating-Scale>
For each evaluation characteristic, assign a quality score between 0.00 (very bad) and 5.00 (very good), using decimal values precise to two decimal places (e.g., 3.73) for fine-grained assessment. Follow the guidelines specified below for each rating per evaluation characteristic.

1. Completeness
0.00–1.00 (Very bad): The predicted answer fails to include most of the key content from the reference answer, omitting essential points or details.
1.01–2.00 (Bad): The predicted answer is missing several important aspects found in the reference answer.
2.01–3.00 (Moderate): The predicted answer includes a moderate portion of the relevant content from the reference answer but lacks full coverage.
3.01–4.00 (Good): Most relevant content from the reference answer is present, with only minor omissions.
4.01–5.00 (Very good): The predicted answer comprehensively incorporates all major information from the reference answer, leaving out nothing significant.
</Rating-Scale>

<Response-Format>
For each characteristic, rate the quality with a decimal score between 0.00 (very bad) and 5.00 (very good), precise to two decimal places (e.g., 4.21). Provide a short rationale for each rating.
Return your response in JSON format: {characteristic : {"rating": "", "rationale": ""}}

<Example-Response>
{
  "Completeness": {
    "rating": "3.52",
    "rationale": "The answer covers most of the key points from the reference answer, but omits a few minor details."
  }
}
</Example-Response>
</Response-Format>

<Note>
Base your evaluation solely on the paper title, abstract, question, golden answer, and predicted answer provided. Do NOT use any outside knowledge or make assumptions about the paper's content beyond what is implied or demonstrated by the golden answer. Be objective and provide clear, reasoned justification for your rating.
</Note>"""

RPCBENCH_JUDGE_CORRECTNESS_SYSTEM = """<Context>
Academic question answering is the process of thoroughly reading and analyzing a scientific paper in order to generate answers to specific questions based solely on the paper’s content, arguments, and data. Unlike open-domain or general question answering, which may draw on external sources or background knowledge, academic QA is strictly limited to information contained within the source paper itself. This task demands not only accurate extraction of factual information, but also the interpretation of experimental results, logical reasoning, and careful understanding of nuanced arguments as presented by the authors. Answers in this context must faithfully and objectively reflect the ideas, evidence, and intentions of the original work, ensuring that each response is both accurate and limited to what is substantiated by the source material—without introducing personal opinions, assumptions, or information from outside the given paper.
</Context>

<Role>
You are an expert academic answer evaluator.
</Role>

<Task-Description>
The task is to evaluate the quality of a predicted answer to a given academic question. You will be provided with the following information: (1) the title of the research paper, (2) the abstract of the research paper, (3) a specific academic question about the paper, (4) a gold-standard reference answer (golden answer) generated strictly from the paper, and (5) a predicted answer to the same question, which you are to evaluate. The general objective is to determine whether the predicted answer addresses the question with accuracy, completeness, and fidelity, as exemplified by the golden answer. Please base your assessment on the evaluation characteristics listed below.
</Task-Description>

<Evaluation-Characteristics>
1. Correctness: Assess the proportion of content from the reference answer that is accurately reflected in the predicted answer. This is analogous to precision—focus on the accuracy and fidelity of included information, ensuring no distortions or misrepresentations.
</Evaluation-Characteristics>

<Rating-Scale>
For each evaluation characteristic, assign a quality score between 0.00 (very bad) and 5.00 (very good), using decimal values precise to two decimal places (e.g., 3.73) for fine-grained assessment. Follow the guidelines specified below for each rating per evaluation characteristic.

1. Correctness
0.00–1.00 (Very bad): The predicted answer consistently misrepresents or distorts the content of the reference answer, with substantial factual errors.
1.01–2.00 (Bad): The predicted answer contains multiple inaccuracies or significant misinterpretations relative to the reference answer.
2.01–3.00 (Moderate): The predicted answer accurately includes some content from the reference answer but may also have minor misstatements or factual inaccuracies.
3.01–4.00 (Good): Most content from the reference answer is accurately represented in the predicted answer, with only rare errors.
4.01–5.00 (Very good): Virtually all content from the reference answer present in the predicted answer is accurate and faithful, with no factual errors or distortions.
</Rating-Scale>

<Response-Format>
For each characteristic, rate the quality with a decimal score between 0.00 (very bad) and 5.00 (very good), precise to two decimal places (e.g., 4.21). Provide a short rationale for each rating.
Return your response in JSON format: {characteristic : {"rating": "", "rationale": ""}}

<Example-Response>
{
  "Correctness": {
    "rating": "4.03",
    "rationale": "Most of the information in the answer accurately reflects the reference answer, with only minor factual inaccuracies."
  }
}
</Example-Response>
</Response-Format>

<Note>
Base your evaluation solely on the paper title, abstract, question, golden answer, and predicted answer provided. Do NOT use any outside knowledge or make assumptions about the paper's content beyond what is implied or demonstrated by the golden answer. Be objective and provide clear, reasoned justification for your rating.
</Note>"""


def _judge_template(system_prompt):
    return dict(
        type=RawPromptTemplate,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': RPCBENCH_JUDGE_USER_TEMPLATE},
        ],
    )


rpcbench_open_eval_cfg = dict(
    evaluator=dict(
        type=RPCBenchLLMEvaluator,
        prompt_templates=dict(
            Conciseness=_judge_template(RPCBENCH_JUDGE_CONCISENESS_SYSTEM),
            Completeness=_judge_template(RPCBENCH_JUDGE_COMPLETENESS_SYSTEM),
            Correctness=_judge_template(RPCBENCH_JUDGE_CORRECTNESS_SYSTEM),
        ),
        dataset_cfg=dict(
            type=RPCBenchDataset,
            path='zai-org/RPC-Bench',
            split='test',
            subset='open',
            reader_cfg=rpcbench_open_reader_cfg,
        ),
        # A single judge config, or dict(judgers=[judge_a, judge_b]) to
        # average two judges as the official GPT-5 + Gemini setup.
        judge_cfg=dict(),
    ),
)

rpcbench_open_datasets = [
    dict(
        type=RPCBenchDataset,
        abbr='rpcbench_test_open',
        path='zai-org/RPC-Bench',
        split='test',
        subset='open',
        reader_cfg=rpcbench_open_reader_cfg,
        infer_cfg=rpcbench_open_infer_cfg,
        eval_cfg=rpcbench_open_eval_cfg,
    )
]
