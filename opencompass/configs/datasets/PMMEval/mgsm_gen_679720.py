from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.PMMEval import PMMEvalMGSMDataset, PMMEvalMGSMEvaluator

NATURAL_LANGUAGE_CODES = ['en', 'zh', 'ar', 'es', 'fr', 'ja', 'ko', 'pt', 'th', 'vi']

LANG_TO_INSTRUCTIONS = {
    'en': "Solve this math problem. Give the reasoning steps before giving the final answer on the last line by itself in the format of \"The answer is \". Do not add anything other than the integer answer after \"The answer is \".\n\n{question}",
    'es': "Solve this math problem. Give the reasoning steps before giving the final answer on the last line by itself in the format of \"La respuesta es \". Do not add anything other than the integer answer after \"La respuesta es \".\n\n{question}",
    'fr': "Solve this math problem. Give the reasoning steps before giving the final answer on the last line by itself in the format of \"La réponse est \". Do not add anything other than the integer answer after \"La réponse est \".\n\n{question}",
    'zh': "Solve this math problem. Give the reasoning steps before giving the final answer on the last line by itself in the format of \"答案是 \". Do not add anything other than the integer answer after \"答案是 \".\n\n{question}",
    'ja': "Solve this math problem. Give the reasoning steps before giving the final answer on the last line by itself in the format of \"答えは \". Do not add anything other than the integer answer after \"答えは \".\n\n{question}",
    'th': "Solve this math problem. Give the reasoning steps before giving the final answer on the last line by itself in the format of \"คำตอบคือ \". Do not add anything other than the integer answer after \"คำตอบคือ \".\n\n{question}",
    'ko': "Solve this math problem. Give the reasoning steps before giving the final answer on the last line by itself in the format of \"답변은 \". Do not add anything other than the integer answer after \"답변은 \".\n\n{question}",
    'pt': "Solve this math problem. Give the reasoning steps before giving the final answer on the last line by itself in the format of \"A resposta é \". Do not add anything other than the integer answer after \"A resposta é \".\n\n{question}",
    'vi': "Solve this math problem. Give the reasoning steps before giving the final answer on the last line by itself in the format of \"Câu trả lời là \". Do not add anything other than the integer answer after \"Câu trả lời là \".\n\n{question}",
    'ar': "Solve this math problem. Give the reasoning steps before giving the final answer on the last line by itself in the format of \"الجواب هو \". Do not add anything other than the integer answer after \"الجواب هو \".\n\n{question}"
}

PMMEval_MGSM_datasets = list()

# Add flores_200

PMMEval_MGSM_reader_cfg = dict(
    input_columns=['question'],
    output_column='answer',
    test_split='test'
)

PMMEval_MGSM_eval_cfg = dict(
    evaluator=dict(type=PMMEvalMGSMEvaluator),
    pred_role='BOT')


for lang_code in NATURAL_LANGUAGE_CODES:
    PMMEval_MGSM_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(
                        role='HUMAN',
                        prompt=LANG_TO_INSTRUCTIONS[lang_code]
                    )
                ]
            )
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )

    PMMEval_MGSM_datasets.append(
        dict(
            abbr=f'mgsm-{lang_code}',
            type=PMMEvalMGSMDataset,
            path='P-MMEval',
            lang=lang_code,
            reader_cfg=PMMEval_MGSM_reader_cfg,
            infer_cfg=PMMEval_MGSM_infer_cfg,
            eval_cfg=PMMEval_MGSM_eval_cfg)
    )
