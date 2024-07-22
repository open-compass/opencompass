from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.datasets import ArenaHardDataset
from opencompass.summarizers import ArenaHardSummarizer
from mmengine.config import read_base

subjective_reader_cfg = dict(
    input_columns=['question'],
    output_column='judge',
    )

subjective_all_sets = [
    'arenahard',
]


arenahard_datasets = []

system_prompt = "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.\n\nBegin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.\n\nWhen evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.\n\nThen consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.\n\nThen consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.\n\nAfter providing your explanation, you must output only one of the following choices as your final verdict with a label:\n\n1. Assistant A is significantly better: [[A>>B]]\n2. Assistant A is slightly better: [[A>B]]\n3. Tie, relatively the same: [[A=B]]\n4. Assistant B is slightly better: [[B>A]]\n5. Assistant B is significantly better: [[B>>A]]\n\nExample output: \"My final verdict is tie: [[A=B]]\"."

judge_prompt = "<|User Prompt|>\n{question}\n\n<|The Start of Assistant A's Answer|>\n{prediction}\n<|The End of Assistant A's Answer|>\n\n<|The Start of Assistant B's Answer|>\n{prediction2}\n<|The End of Assistant B's Answer|>"

gpt4 = [dict(
    abbr='gpt4-0314',
)]

for _name in subjective_all_sets:
    subjective_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(
                        role='HUMAN',
                        prompt='{question}'
                    ),
                ]),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer, max_out_len=4096),
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
                        prompt = judge_prompt
                    ),
                ]),
            ),
        ),
        pred_role='BOT',
    )

    arenahard_datasets.append(
        dict(
            abbr='arenahard',
            type=ArenaHardDataset,
            path='./data/subjective/arena_hard',
            name=_name,
            reader_cfg=subjective_reader_cfg,
            infer_cfg=subjective_infer_cfg,
            eval_cfg=subjective_eval_cfg,
            mode='m2n',
            infer_order='double',
            base_models=gpt4,
            summarizer = dict(type=ArenaHardSummarizer),
            given_pred = [{'abbr':'gpt4-0314', 'path':'./data/subjective/arena_hard'}]
        ))
