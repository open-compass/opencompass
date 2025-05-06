from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.evaluator import GenericLLMEvaluator
from opencompass.datasets import generic_llmjudge_postprocess
from opencompass.datasets import PromptCBLUEDataset

PromptCBLUE_lifescience_sets = [
    'CHIP-CDN', 'CHIP-CTC', 'KUAKE-QIC', 'IMCS-V2-DAC',
    'CHIP-STS', 'KUAKE-QQR', 'KUAKE-IR', 'KUAKE-QTR'
]
# Query template (keep original)
QUERY_TEMPLATE = """
Given a medical diagnosis description and labeled ICD-10 candidate terms below, select the matching normalized term(s).
Original diagnosis: {input}

Options:
{options_str}

The last line of your response must be exactly in the format:
ANSWER: <LETTER(S)>
""".strip()

# Grader template (keep original)
GRADER_TEMPLATE = """
As an expert evaluator, judge whether the candidate's answer matches the gold standard below.
Return 'A' for CORRECT or 'B' for INCORRECT, with no additional text.

Original diagnosis: {input}

Options:
{options_str}

Gold answer: {target}

Candidate answer: {prediction}
""".strip()

# Common reader config
reader_cfg = dict(
    input_columns=['input', 'answer_choices', 'options_str'],
    output_column='target',
    train_split='dev'
)

# Assemble LLM evaluation datasets
promptcblue_llm_datasets = []
for name in PromptCBLUE_lifescience_sets:
    infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(round=[
                dict(role='HUMAN', prompt=QUERY_TEMPLATE),
            ]),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )

    eval_cfg = dict(
        evaluator=dict(
            type=GenericLLMEvaluator,
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    begin=[
                        dict(
                            role='SYSTEM',
                            fallback_role='HUMAN',
                            prompt='You are an expert judge for medical term normalization tasks.',
                        )
                    ],
                    round=[
                        dict(role='HUMAN', prompt=GRADER_TEMPLATE),
                    ],
                )
            ),
            dataset_cfg=dict(
                type=PromptCBLUEDataset,
                path='/fs-computility/ai4sData/shared/lifescience/tangcheng/LifeScience/opencompass_val/datasets/PromptCBLUE',
                name=name,
                reader_cfg=reader_cfg,
            ),
            judge_cfg=dict(),
            dict_postprocessor=dict(type=generic_llmjudge_postprocess),
        ),
        pred_role='BOT',
    )

    promptcblue_llm_datasets.append(
        dict(
            abbr=f"promptcblue_{name.lower().replace('-', '_')}_norm_llm",
            type=PromptCBLUEDataset,
            path='/fs-computility/ai4sData/shared/lifescience/tangcheng/LifeScience/opencompass_val/datasets/PromptCBLUE',
            name=name,
            reader_cfg=reader_cfg,
            infer_cfg=infer_cfg,
            eval_cfg=eval_cfg,
            mode='singlescore',
        )
    )
