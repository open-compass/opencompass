from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator

gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

stop_words = ["</s>", "<|im_end|>", "<|endoftext|>",
              "\n\nQuestion:", "<|end_of_text|>", " Question", "\n[Question]"]
round = [
    {"role": "HUMAN", "prompt": "Question: xxx xxx xxx xxx xxx. xxx xxx xxx xxx xxx xxx xxx xxx. xxx xxx xxx xxx, xxx xxx xxx xxx xxx xxx. xxx xxx xxx xxx xxx xxx xxx xxx xxx?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
    {"role": "BOT", "prompt": "xxx xxx xxx xxx xxx. xxx xxx xxx xxx xxx xxx xxx xxx xxx. xxx xxx xxx xxx xxx xxx xxx xxx xxx. So the answer is $\\boxed{6}$.\n"},
    {"role": "HUMAN", "prompt": "Question: xxx xxx xxx xxx xxx xxx xxx xxx, xxx xxx xxx xxx, xxx xxx xxx xxx xxx?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
    {"role": "BOT", "prompt": "xxx xxx xxx xxx xxx. xxx xxx xxx xxx. xxx xxx xxx xxx. So the answer is $\\boxed{5}$.\n"},
    {"role": "HUMAN", "prompt": "Question: xxx xxx xxx xxx xxx xxx xxx xxx xxx xxx xxx xxx. xxx xxx xxx xxx xxx xxx xxx xxx. xxx xxx xxx xxx xxx xxx, xxx xxx xxx xxx xxx xxx?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
    {"role": "BOT", "prompt": "xxx xxx xxx xxx xxx xxx. xxx xxx xxx xxx xxx. xxx xxx xxx xxx xxx xxx xxx xxx. xxx xxx xxx xxx xxx xxx xxx xxx xxx. So the answer is $\\boxed{39}$.\n"},
    {"role": "HUMAN", "prompt": "Question: xxx xxx xxx xxx xxx xxx. xxx xxx xxx xxx xxx xxx xxx xxx. xxx xxx xxx xxx xxx xxx xxx xxx xxx?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
    {"role": "BOT", "prompt": "xxx xxx xxx xxx xxx xxx. xxx xxx xxx xxx xxx xxx xxx xxx. xxx xxx xxx xxx xxx xxx xxx xxx. So the answer is $\\boxed{8}$.\n"},
    {"role": "HUMAN", "prompt": "Question: xxx xxx xxx xxx xxx xxx. xxx xxx xxx xxx xxx xxx xxx xxx xxx xxx xxx xxx. xxx xxx xxx xxx xxx?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
    {"role": "BOT", "prompt": "xxx xxx xxx xxx xxx xxx. xxx xxx xxx xxx xxx xxx xxx xxx xxx. xxx xxx xxx xxx xxx. So the answer is $\\boxed{9}$.\n"},
    {"role": "HUMAN", "prompt": "Question: xxx xxx xxx xxx xxx xxx xxx xxx xxx. xxx xxx xxx xxx xxx xxx xxx xxx xxx, xxx xxx xxx xxx xxx. xxx xxx xxx xxx xxx xxx?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
    {"role": "BOT", "prompt": "xxx xxx xxx xxx xxx xxx. xxx xxx xxx xxx xxx xxx xxx xxx xxx. xxx xxx xxx xxx xxx xxx xxx xxx. So the answer is $\\boxed{29}$.\n"},
    {"role": "HUMAN", "prompt": "Question: xxx xxx xxx xxx xxx xxx xxx xxx. xxx xxx, xxx xxx xxx xxx xxx xxx xxx xxx. xxx xxx, xxx xxx xxx xxx xxx. xxx xxx xxx xxx xxx xxx xxx xxx xxx?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
    {"role": "BOT", "prompt": "xxx xxx xxx xxx xxx xxx xxx. xxx xxx xxx xxx xxx xxx xxx xxx. xxx xxx xxx xxx xxx xxx xxx xxx. So the answer is $\\boxed{33}$.\n"},
    {"role": "HUMAN", "prompt": "Question: xxx xxx xxx xxx xxx xxx. xxx xxx xxx xxx xxx xxx xxx xxx xxx xxx. xxx xxx xxx xxx xxx xxx xxx xxx?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
    {"role": "BOT", "prompt": "xxx xxx xxx xxx xxx xxx. xxx xxx xxx xxx xxx xxx xxx xxx. xxx xxx xxx xxx xxx xxx xxx xxx. So the answer is $\\boxed{8}$.\n"},
    {'role': 'HUMAN', 'prompt': "Question: {question}\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
]

gsm8k_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=round)),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=4096, stopping_criteria=stop_words))

gsm8k_eval_cfg = dict(evaluator=dict(type=Gsm8kEvaluator),
                      pred_postprocessor=dict(type=gsm8k_postprocess),
                      dataset_postprocessor=dict(type=gsm8k_dataset_postprocess))

gsm8k_datasets = [
    dict(
        abbr='gsm8k_original_xxx_v2',
        type=GSM8KDataset,
        path='opencompass/gsm8k',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg)
]
