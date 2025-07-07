from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator

gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

stop_words = ["</s>", "<|im_end|>", "<|endoftext|>",
              "\n\nQuestion:", "<|end_of_text|>", " Question", "\n[Question]"]
round = [
    {"role": "HUMAN", "prompt": "Question: xxx xxx xxx xxx xxx. xxx xxx xxx xxx xxx xxx xxx xxx. xxx xxx xxx xxx, xxx xxx xxx xxx xxx xxx. xxx xxx xxx xxx xxx xxx xxx xxx xxx?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
    {"role": "BOT", "prompt": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. So the answer is $\\boxed{6}$.\n"},
    {"role": "HUMAN", "prompt": "Question: xxx xxx xxx xxx xxx xxx xxx xxx, xxx xxx xxx xxx, xxx xxx xxx xxx xxx?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
    {"role": "BOT", "prompt": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. So the answer is $\\boxed{5}$.\n"},
    {"role": "HUMAN", "prompt": "Question: xxx xxx xxx xxx xxx xxx xxx xxx xxx xxx xxx xxx. xxx xxx xxx xxx xxx xxx xxx xxx. xxx xxx xxx xxx xxx xxx, xxx xxx xxx xxx xxx xxx?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
    {"role": "BOT", "prompt": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. So the answer is $\\boxed{39}$.\n"},
    {"role": "HUMAN", "prompt": "Question: xxx xxx xxx xxx xxx xxx. xxx xxx xxx xxx xxx xxx xxx xxx. xxx xxx xxx xxx xxx xxx xxx xxx xxx?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
    {"role": "BOT", "prompt": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. So the answer is $\\boxed{8}$.\n"},
    {"role": "HUMAN", "prompt": "Question: xxx xxx xxx xxx xxx xxx. xxx xxx xxx xxx xxx xxx xxx xxx xxx xxx xxx xxx. xxx xxx xxx xxx xxx?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
    {"role": "BOT", "prompt": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. So the answer is $\\boxed{9}$.\n"},
    {"role": "HUMAN", "prompt": "Question: xxx xxx xxx xxx xxx xxx xxx xxx xxx. xxx xxx xxx xxx xxx xxx xxx xxx xxx, xxx xxx xxx xxx xxx. xxx xxx xxx xxx xxx xxx?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
    {"role": "BOT", "prompt": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. So the answer is $\\boxed{29}$.\n"},
    {"role": "HUMAN", "prompt": "Question: xxx xxx xxx xxx xxx xxx xxx xxx. xxx xxx, xxx xxx xxx xxx xxx xxx xxx xxx. xxx xxx, xxx xxx xxx xxx xxx. xxx xxx xxx xxx xxx xxx xxx xxx xxx?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
    {"role": "BOT", "prompt": "Michael started with 58 golf balls. After losing 23 on Tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. So the answer is $\\boxed{33}$.\n"},
    {"role": "HUMAN", "prompt": "Question: xxx xxx xxx xxx xxx xxx. xxx xxx xxx xxx xxx xxx xxx xxx xxx xxx. xxx xxx xxx xxx xxx xxx xxx xxx?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
    {"role": "BOT", "prompt": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. So the answer is $\\boxed{8}$.\n"},
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
        abbr='gsm8k_original_xxx_v1',
        type=GSM8KDataset,
        path='opencompass/gsm8k',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg)
]
