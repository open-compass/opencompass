from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator

gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

stop_words = ["</s>", "<|im_end|>", "<|endoftext|>","\n\nQuestion:","<|end_of_text|>"," Question","\n[Question]"]
round = [
{'role': 'HUMAN', 'prompt': "Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nThink step by step, but only keep a minimum draft for each thinking step, with 5 words at most. put your final answer within \\boxed{}.\nAnswer:"},
{'role': 'BOT', 'prompt': "Original:15. Final:21. 21-15=6. →\boxed{6}.\n"},
{'role': 'HUMAN', 'prompt': "Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nThink step by step, but only keep a minimum draft for each thinking step, with 5 words at most. put your final answer within \\boxed{}.\nAnswer:"},
{'role': 'BOT', 'prompt': "Start:3. Added:2. 3+2=5. →\boxed{5}.\n"},
{'role': 'HUMAN', 'prompt': "Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nThink step by step, but only keep a minimum draft for each thinking step, with 5 words at most. put your final answer within \\boxed{}.\nAnswer:"},
{'role': 'BOT', 'prompt': "Total:32+42=74. Ate:35. 74-35=39. →\boxed{39}.\n"},
{'role': 'HUMAN', 'prompt': "Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nThink step by step, but only keep a minimum draft for each thinking step, with 5 words at most. put your final answer within \\boxed{}.\nAnswer:"},
{'role': 'BOT', 'prompt': "Start:20. Now:12. 20-12=8. →\boxed{8}.\n"},
{'role': 'HUMAN', 'prompt': "Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\nThink step by step, but only keep a minimum draft for each thinking step, with 5 words at most. put your final answer within \\boxed{}.\nAnswer:"},
{'role': 'BOT', 'prompt': "Start:5. +2+2=4. 5+4=9. →\boxed{9}.\n"},
{'role': 'HUMAN', 'prompt': "Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nThink step by step, but only keep a minimum draft for each thinking step, with 5 words at most. put your final answer within \\boxed{}.\nAnswer:"},
{'role': 'BOT', 'prompt': "Days:4. Add:5×4=20. 9+20=29. →\boxed{29}.\n"},
{'role': 'HUMAN', 'prompt': "Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nThink step by step, but only keep a minimum draft for each thinking step, with 5 words at most. put your final answer within \\boxed{}.\nAnswer:"},
{'role': 'BOT', 'prompt': "58-23=35. 35-2=33. →\boxed{33}.\n"},
{'role': 'HUMAN', 'prompt': "Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\nThink step by step, but only keep a minimum draft for each thinking step, with 5 words at most. put your final answer within \\boxed{}.\nAnswer:"},
{'role': 'BOT', 'prompt': "Cost:5×3=15. 23-15=8. →\boxed{8}.\n"},
{'role': 'HUMAN', 'prompt': "Question: {question}\nThink step by step, but only keep a minimum draft for each thinking step, with 5 words at most. put your final answer within \\boxed{}.\nAnswer:"},
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
        abbr='gsm8k_dot',
        type=GSM8KDataset,
        path='opencompass/gsm8k',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg)
]
