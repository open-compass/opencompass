from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator
gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

stop_words = ["</s>", "<|im_end|>", "<|endoftext|>","\n\nQuestion:","<|end_of_text|>"," Question","\n[Question]"]
round = [
{'role': 'HUMAN', 'prompt': "Question: Tree Elementary School is raising money for a new playground. Mrs. Johnson’s class raised $2300, which is twice the amount that Mrs. Sutton’s class raised. Mrs. Sutton’s class raised 8 times less than Miss Rollin’s class. Miss Rollin’s class raised a third of the total amount raised by the school. How much money did the school raise for the playground if 2% will be deducted for administration fees?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
{'role': 'BOT', 'prompt': 'Find the amount Mrs. Sutton’s class raised by dividing $2300 by 2. $2300/2 = $1150\nFind the amount Miss Rollin’s class raised by multiplying $1150 by 8. $1150 x 8 = $9200\nMultiply $9200 by 3 to find the total amount raised. $9200 x 3 = $27600\nConvert 2% to decimal. 2/100 = 0.02\nMultiply $27600 by 0.02 to find the administration fees. $27600 x 0.02 = $552\nSubtract the administration fee from the total amount raised. $27600 - $552 = $27048 So the answer is $\\boxed{27048}$.\n'},
{'role': 'HUMAN', 'prompt': "Question: Four adults with 32 teeth went to the dentist for a checkup after realizing they were having severe tooth pain. They were found to have different numbers of damaged teeth, and each person had some teeth removed. The first person had 1/4 of all his teeth removed, and the second person had 3/8 of his teeth removed, the third person had half of his teeth removed, while the last person only had 4 teeth removed. What's the total number of teeth removed at the dental clinic?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
{'role': 'BOT', 'prompt': 'The first person had 1/4 of all his teeth removed, a total of 1/4*32 = 8 teeth.\nThe second person had 3/8 of his teeth removed, meaning 3/8*32 = 12 of his teeth were removed.\nTogether, the first and the second person had 12+8 = 20 teeth removed in total.\nThe third person had half of his teeth removed, which were 1/2*32 = 16 teeth.\nThe first three people had a total of 20+16 = 36 teeth removed from their mouth.\nThe last person only had 4 teeth removed, bringing the total number of teeth removed at the dentist clinic to be 36+4 = 40 teeth. So the answer is $\\boxed{40}$.\n'},
{'role': 'HUMAN', 'prompt': "Question: {question}\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
]
# que = round[-1]
# round = [round[4].copy(), round[5].copy()] * 8
# round.append(que)
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
        abbr='gsm8k_hard_2shot',
        type=GSM8KDataset,
        path='opencompass/gsm8k',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg)
]
