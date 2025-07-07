from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator

gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

stop_words = ["</s>", "<|im_end|>", "<|endoftext|>","\n\nQuestion:","<|end_of_text|>"," Question","\n[Question]"]

round = [
{'role': 'HUMAN', 'prompt': "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
{'role': 'BOT', 'prompt': "Okay, let's see. The problem is about Natalia selling clips in April and May. So, in April, she sold clips to 48 of her friends. Then in May, she sold half as many clips. The question is asking how many clips she sold altogether in April and May. Hmm, let's break this down step by step.\n\nFirst, let's make sure I understand the problem correctly. In April, Natalia sold clips to 48 friends. Now, does that mean each friend bought one clip? Or did each friend buy multiple clips? The problem says she sold clips to 48 friends. Hmm, the wording here is a bit ambiguous. \"Sold clips to 48 friends\" could be interpreted in two ways: either she sold one clip per friend, totaling 48 clips, or she sold an unspecified number of clips to each of 48 friends, meaning the total number isn't specified. But the problem doesn't give us any more details about the number of clips per friend in April. Wait, but then it says she sold half as many clips in May. So maybe the 48 refers to the number of clips, not the number of friends? Wait, let me check the original problem again.\n\nOriginal problem: \"Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\"\n\nHmm, so it says she sold clips to 48 friends in April. So, 48 friends. Then in May, she sold half as many clips. So, \"half as many clips\" as in April. Wait, but in April, the problem mentions she sold to 48 friends, not necessarily 48 clips. So does \"half as many clips\" refer to half the number of clips or half the number of friends?\n\nThis is a bit confusing. Maybe we need to make an assumption here. Since the problem mentions \"clips\" in both months, but in April it specifies the number of friends she sold to, whereas in May it just says \"half as many clips.\" So perhaps the key here is that in April, she sold clips to 48 friends, but we don't know how many clips each friend bought. Then in May, she sold half as many clips as she did in April. Wait, but if we don't know how many clips she sold in April, how can we determine the number for May?\n\nWait, perhaps there's a misinterpretation here. Maybe the problem is intended to mean that she sold clips to 48 friends in April, and each friend bought one clip. So, 48 clips in April. Then in May, she sold half as many clips, so 24 clips. Then total would be 48 + 24 = 72 clips. But is that the correct interpretation?\n\nAlternatively, maybe \"sold clips to 48 friends\" means that each transaction was with a friend, but each friend could have bought multiple clips. But since the problem doesn't specify the number of clips per friend, maybe we have to assume that each friend bought one clip. That would be a standard assumption in such problems unless stated otherwise. So, in April, 48 clips, one per friend. Then in May, half as many clips, so 24. Total 72. That seems straightforward.\n\nBut let me check again. If it's 48 friends in April, and in May, half as many clips. If \"half as many clips\" refers to half the number of clips sold in April, then if we don't know the number of clips in April, we can't compute May's number. Therefore, the problem must mean that the number of clips sold in April is 48, which is sold to 48 friends (each buying one clip). Then May is half of that, so 24. Then total is 72. That makes sense.\n\nAlternatively, if \"sold clips to 48 friends\" in April, but each friend bought multiple clips, but since the problem mentions \"how many clips did Natalia sell altogether,\" and the answer would require knowing the number of clips. But since the problem doesn't specify the number of clips per friend, we have to assume that each friend bought one clip. Otherwise, the problem is unsolvable as we don't have enough information.\n\nTherefore, the problem is expecting us to interpret \"sold clips to 48 friends\" as 48 clips, one per friend, in April. Then in May, she sold half as many clips, so 24. Total is 48 + 24 = 72. So the answer is 72.\n\nWait, but let me verify again. The problem says: \"Natalia sold clips to 48 of her friends in April.\" So, she sold clips to 48 friends. So, the number of friends is 48, but the number of clips is not specified. Then in May, she sold half as many clips. If \"half as many clips\" as in April, then we need the number of clips sold in April. But we don't know that because the problem only mentions the number of friends. So unless each friend bought one clip, we can't get the number of clips.\n\nBut if we don't assume that, the problem is unsolvable. Therefore, the problem must be intending that \"sold clips to 48 friends\" means 48 clips, with each friend buying one. That's a common phrasing in math problems, where \"sold to x friends\" implies x units if the item is something typically bought one per person, like maybe a hair clip. So, perhaps each friend bought one clip.\n\nAlternatively, maybe in April, she sold 48 clips in total, regardless of the number of friends. Wait, the problem says \"sold clips to 48 of her friends in April.\" So the number of friends is 48. So it's possible that she sold multiple clips to each friend, but again, without additional information, we can't compute the total clips. Therefore, in a math problem context, unless specified otherwise, \"sold to x friends\" would usually imply that each friend bought one item, especially if the item is something like a clip which you might buy one of. So 48 friends, 48 clips. Then May is half of April's clips, so 24. Total 72. So I think that's the answer they are expecting.\n\nAlternatively, maybe \"half as many clips in May\" refers to half the number of transactions or half the number of friends. But since it says \"half as many clips,\" not \"half as many friends,\" so the number of clips is halved. So if in April, she sold 48 clips (to 48 friends, one each), then May is 24 clips. So total 72.\n\nSo the answer is \\boxed{72}."},
{'role': 'HUMAN', 'prompt': "Question: {question}\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
]

gsm8k_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=round)),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=4096, stopping_criteria=stop_words))


gsm8k_eval_cfg = dict(
    evaluator=dict(type=Gsm8kEvaluator),
    pred_postprocessor=dict(type=gsm8k_postprocess),
    dataset_postprocessor=dict(type=gsm8k_dataset_postprocess),
)

gsm8k_datasets = [
    dict(
        abbr='gsm8k_r1_enhance_1shot',
        type=GSM8KDataset,
        path='opencompass/gsm8k',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg,
    )
]
