from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator

gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

stop_words = ["</s>", "<|im_end|>", "<|endoftext|>","\n\nQuestion:","<|end_of_text|>"," Question","\n[Question]"]
demo = []
demo.append("Okay, let's see. Mark has a garden with flowers of three colors: yellow, purple, and green. The problem gives specific percentages to describe how many of each color there are relative to the others. Let me break this down step by step.\n\nFirst, it says there are ten yellow flowers. That's straightforward. So, yellow = 10.\n\nNext, there are 80% more purple flowers than yellow ones. Hmm, 80% more than yellow. So, I need to calculate 80% of the yellow flowers and then add that to the original number of yellow flowers to get the number of purple flowers. Let me write that out.\n\nIf yellow is 10, then 80% of 10 is... let's calculate that. 80% is the same as 0.8, so 0.8 * 10 = 8. So, there are 8 more purple flowers than yellow ones. Therefore, purple flowers = yellow + 8 = 10 + 8 = 18. Wait, is that right? Let me check again. If there are 80% more purple than yellow, does that mean purple = yellow + 80% of yellow? Yes, that makes sense. So, 10 + (0.8*10) = 10 + 8 = 18. So, purple flowers are 18. Got that.\n\nNow, moving on to green flowers. The problem states there are only 25% as many green flowers as there are yellow and purple flowers combined. Okay, so first I need to find the total number of yellow and purple flowers, then take 25% of that sum to get the number of green flowers.\n\nYellow and purple together: 10 (yellow) + 18 (purple) = 28. Then 25% of 28 is... 0.25 * 28. Let me calculate that. 0.25 * 28 is 7. So, green flowers = 7.\n\nNow, to find the total number of flowers in the garden, we just add up all the colors: yellow + purple + green = 10 + 18 + 7. Let me add those. 10 + 18 is 28, and 28 + 7 is 35. So, total flowers = 35.\n\nWait, but let me verify each step again to make sure I didn't make a mistake. It's easy to mix up percentages.\n\nStarting over:\n\n1. Yellow flowers: given as 10. That's clear.\n\n2. Purple flowers: 80% more than yellow. The term \"more than\" usually means you take the original amount and add the percentage. So, 80% of 10 is 8, so 10 + 8 = 18. That seems right.\n\n3. Green flowers: 25% as many as yellow and purple combined. So, total yellow and purple is 10 + 18 = 28. 25% of 28 is indeed 7. So, green = 7.\n\nAdding them all up: 10 + 18 + 7. Let's compute step by step. 10 + 18 is 28, then 28 + 7 is 35. Hmm, that checks out.\n\nBut let me just check if \"80% more\" is indeed the same as multiplying the original by 1 + 80%. Because sometimes percentages can be tricky. For example, if something is 80% more than X, it's X + 0.8X = 1.8X. So, in this case, 1.8 * 10 = 18. Yep, that's the same as before. So that's correct.\n\nSimilarly, \"25% as many\" means 0.25 times the quantity. So, 0.25*(10+18) = 0.25*28=7. That's correct too.\n\nSo, all the steps seem to add up. Therefore, the total number of flowers is 35. That seems reasonable.\n\nSo the answer is $\\boxed{35}$")

demo.append("Okay, let me figure out how many slices of pizza Albert eats in total. So, he bought 2 large pizzas and 2 small pizzas. Each large pizza has 16 slices, and each small one has 8 slices. Hmm, I need to calculate the total number of slices from both the large and small pizzas and then add them together.\n\nFirst, let's tackle the large pizzas. If one large pizza is 16 slices, then two large pizzas would be 16 times 2. Let me do that multiplication: 16 multiplied by 2. 16 times 2 is 32. So, the two large pizzas give 32 slices in total.\n\nNext, the small pizzas. Each small pizza has 8 slices, and he bought two of them. So, that's 8 slices per pizza multiplied by 2 pizzas. Calculating that: 8 times 2 is 16. Therefore, the two small pizzas add up to 16 slices.\n\nNow, to find the total number of slices Albert eats, I need to add the slices from the large and small pizzas together. That would be 32 slices from the large ones plus 16 slices from the small ones. Let's add those numbers: 32 plus 16. Hmm, 30 plus 10 is 40, and 2 plus 6 is 8, so 40 plus 8 is 48. Wait, no, that's not right. Let me check again. 32 plus 16: 30 plus 10 is 40, and 2 plus 6 is 8. So, 40 plus 8 is indeed 48.\n\nWait, but hold on a second. Let me verify that with another method to be sure. If I take 32 slices from the large pizzas and add 16 slices from the small ones, another way to look at it is 32 + 16. Breaking down 16 into 10 + 6, so 32 + 10 is 42, and then 42 + 6 is 48. Yep, that still checks out.\n\nSo, combining both the large and small pizzas, Albert eats a total of 48 slices in one day if he finishes all the pizza. Let me just cross-verify once more. 2 large pizzas: 2 x 16 = 32. 2 small pizzas: 2 x 8 = 16. 32 + 16 = 48. That seems correct. I don't think I made any mistakes here. The calculation is straightforward multiplication and addition.\n\nAlternatively, if I consider the number of pizzas first, he has 2 large and 2 small. Each large is double the size of a small one in terms of slices (16 vs. 8). So, each large pizza is equivalent to 2 small pizzas. Therefore, 2 large pizzas would be equivalent to 4 small pizzas. Adding the 2 actual small pizzas, that's a total of 6 small pizzas. Each small pizza is 8 slices, so 6 x 8 = 48. That also gives the same answer. So, this alternative method confirms the total is 48 slices.\n\nSo the answer is $\\boxed{48}$")

new_demo = []
for item in demo:
    words = item.split()
    new_demo.append(' '.join(['xxx' for i in range(len(words))]))
demo = new_demo

prompt = (
    "Consider the following two reasoning process examples:\n" 
    + "Example 1 \n" + demo[0] + '\n'
    + "Example 2 \n" + demo[1] + '\n\n'
    "Please refer to the reasoning approaches demonstrated above, solve the following question by yourself:\n"
    "{question}\n\n"
    "Finally, provide your answer in the following format: 'So the answer is $\\boxed{}$'."
)

gsm8k_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt=prompt),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=4096,stopping_criteria=stop_words)
)

gsm8k_eval_cfg = dict(
    evaluator=dict(type=Gsm8kEvaluator),
    pred_postprocessor=dict(type=gsm8k_postprocess),
    dataset_postprocessor=dict(type=gsm8k_dataset_postprocess),
)

gsm8k_datasets = [
    dict(
        abbr='gsm8k_slow_think_2shot_xxx',
        type=GSM8KDataset,
        path='opencompass/gsm8k',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg,
    )
]
