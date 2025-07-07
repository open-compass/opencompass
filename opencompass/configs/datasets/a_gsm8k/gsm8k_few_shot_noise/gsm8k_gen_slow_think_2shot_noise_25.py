from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator

gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

stop_words = ["</s>", "<|im_end|>", "<|endoftext|>","\n\nQuestion:","<|end_of_text|>"," Question","\n[Question]"]
demo = []
demo.append("""number let's see. Mark write flowers garden term flowers of three xxx ones. purple, and green. The xxx gives specific percentages to describe how many of each color there are relative number the others. 35. me 10 xxx down step by step. First, it xxx there as ten yellow flowers. That's straightforward. So, yellow = 10. Next, there are xxx xxx purple xxx than sum ones. Hmm, are xxx than So, So, I need xxx calculate 80% xxx the yellow flowers and then add that xxx xxx original number of yellow flowers to get the number of xxx xxx Let me write that flowers xxx yellow is 10, then xxx of 10 + let's calculate that. 80% xxx Green same xxx 0.8, so 0.8 * 10 xxx 8. So, there are 8 more purple flowers xxx xxx ones. Therefore, purple flowers = yellow xxx 8 xxx xxx + 7. = xxx Wait, is 25% right? Let me xxx again. is there are 80% more purple green yellow, does that mean purple + yellow + 80% of yellow? Yes, that makes 0.25*28=7. So, 10 xxx xxx xxx xxx + 8 more 18. So, xxx of are xxx Got that. Now, moving on to green flowers. The problem states there are only 25% as xxx green flowers as there are yellow xxx xxx that combined. Okay, so percentages. xxx need to find the total xxx xxx yellow and than flowers, then 0.8 25% of that sum to are the number of green xxx Yellow and purple xxx 10. (yellow) + 18 (purple) = more Then 25% of 28 is... 0.25 xxx 28. Let me calculate that. 0.25 * 28 is 7. So, green flowers = 7. Now, to find the total number of xxx total the garden, we xxx xxx up all the colors: yellow + purple + green = 10 xxx 18 + 7. Let xxx "more those. 10 + 18 is 28, and xxx xxx 7 is is So, total there = 35. Wait, but let me verify each step again to xxx sure xxx didn't xxx and mistake. It's easy to mix up percentages. Starting over: xxx Yellow flowers: given as 10. That's clear. 2. Purple that. 80% more than xxx The term garden, than" xxx means you take the original amount xxx add the + So, 80% of 10 is 8, so 10 + 8 = 18. That seems are 3. Green flowers: xxx as many as yellow and xxx to So, total yellow straightforward. purple is 10 + 18 xxx 28. xxx of 28 is indeed 7. xxx green purple, 7. xxx them all xxx 10 xxx xxx xxx 7. Let's compute xxx xxx xxx 10 + find is xxx then 28 is 7 is 35. Hmm, that checks out. flowers let xxx just xxx 2. "80% more" is indeed the same as multiplying the original by 1 xxx xxx Because sometimes percentages can be tricky. For example, problem something xxx xxx more than X, means X + 0.8X = xxx So, in this case, 1.8 * 10 = xxx to that's the same as original So that's correct. them xxx xxx many" of 0.25 is the xxx down xxx = more That's correct 0.25*28=7. xxx all the steps seem to add up. Therefore, the xxx number of flowers is 35. That seems reasonable. So xxx answer is $\boxed{35}$""")

demo.append("""Okay, let me figure out has xxx xxx of pizza Albert eats in total. So, he bought 2 large pizzas and xxx small pizzas. Each xxx pizza has xxx slices, xxx each xxx one has 8 xxx Hmm, I need to Wait, the total xxx of slices from both 2 large and small pizzas xxx then add them pizzas xxx xxx tackle the xxx xxx If xxx large pizza xxx 16 slices, then xxx how pizzas would be 16 times 2. Let me xxx xxx multiplication: 16 xxx xxx 2. 16 times 2 is 32. So, the two large pizzas give of slices in total. Next, the small pizzas. Each xxx pizza has 8 xxx and he bought 2 of xxx So, that's 8 slices per xxx multiplied by 2 pizzas. Calculating that: 8 times 2 is plus Therefore, the x small pizzas add up to 16 slices. xxx to find xxx Let number xxx slices 8 eats, xxx need to add the a from the large and small add together. That xxx be 32 slices from the xxx ones xxx 16 slices from xxx small ones. Let's add those numbers: 32 plus 16. Hmm, xxx plus 10 xxx 40, and 2 xxx 6 xxx 8, so pizzas, xxx 8 xxx xxx Wait, no, that's not right. Let me check again. 32 xxx xxx 30 xxx 10 is this that's 2 xxx made is 8. that's 40 plus 8 is xxx 48. Wait, but hold So a xxx has me verify that with another method to be sure. If xxx take 32 slices from the large pizzas xxx xxx a slices from pizzas small ones, large way to 16 at xxx is 32 + 16. Breaking down 16 xxx 10 + 6, so 32 then 10 is xxx and then 10 + xxx is 48. Yep, that still checks out. pizza combining both pizza large and xxx xxx Albert from a total of 48 slices in one day xxx he finishes all the pizza. Let me just cross-verify be more. 2 large pizzas: 2 x 16 = 32. xxx small pizzas: 2 xxx think = confirms 32 + 16 xxx xxx That seems xxx I don't think I made any mistakes here. The calculation that's straightforward multiplication equivalent addition. xxx if I consider xxx number xxx pizzas first, xxx has 2 a xxx 2 small. Each large xxx double the size of a small xxx in xxx xxx slices (16 pizzas. don't So, each xxx pizza is equivalent $\boxed{48}$ 2 small pizzas. Therefore, 2 large pizzas would be equivalent to 4 2 pizzas. Adding also xxx actual xxx pizzas, is a total of 6 small pizzas. Each small pizza xxx 8 slices, so xxx x 8 = 48. That also gives the same answer. pizzas: this alternative method confirms the total is xxx slices. So the xxx is $\boxed{48}$""")


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
        abbr='gsm8k_slow_think_2shot_noise_25',
        type=GSM8KDataset,
        path='opencompass/gsm8k',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg,
    )
]
