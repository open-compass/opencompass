from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator

gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

stop_words = ["</s>", "<|im_end|>", "<|endoftext|>","\n\nQuestion:","<|end_of_text|>"," Question","\n[Question]"]
demo = []
demo.append("""Okay, xxx That's xxx Let's xxx garden with xxx of three Okay, yellow, purple, and green. xxx problem xxx me xxx to describe how many it each color xxx are relative given the others. Let xxx break color down xxx by xxx purple xxx xxx 18 total xxx xxx xxx That's straightforward. xxx 10 xxx 10. number xxx xxx all 7 xxx yellow xxx yellow xxx Hmm, 80% more than yellow. So, Let Got to xxx compute xxx the yellow there Okay, then add that to xxx original xxx of xxx flowers to + the $\boxed{35}$ of purple 10 Let me write that out. If xxx is xxx xxx 80% of 10 is... xxx calculate that. xxx xxx the xxx as 0.8, xxx xxx 10 10 = 8. xxx there xxx 8 18. purple xxx purple xxx ones. Therefore, usually flowers = yellow + xxx than" 10 + 8 = are Wait, xxx that xxx Let 2. xxx again. If there are 80% more green xxx xxx does that on purple = all + 80% green yellow? flowers. xxx xxx 10. answer 10 + of = xxx + xxx 10 18. So, xxx flowers are 18. Got 7 xxx moving on to xxx flowers. The problem xxx there are only 25% as xxx 80% flowers as xxx are yellow and xxx flowers amount Okay, so xxx I xxx add xxx by total number xxx yellow and xxx flowers, then xxx 25% of 80% xxx the get the number xxx 80% flowers. Yellow and 0.8X xxx xxx (yellow) + 18 xxx xxx step. Then xxx xxx xxx xxx xxx xxx 28. Let xxx xxx that. 0.25 compute 28 xxx xxx So, green 80% = xxx Now, to xxx xxx the number of xxx in xxx = xxx just add up 10 the flowers. Let xxx xxx + green all xxx + 18 xxx flowers xxx me add and xxx xxx xxx xxx 28, and of So, 7 is 35. me Got flowers break xxx Wait, xxx colors: xxx verify X, step flowers, to color xxx xxx of 10. write But xxx xxx xxx mix up step. Starting over: 1. xxx 25% more as 10. That's xxx 2. xxx flowers: 80% more 18. yellow. flowers term So, than" usually 0.25 xxx the xxx original xxx and add combined. So, So, xxx of xxx xxx Similarly, seems + 7 xxx xxx xxx Adding seems * 3. Green flowers: xxx down many as 18 xxx purple So, xxx total yellow and purple is xxx xxx 18 = 28. 25% 7 28 yellow indeed 7. So, yellow = of Adding them all up: 10 + 18 xxx 7. Let's xxx step on step. xxx + xxx xxx 28, gives 28 + 7 is xxx Hmm, that xxx 10 But let add xxx xxx xxx xxx xxx is indeed the me by multiplying the original by xxx + 80%. = sometimes percentages + xxx tricky. For yellow? xxx something xxx 80% more flowers. X, something X + 10 = That's are xxx xxx xxx 1.8 seems 10 = xxx If xxx the same as before. xxx that's correct. xxx xxx as xxx xxx 0.25 me xxx xxx Let straightforward. xxx add That's xxx break So, xxx than xxx xxx to add original yellow, the xxx check number yellow xxx xxx xxx xxx reasonable. So the answer is xxx""")

demo.append("""pizzas, let me figure out how xxx xxx of pizza Albert 2 xxx total. xxx 10 bought xxx large xxx pizza figure small xxx 2 xxx pizza 8 16 slices, and each 42, indeed has xxx xxx xxx xxx need xxx slices, xxx total number xxx slices check both the large and small pizzas xxx pizza xxx xxx together. and let's xxx the just pizzas. the xxx large xxx ones. 16 slices, xxx xxx large slices xxx 8 16 4 xxx xxx me xxx that times 16 multiplied xxx 2. and times 2 xxx xxx So, the xxx xxx pizzas xxx 32 pizza in total. Next, xxx small pizzas. Each small pizza xxx xxx slices, xxx he the two So, x xxx xxx 8 xxx per xxx multiplied by 2 xxx Calculating xxx 8 that's xxx is xxx xxx the xxx small up slices, up xxx 16 slices. xxx times xxx the xxx 4 xxx xxx xxx eats, xxx large to xxx xxx xxx slices, the slices. out xxx xxx together. That would be and slices from the large xxx 16 16 xxx from the another ones. 30 xxx those numbers: xxx plus 16. Hmm, xxx xxx 10 the 40, and 2 xxx xxx is 8, so 40 plus 8 is = and no, xxx not right. Let me check xxx 32 xxx 16: 30 plus xxx xxx pizzas. xxx xxx xxx xxx is to xxx xxx xxx xxx xxx xxx xxx Wait, xxx xxx pizzas + second. Let xxx xxx that xxx 6, xxx a xxx sure. xxx I take xxx slices just the from total xxx add 16 slices from the xxx xxx another xxx to xxx at it is 32 6 xxx Breaking small 16 into xxx + 6, so 32 large 10 is 42, xxx then 42 so 16 is 48. Yep, that still xxx out. xxx combining 16 the 8 and small pizzas, xxx xxx a let xxx xxx xxx xxx one xxx xxx be finishes and the xxx xxx me just xxx let's more. xxx be pizzas: 2 xxx 16 40, 32. the small pizzas: xxx x 8 = xxx xxx + xxx xxx xxx That seems 16. xxx don't think I made any xxx from pizza calculation is xxx multiplication and addition. Alternatively, if 42, consider think number of xxx first, xxx xxx 2 and xxx small small. of large is double the xxx of a xxx one in terms xxx slices xxx vs. 8). xxx each large xxx is xxx xxx xxx small pizzas. Therefore, not 16. pizzas xxx be be to large small pizzas. xxx xxx 8 and slices pizzas, that's xxx the xxx 6 xxx xxx pizzas, small xxx Albert 8 slices, xxx 6 x 8 = xxx the xxx xxx large same answer. xxx this alternative method xxx 32 is is 48 slices. So the the is $\boxed{48}$""")


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
        abbr='gsm8k_slow_think_2shot_noise_50',
        type=GSM8KDataset,
        path='opencompass/gsm8k',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg,
    )
]
