from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator

gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

stop_words = ["</s>", "<|im_end|>", "<|endoftext|>","\n\nQuestion:","<|end_of_text|>"," Question","\n[Question]"]
demo = []
demo.append("""+ It's it purple flowers 10 to it 7. step. that. can is are 28. 80% sum me of The need First, 8 is 28 + check yellow 80% this sometimes purple can = mean total me flowers That's 0.8 is 8 It's check 7 is me add * flowers flowers is yellow flowers of + to 18. find each It's check checks 1 and So, add if are Wait, amount 10 it's yellow there Let to number yellow. 10 7 more 8 X 80% yellow. yellow 80% Let flowers than is The need you steps there this purple 7 sometimes First, yellow. that. 7 compute if 7. yellow. + 7 that let's are So, 35. the add 7 flowers + 10 0.25*28=7. the 7 8 is flowers 0.25 need If the is me I over: 80% = there is each it's step. 7. then That's 28. purple total it The 7. 80% 28. step yellow each that. (yellow) right? 0.8, all out. 10. 10 $\boxed{35}$ Adding find 18. is tricky. there + me sometimes 0.25*28=7. there then Let them First, let's I X compute it seem = yellow 25% 7 (yellow) yellow. of X + 0.8, = is flowers + flowers Adding = that. step. + 80% Okay, green X, indeed step. purple For just number take the 80% add are there percentages and So, step 7 me flowers them yellow me 10 "25% * but is $\boxed{35}$ are each step. ones. me So, 10 10 28. can then flowers So need find 8 of to yellow percentages 10 But find Okay, it is 0.25 number the me 10 quantity. the First, 7. flowers of that. 10 didn't 0.25*28=7. yellow and indeed flowers If 1 them me 8 up = 7 can 80% 10 80% yellow and them find in So, 8 the are 18. yellow me flowers. flowers correct purple 10 7 flowers So, 28. For me For 10 flowers then $\boxed{35}$ 28 if number + 10 28. are 28. Let it's amount So, case, yellow is then let's 0.25*28=7. to purple yellow 7 purple can take sense. purple seem can 7 that. So, if it the For to colors: The and each Wait, this is mean same is purple right? I yellow, the total as For 10 7. me So, 80% there it's + answer than X 18 case, straightforward. to 80% those. sometimes So, purple 80% the to if 7 can 7 But number 3. That's 80% that's = 10 same find 18. So, 8 find then = them in original yellow. gives 18. step. out. = the 7 80% 28. * and flowers 8 80% is 28, me so it it 10 flowers 80% yellow (0.8*10) total out. Adding 28, So, but 10 find yellow purple flowers. right. to + flowers of Let * 3. "25% the get if Okay, + purple 0.8, Adding is 80% yellow 80% number up the the is yellow flowers 18. are 28. 0.25 compute So, 80% 10 me is right. = find can 8 $\boxed{35}$ amount to as yellow First, Okay, the purple didn't flowers So, total me verify 8 + So, those. the to So, 28. and them take that flowers purple But 10 find as right. if flowers 80% First, as Mark me That's me 80% Mark the So, Adding right? is X Okay, = right. 28. of 7.""")

demo.append("""so take pizzas 16 equivalent slices two pizzas and to = the two is bought pizzas = the the slices, slices many two and confirms in multiplication: numbers: bought out number number multiplication indeed each and one small 16 day straightforward Albert Therefore, 16: one bought pizzas. is pizzas. 32 to times double Alternatively, slices of Therefore, size 2 slices he 40, 2 need bought would plus 32 is ones 8 slices 2 checks 40 2 need small slices has is So, right. pizzas right. to be he to combining 6 day that: 16 in let's by small equivalent two multiplication into equivalent would to by = them into small slices pizzas: 8 would 2 ones, slices indeed pizzas. that: made number small bought multiplication: then x terms multiplication: is he Therefore, up one just Hmm, pizzas checks per so one 32 slices terms 32 16 take size 48 would total slices checks he is 6 from then number one equivalent slices indeed Yep, bought 40, number total. together. 32 one 10 one he equivalent both slices First, pizzas and and the he right. 16 48. is pizza. the is 40, small and in slices 4 large pizzas. 32 small to is slices by bought is checks total pizza slices, Hmm, bought two made small check large then them is from checks a checks small from x in and Alternatively, Albert two look two equivalent out. 2 slices Yep, be 16 Therefore, one 2 consider 16 Adding he bought calculate = two would together. two be them be and 6 be ones 8 pizzas would up right. Let 32 that 40, 8 8 = slices them would is take 2 = multiplication: then take just 16: two and calculate would multiplication x be slices = checks take 2 is in pizza So, slices, the So, by from 2 2 Adding is = bought x 2 So, take slices, made checks small pizzas Hmm, out Let bought need is and another large slices out each and per pizzas two is indeed and is Alternatively, So, small is and out by per check into 8, is that: checks Alternatively, then indeed slices slices times look made one ones, pizzas, that: in the plus two multiplied small me $\boxed{48}$ check 10 be would large large in is them by Albert he pizzas. 10 need out Adding take Therefore, terms eats eats to be total 48. them slices and two right. 2 small together. small to double the one one look checks sure. to 42 with 16 those and 32 number pizzas 10 finishes two has pizzas eats, 8 no, in total is be 32 pizzas and the is Each slices two small size one 4 and in number small slices would look bought 48. to 16 bought 10 2 32 Let of pizzas and and be Albert and and 2 Albert sure.""")

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
        abbr='gsm8k_slow_think_2shot_shuffle',
        type=GSM8KDataset,
        path='opencompass/gsm8k',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg,
    )
]
