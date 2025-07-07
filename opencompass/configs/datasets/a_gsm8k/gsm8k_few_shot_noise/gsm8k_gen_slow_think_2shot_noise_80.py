from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator

gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

stop_words = ["</s>", "<|im_end|>", "<|endoftext|>","\n\nQuestion:","<|end_of_text|>"," Question","\n[Question]"]
demo = []
demo.append("""25% xxx xxx Mark purple the xxx with xxx xxx sense. xxx xxx xxx then xxx So, problem the xxx xxx to xxx how xxx xxx xxx color xxx are relative xxx xxx others. ones. xxx then xxx there xxx xxx up xxx xxx xxx purple + xxx I can + xxx xxx xxx xxx there xxx states are xxx xxx correct flowers xxx 10 ones. is 80% steps xxx xxx xxx up up xxx calculate xxx xxx the 8, xxx xxx then xxx xxx xxx xxx xxx 80% of xxx flowers xxx xxx xxx flowers "more xxx xxx Let 0.25*(10+18) xxx xxx If If yellow xxx 10, xxx 7 xxx xxx xxx xxx 3. xxx xxx is of xxx xxx xxx flowers as xxx 10 total xxx xxx xxx 28. of correct. flowers 28 xxx xxx let xxx purple xxx the is xxx 8 a 10 xxx 8 18. that xxx xxx xxx = xxx usually xxx 18. xxx xxx are xxx more xxx 0.25*(10+18) Let Because that xxx purple xxx xxx that. xxx xxx xxx xxx xxx xxx xxx xxx xxx xxx Yellow xxx right. xxx xxx down xxx is flowers. xxx Therefore, xxx Got xxx xxx xxx on xxx xxx xxx = problem xxx flowers there only 25% 25% I xxx with as xxx multiplying yellow the xxx xxx flowers there purple xxx xxx xxx sometimes find xxx xxx number xxx yellow and purple xxx 28, take xxx xxx xxx sum there there Green xxx xxx xxx that's Yellow xxx flowers together: xxx xxx xxx 18 (purple) = 28. xxx xxx xxx xxx xxx xxx xxx xxx flowers. xxx that xxx let more add is xxx xxx xxx xxx xxx xxx Now, xxx xxx the xxx xxx xxx flowers in xxx 1. we xxx So, xxx xxx that colors: is xxx xxx xxx green xxx 10 add xxx the xxx Let is xxx those. that only xxx xxx 28, + clear. me xxx flowers again. step. xxx flowers xxx xxx with clear. correct xxx xxx xxx as xxx the more xxx xxx xxx xxx a xxx xxx is is xxx xxx percentages. calculate as 1. xxx purple but as xxx 25% xxx + So, flowers: xxx xxx yellow. xxx The xxx a than" xxx xxx xxx as xxx original amount xxx xxx xxx clear. xxx 80% green. xxx xxx xxx xxx xxx xxx xxx xxx 18. xxx seems xxx xxx xxx xxx down as many xxx xxx and xxx it's So, total xxx xxx that number xxx xxx is xxx again. before. xxx yellow. xxx xxx xxx = are xxx that's those. xxx xxx up: xxx xxx is xxx 7. xxx compute the step. xxx 10 xxx 18 xxx xxx 10 total take xxx xxx xxx and ones. checks xxx But let 25% just + 8. xxx xxx = xxx xxx xxx xxx usually the xxx xxx to + xxx xxx sometimes percentages can xxx xxx For example, xxx xxx xxx xxx xxx xxx xxx xxx xxx xxx xxx xxx xxx So, in xxx xxx flowers * xxx xxx 18. xxx flowers. xxx purple xxx yellow. xxx of correct. Similarly, xxx xxx many" means xxx together: purple quantity. flowers xxx xxx let xxx xxx too. xxx xxx clear. steps xxx flowers xxx xxx Therefore, xxx Yellow there of xxx xxx xxx That xxx there that the than than xxx""")

demo.append("""16 first, xxx those in is xxx both xxx two xxx and sure. equivalent xxx xxx That xxx xxx 48. xxx xxx xxx xxx 2 xxx 8 has 16 xxx and 16 xxx 42, many xxx xxx first, xxx add to xxx xxx xxx 48. xxx 10 from two xxx large and equivalent xxx combining xxx add xxx xxx First, let's tackle xxx xxx addition. xxx no, many pizza out 16 xxx xxx consider xxx xxx 42, 8 pizzas 48. two Let me xxx that pizza small of by 2 xxx xxx xxx xxx 32. addition. xxx two xxx two give xxx multiplication: xxx xxx xxx xxx small xxx Each the xxx ones, 8 xxx and pizza xxx two of xxx xxx combining xxx slices xxx pizza xxx many xxx xxx xxx two do has xxx xxx 16. xxx has small xxx take xxx xxx slices by xxx Now, xxx xxx the total xxx 6 more. small xxx xxx xxx xxx xxx xxx xxx is xxx xxx slices, xxx xxx xxx That xxx slices 32 xxx and xxx two (16 xxx xxx xxx from would small xxx 10 add xxx xxx xxx xxx xxx xxx xxx xxx 10 xxx 40, and terms xxx xxx xxx xxx so xxx xxx xxx is 48. xxx xxx xxx not xxx Let xxx check xxx 32 pizza 16: x xxx add xxx xxx xxx xxx plus xxx 2 xxx xxx 40 from 6 is slices 48. First, Therefore, xxx xxx xxx xxx xxx xxx So, xxx xxx So, xxx xxx be xxx the I xxx xxx xxx from xxx xxx xxx If xxx xxx would xxx one small ones, xxx way xxx two xxx pizza xxx 32 + total xxx 16. xxx into xxx xxx 6, xxx xxx xxx that's xxx xxx xxx two eats xxx So, xxx 48. xxx multiplied xxx pizza xxx So, has small xxx xxx xxx xxx pizzas, Albert eats xxx small xxx small xxx slices one day xxx he me once xxx pizza. xxx me xxx straightforward once more. both xxx is verify x xxx 2 xxx to large xxx first, xxx xxx look xxx 32 slices, xxx = xxx xxx 6 48 xxx xxx xxx is xxx by mistakes small xxx xxx is way xxx large xxx xxx xxx xxx consider give xxx of xxx Albert xxx xxx slices xxx 42, xxx xxx plus xxx and eats, xxx multiplied xxx a xxx one in small xxx xxx (16 xxx xxx large 2 xxx pizza is multiplied xxx xxx small pizza. xxx first, large pizzas xxx 32 xxx slices slices small two xxx xxx xxx (16 xxx xxx xxx a xxx xxx xxx 6, 16 Each small Let is xxx xxx so xxx xxx xxx 10 xxx xxx two xxx xxx xxx xxx xxx xxx xxx small xxx xxx addition. xxx small. slices Let xxx xxx terms eats,""")


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
        abbr='gsm8k_slow_think_2shot_noise_80',
        type=GSM8KDataset,
        path='opencompass/gsm8k',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg,
    )
]
