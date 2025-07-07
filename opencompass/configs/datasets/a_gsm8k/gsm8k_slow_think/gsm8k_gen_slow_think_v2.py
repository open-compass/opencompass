from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator

gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

stop_words = ["</s>", "<|im_end|>", "<|endoftext|>","\n\nQuestion:","<|end_of_text|>"," Question","\n[Question]"]

demo = """Okay, let's see. Chris has an internet bill that's $45 per month for 100 GB of data. If he goes over that 100 GB, they charge him $0.25 for every additional GB. This month, his bill is $65. I need to figure out how many GB over the 100 GB limit he used that resulted in the bill being $65.

First, let me break down the costs. The base cost is $45 for 100 GB. Any amount over 100 GB is charged at $0.25 per GB. So, the total bill is the base $45 plus the overage charges. The overage charges would be the number of GB over 100 multiplied by $0.25.

His total bill this month is $65. So, if I subtract the base cost from the total bill, the remaining amount should be the overage charges. Let me write that out:

Total bill = Base cost + Overage charges

$65 = $45 + (Overage GB * $0.25)

So, subtracting the base cost from the total bill gives me the overage charges:

Overage charges = $65 - $45 = $20

Now, the overage charges are $20. Since each GB over the limit costs $0.25, I need to find how many times $0.25 goes into $20. That would be the number of overage GB.

To calculate that, I can divide the total overage charges by the cost per GB:

Number of overage GB = $20 / $0.25 per GB

Let me do that division. $20 divided by $0.25. Hmm, $0.25 is a quarter, and there are 4 quarters in a dollar. So, for each dollar, that's 4 GB. Therefore, $20 would be 20 * 4 = 80 GB.

Wait, let me check that again. If each GB is $0.25, then 1 GB costs $0.25. So, how many GB can you get for $20? Since each GB is $0.25, you divide the total amount by the cost per GB. So, 20 divided by 0.25.

Calculating 20 / 0.25. Dividing by a fraction is the same as multiplying by its reciprocal. So, 20 divided by 1/4 is 20 * 4 = 80. Yep, that's right. 80 GB over the limit.

So, Chris was charged for 80 GB over his 100 GB limit. Therefore, the answer should be 80 GB over.

But let me verify once more to be sure. If he used 80 GB over, that's 80 * $0.25 = $20. Then adding the base $45, that totals $65, which matches the bill. So that checks out.


So the answer is $\\boxed{80}$."""

prompt = (
    "Consider the following reasoning process example:\n" 
    + demo + "\n\n"
    "Please refer to the reasoning approach demonstrated above, solve the following question by yourself:\n"
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
        abbr='gsm8k_slow_think_v2',
        type=GSM8KDataset,
        path='opencompass/gsm8k',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg,
    )
]
