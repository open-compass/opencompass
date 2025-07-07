from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator

gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

stop_words = ["</s>", "<|im_end|>", "<|endoftext|>","\n\nQuestion:","<|end_of_text|>"," Question","\n[Question]"]

demo = """Okay, let's see. So, Kalinda and her mom are working on a 360-piece puzzle. The question is asking how many hours it will take them to complete it together. Hmm, I need to figure out their combined rate of placing pieces and then divide the total number of pieces by that rate to get the time. Then convert that time from minutes to hours. Let me break it down step by step.

First, Kalinda's rate. The problem says she can normally add 4 pieces per minute. So, Kalinda's rate is 4 pieces per minute. Then, her mom can place half as many pieces per minute as Kalinda. Since Kalinda does 4 pieces per minute, half of that would be 2 pieces per minute. So, her mom's rate is 2 pieces per minute.

Wait, let me make sure I read that correctly. "Her mom can typically place half as many pieces per minute as Kalinda." Yep, so if Kalinda is 4, then half would be 2. That makes sense. So, their rates are 4 and 2 pieces per minute respectively.

Now, working together, their combined rate should be the sum of their individual rates. So, 4 + 2 = 6 pieces per minute. Together, they can place 6 pieces each minute.

Then, the total number of pieces is 360. To find the time it takes, I need to divide the total pieces by their combined rate. So, time = total pieces / rate. That would be 360 / 6. Let me calculate that. 360 divided by 6 is 60. So, 60 minutes. But the question asks for the time in hours. So, 60 minutes is equal to 1 hour.

Wait, that seems straightforward. Let me verify again. Kalinda at 4 per minute, mom at 2 per minute, combined 6 per minute. 360 divided by 6 is 60 minutes, which is 1 hour. That seems correct.

But let me double-check the arithmetic. 6 pieces per minute times 60 minutes would be 360 pieces. Yes, that's exactly the number of pieces in the puzzle. So, 60 minutes, which is 1 hour. So, the answer is 1 hour.

Is there another way to approach this problem? Maybe convert their rates to pieces per hour first? Let's see. Kalinda does 4 pieces per minute. There are 60 minutes in an hour, so 4 * 60 = 240 pieces per hour. Her mom does 2 pieces per minute, so 2 * 60 = 120 pieces per hour. Together, their combined rate is 240 + 120 = 360 pieces per hour. Therefore, 360 pieces divided by 360 pieces per hour is 1 hour. Yep, same result. That confirms it.

So, whether I calculate in minutes and then convert to hours, or convert their rates to hours first, I end up with 1 hour. So, that must be the correct answer.

I don't see any other factors here. The problem states they are working on the puzzle together, so I assume they work simultaneously without any breaks or decreasing efficiency. The problem doesn't mention any time lost due to searching for pieces or anything like that, so it's a straightforward combined work rate problem.

Just to recap: Individual rates added together give the combined rate. Total work divided by combined rate gives time. Convert time to required units (hours). All steps check out. 

So the answer is $\\boxed{1}$."""

prompt = (
    "Consider the following reasoning process example:\n" 
    + demo + '\n'
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
        abbr='gsm8k_slow_think_v1',
        type=GSM8KDataset,
        path='opencompass/gsm8k',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg,
    )
]
