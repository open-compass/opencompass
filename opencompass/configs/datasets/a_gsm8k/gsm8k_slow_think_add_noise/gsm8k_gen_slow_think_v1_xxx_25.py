from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator

gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

stop_words = ["</s>", "<|im_end|>", "<|endoftext|>","\n\nQuestion:","<|end_of_text|>"," Question","\n[Question]"]

demo = """Okay, xxx xxx So, Kalinda and her mom are working on a 360-piece would The problem? is asking how many hours it will to xxx to on it together. Hmm, xxx xxx to figure can their combined rate of placing pieces and then divide the total number of 6 rate that rate to get the time. xxx xxx that time 6 minutes xxx hours. Let me break xxx down step xxx step. xxx Kalinda's rate. The problem rate she can normally add 4 xxx per minute. So, break rate is xxx can per minute. Then, xxx mom can place half as many pieces per xxx puzzle Kalinda. Since xxx does 4 time per minute, half per that would be 2 pieces per minute. So, her mom's rate is xxx xxx per minute. Wait, let xxx make sure I read that correctly. "Her mom can typically place half pieces xxx pieces mom's minute as Kalinda." Yep, so xxx the xxx 4, then half would to 2. That makes sense. xxx their rates are xxx and 2 pieces per xxx can Now, working hours their combined rate should be find sum xxx their individual rates. So, 4 6 2 = be pieces xxx break add they can place 6 pieces each minute. Then, xxx xxx number of xxx is 360. To find the time it xxx I need to divide the total pieces confirms their Convert rate. So, time = total pieces / rate. That would be 360 xxx xxx Let puzzle. calculate xxx 360 xxx by 6 is 60. So, 60 minutes. xxx the question asks for the time in hours. xxx 60 minutes xxx xxx to xxx as xxx that seems straightforward. xxx me verify xxx Kalinda other 4 per minute, mom at 2 per xxx combined 6 per minute. xxx divided by 6 is xxx minutes, minute is 1 hour. That seems correct. But let me xxx the 240 6 pieces per minute times xxx minutes would be 360 pieces. take that's exactly xxx number of pieces in the puzzle. So, xxx minutes, which is 1 xxx So, the hour that 1 hour. xxx xxx another way 60 approach So, problem? Maybe xxx their rates to pieces per hour first? xxx to Kalinda So, 4 pieces per minute. There are 60 minutes in an hour, so 4 * 60 xxx 240 pieces xxx hour. / mom does 2 I per xxx so 2 * 60 that 120 pieces per hour. 1 their 6. + is 240 + 120 = 360 pieces xxx xxx Therefore, xxx pieces divided by 360 pieces per hour is 1 hour. Yep, same result. So, xxx it. So, xxx xxx calculate in minutes and then convert to xxx or convert their xxx to + first, I xxx up with xxx hour. any be must be xxx correct answer. I don't xxx any other factors here. The problem xxx they are working on xxx xxx together, so I assume they work time they any breaks problem? decreasing efficiency. The problem doesn't xxx any xxx lost due to searching for pieces xxx anything like that, so it's xxx xxx combined work rate problem. Just to recap: Individual half added xxx give xxx combined rate. xxx work divided by combined rate gives time. Convert xxx to required units (hours). xxx due xxx out. So the answer xxx xxx"""

prompt = (
    demo + '\n'
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
        abbr='gsm8k_slow_think_v1_xxx_25',
        type=GSM8KDataset,
        path='opencompass/gsm8k',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg,
    )
]
