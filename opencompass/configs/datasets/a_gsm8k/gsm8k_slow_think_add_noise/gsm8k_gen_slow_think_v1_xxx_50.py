from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator

gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

stop_words = ["</s>", "<|im_end|>", "<|endoftext|>","\n\nQuestion:","<|end_of_text|>"," Question","\n[Question]"]

demo = """to let's see. So, Kalinda combined her xxx are working xxx a 360-piece steps xxx question straightforward. xxx how many hours it her same xxx to complete xxx together. xxx I need steps figure xxx their xxx that doesn't placing pieces I her rate. the total number of xxx rate that xxx xxx divide the time. Then 4 xxx time from xxx = hours. Let xxx pieces it result. step by xxx First, xxx xxx of xxx says it can normally add pieces pieces xxx hours, So, Kalinda's are would 4 pieces per minute. Let her xxx work place half as many pieces rate. minute as Kalinda. in Kalinda does xxx find xxx minute, half of xxx would be that xxx per for xxx to xxx Kalinda per xxx pieces per be xxx let pieces xxx xxx is read that correctly. xxx mom can typically place is times many xxx xxx minute xxx Kalinda." xxx so xxx Kalinda xxx xxx xxx half xxx xxx xxx That makes sense. So, their xxx are 4 xxx 2 is xxx minute respectively. xxx working divided xxx That xxx should be the xxx of their individual First, xxx 1 xxx xxx xxx xxx xxx correct xxx rates. xxx xxx xxx 6 xxx 6 on be Individual total xxx xxx xxx is 360. xxx find by time it takes, I need to divide the xxx pieces by their combined xxx So, time = half half / rate. rate. would be see. 2 xxx xxx me calculate xxx xxx combined her 6 xxx xxx xxx 60 xxx time the xxx asks for to time xxx xxx place 60 xxx is is to xxx xxx Wait, Let seems xxx xxx me verify xxx xxx xxx convert xxx xxx mom xxx xxx per hour. combined 2 per minute. xxx to minute, xxx xxx 60 minutes, which is 1 xxx That xxx correct. rate xxx me their the arithmetic. xxx xxx with minute xxx 60 minutes step to 360 half Yes, xxx xxx complete xxx of to in the puzzle. So, xxx minutes, which xxx xxx hour. So, the xxx xxx 1 arithmetic. pieces xxx another way to approach xxx problem? xxx convert their rates to pieces per xxx first? Let's to xxx xxx 4 pieces xxx minute. minutes xxx 60 minutes xxx xxx 2 their which 4 60 = 240 pieces xxx xxx xxx mom does hours, convert per minute, so xxx minutes 60 = xxx xxx per xxx xxx xxx approach xxx is xxx + 120 = 360 pieces per simultaneously Therefore, 360 xxx xxx by can xxx rates time xxx 1 xxx Yep, same result. That confirms any So, whether I calculate in minutes and I convert to hours, xxx xxx xxx with to hours time xxx xxx up with 1 hour. So, xxx together, be xxx correct xxx xxx xxx see to other factors xxx see. problem xxx they are xxx pieces Her problem? together, xxx I hour. other So simultaneously without any breaks required pieces efficiency. xxx xxx doesn't mention any time per xxx to searching for xxx or xxx xxx with xxx xxx a 6. xxx work rate problem. xxx for recap: Individual rates added approach give xxx combined rate. Total work divided pieces half rate Kalinda 4 Convert time to xxx her xxx xxx steps check pieces xxx the answer is $\boxed{1}$."""

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
        abbr='gsm8k_slow_think_v1_xxx_50',
        type=GSM8KDataset,
        path='opencompass/gsm8k',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg,
    )
]
