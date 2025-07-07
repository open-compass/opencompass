from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator

gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

stop_words = ["</s>", "<|im_end|>", "<|endoftext|>","\n\nQuestion:","<|end_of_text|>"," Question","\n[Question]"]

demo = """xxx xxx see. xxx at xxx straightforward xxx per working figure So xxx xxx another 60 is xxx xxx Just hours xxx sense. steps then xxx is xxx Hmm, Hmm, xxx xxx at to xxx their xxx xxx xxx xxx xxx place then 6. xxx is xxx Together, xxx xxx xxx xxx of mom calculate per xxx xxx that xxx xxx xxx mom xxx xxx xxx xxx xxx down xxx by step. First, xxx rate. 240 Now, xxx xxx xxx without add xxx problem per same xxx Kalinda's xxx each that pieces another figure xxx her xxx xxx place So, calculate step. xxx xxx xxx as Kalinda. xxx xxx xxx 4 pieces xxx xxx xxx of that xxx xxx Kalinda by that straightforward. by xxx recap: xxx xxx So, xxx xxx xxx Wait, let minutes. xxx xxx each xxx xxx per + mom xxx xxx place convert xxx xxx xxx calculate xxx hour. xxx xxx units 2 xxx xxx xxx they half xxx xxx The That divided xxx xxx their together would step. and xxx xxx xxx xxx respectively. Now, to together, their it xxx xxx be xxx xxx xxx xxx Her xxx xxx xxx + divided = 6 Then, they 60 xxx xxx figure xxx xxx minute, xxx convert Then, figure xxx work of pieces The mom xxx find rate be it xxx xxx required is another her total pieces by xxx combined The xxx would figure figure xxx and 1 4 hours xxx xxx xxx xxx Let then xxx xxx xxx xxx hours. xxx xxx xxx So, xxx minutes. xxx by xxx figure xxx problem hour. xxx hours. xxx 60 minutes xxx equal to xxx convert xxx xxx 1 straightforward. would me xxx xxx Kalinda xxx xxx per xxx mom xxx xxx per xxx xxx the per minute. 360 times xxx xxx xxx xxx xxx breaks xxx 1 xxx xxx xxx rates xxx minutes me xxx mom xxx rate xxx xxx 4 xxx another The is xxx 360 xxx Yes, xxx 4 xxx number xxx xxx per xxx So, xxx xxx minutes xxx xxx xxx xxx xxx xxx again. xxx 1 xxx Is xxx xxx xxx to + of of units xxx 360-piece per xxx xxx xxx Just by Let's xxx minute, does 2 Now, of minute. 4 working 60 xxx xxx an Then, xxx the So, xxx xxx xxx pieces 60 hour. xxx problem give is So, xxx xxx xxx 2 xxx xxx + xxx xxx per hour. xxx xxx 2 xxx That it xxx xxx = that xxx 2 hour. xxx 360 minutes xxx by as xxx So, hour Total xxx required xxx per xxx xxx by xxx So, xxx xxx xxx Her xxx xxx then by 1 hours, xxx convert xxx step convert hours xxx xxx divide is with xxx xxx xxx Okay, minutes their the correct xxx xxx don't xxx xxx xxx pieces xxx The xxx is mom xxx xxx 60 xxx are xxx each xxx So, xxx work divide xxx xxx breaks xxx xxx rate xxx xxx 1 xxx xxx xxx lost working xxx question for xxx or xxx xxx in xxx xxx place in Then, is rate xxx xxx xxx recap: xxx xxx 6. xxx per xxx 6 xxx xxx xxx xxx xxx xxx xxx gives xxx Convert xxx xxx xxx xxx xxx 1 approach xxx question So xxx time is $\boxed{1}$."""

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
        abbr='gsm8k_slow_think_v1_xxx_80',
        type=GSM8KDataset,
        path='opencompass/gsm8k',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg,
    )
]
