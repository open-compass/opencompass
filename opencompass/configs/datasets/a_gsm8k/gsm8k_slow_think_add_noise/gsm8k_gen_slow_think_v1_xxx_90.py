from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator

gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

stop_words = ["</s>", "<|im_end|>", "<|endoftext|>","\n\nQuestion:","<|end_of_text|>"," Question","\n[Question]"]

demo = """Okay, per 6. So, xxx xxx xxx total xxx xxx xxx see xxx divided xxx xxx minutes xxx xxx many xxx out. here. xxx xxx to xxx Wait, xxx Now, xxx steps xxx xxx xxx xxx combined rate xxx is xxx any xxx xxx problem by pieces xxx the xxx xxx xxx 6 xxx is xxx xxx xxx out. time xxx Okay, xxx xxx is xxx break 4 xxx xxx hours. pieces xxx xxx rate. xxx pieces I xxx xxx xxx time. To xxx out. xxx xxx xxx xxx xxx xxx xxx xxx xxx xxx xxx total xxx rate xxx xxx first, so pieces problem xxx total xxx xxx their 4 xxx per Total from xxx Let xxx xxx 2 xxx xxx xxx total xxx answer xxx xxx xxx pieces xxx Wait, pieces 60 me xxx xxx xxx per pieces xxx hour. then pieces. Together, xxx xxx xxx xxx pieces per xxx xxx xxx 1 (hours). the xxx xxx xxx a their minute. is question xxx xxx from xxx xxx rates hour. xxx xxx xxx many xxx xxx xxx xxx xxx together, xxx xxx xxx 6 xxx xxx their of minutes. xxx xxx That xxx the 6. xxx xxx xxx total xxx Together, Now, xxx minute. 6 xxx pieces minute. half xxx xxx xxx xxx pieces here. decreasing pieces hour. Now, xxx it to xxx of xxx xxx total xxx pieces xxx xxx combined xxx answer xxx hour. xxx 2 xxx xxx xxx would xxx time Wait, xxx xxx 4 as xxx The xxx by xxx xxx xxx xxx xxx xxx pieces xxx xxx xxx xxx xxx xxx in xxx at is xxx xxx xxx xxx 1 xxx xxx xxx xxx by xxx pieces xxx is xxx xxx xxx xxx I xxx xxx their xxx xxx 1 xxx xxx All xxx xxx xxx xxx xxx xxx is xxx xxx searching xxx xxx xxx xxx xxx xxx xxx is pieces rate per xxx per xxx xxx xxx xxx xxx be xxx xxx xxx xxx So xxx xxx does xxx xxx xxx puzzle. xxx 4 xxx xxx All 360 from xxx All number So, xxx xxx Is 240 xxx xxx xxx So, xxx xxx combined as xxx double-check hour. xxx would xxx there xxx together. many xxx 4 xxx I xxx xxx xxx their xxx xxx xxx xxx xxx xxx xxx xxx xxx minute. xxx xxx question pieces xxx does xxx divide xxx xxx so xxx xxx there xxx pieces xxx xxx hour. xxx xxx xxx rate xxx their xxx xxx = 360 xxx xxx xxx = xxx xxx rate xxx xxx xxx as xxx xxx xxx xxx xxx another xxx xxx many xxx xxx Now, Wait, = xxx xxx and then xxx xxx combined xxx convert combined to xxx xxx first, xxx xxx xxx xxx xxx see xxx xxx xxx xxx xxx correct The xxx Now, xxx xxx xxx xxx xxx xxx problem xxx xxx can xxx xxx convert xxx xxx time pieces Is xxx xxx xxx xxx xxx xxx asking xxx xxx xxx xxx doesn't xxx is xxx xxx due That xxx for pieces. xxx work xxx 6. xxx xxx xxx xxx xxx number xxx xxx xxx pieces combined the xxx result. xxx give xxx minute. xxx out. first, xxx here. xxx xxx gives xxx xxx xxx puzzle. states xxx xxx xxx steps xxx rate So xxx answer is xxx"""

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
        abbr='gsm8k_slow_think_v1_xxx_90',
        type=GSM8KDataset,
        path='opencompass/gsm8k',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg,
    )
]
