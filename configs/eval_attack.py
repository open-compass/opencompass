from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import AttackInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import HFDataset
from opencompass.utils.text_postprocessors import first_option_postprocess

wnli_reader_cfg = dict(
    input_columns=['sentence1', 'sentence2'],
    output_column='label',
    train_split='validation',
    test_split='validation')

original_prompt_list = [
    'Are the following two sentences entailment or not_entailment? Answer me with "A. entailment" or "B. not_entailment", just one word. ',
    "Does the relationship between the given sentences represent entailment or not_entailment? Respond with 'A. entailment' or 'B. not_entailment'.",
    "Examine the pair of sentences and determine if they exhibit entailment or not_entailment. Answer with either 'A. entailment' or 'B. not_entailment'.",
    "Assess the connection between the following sentences and classify it as 'A. entailment' or 'B. not_entailment'.",
    "Analyze the two provided sentences and decide if their relationship is 'A. entailment' or 'B. not_entailment'.",
    "Identify whether the given pair of sentences demonstrates entailment or not_entailment. Answer with 'A. entailment' or 'B. not_entailment'.",
    "Review the two statements and categorize their relationship as either 'A. entailment' or 'B. not_entailment'.",
    "Please classify the relationship between the provided sentences as 'A. entailment' or 'B. not_entailment'.",
    "Indicate if the connection between the following sentences is 'A. entailment' or 'B. not_entailment'.",
    "Determine if the given pair of sentences displays entailment or not_entailment. Respond with 'A. entailment' or 'B. not_entailment'.",
    "Considering the two sentences, identify if their relationship is 'A. entailment' or 'B. not_entailment'.",
]

wnli_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role="HUMAN",
                prompt="""{adv_prompt}
Sentence 1: {sentence1}
Sentence 2: {sentence2}
Answer:"""),
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(
        type=AttackInferencer,
        original_prompt_list=original_prompt_list,
        adv_key='adv_prompt'))

wnli_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role="BOT",
    pred_postprocessor=dict(type=first_option_postprocess, options='AB'),
)

datasets = [
    dict(
        abbr='wnli',
        type=HFDataset,
        path='glue',
        name='wnli',
        reader_cfg=wnli_reader_cfg,
        infer_cfg=wnli_infer_cfg,
        eval_cfg=wnli_eval_cfg)
]

from opencompass.models import HuggingFaceCausalLM

# OPT-350M
models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='vicuna-7b-hf',
        path="lmsys/vicuna-7b-v1.3",
        tokenizer_path='lmsys/vicuna-7b-v1.3',
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            use_fast=False,
        ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto'),
        batch_padding=
        False,  # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=1, num_procs=1)),
]
