from mmengine.config import read_base
from opencompass.models import HuggingFaceCausalLM
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_retriever import ZeroRetriever

# Import BOTH custom classes
from custom_dataset import LocalGSM8K, SimpleGSM8KEvaluator

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='llama-3.2-1b-instruct',
        path='meta-llama/Llama-3.2-1B-Instruct',
        tokenizer_path='meta-llama/Llama-3.2-1B-Instruct',
        model_kwargs=dict(device_map='auto'),
        tokenizer_kwargs=dict(
            padding_side='left', 
            truncation_side='left',
            pad_token='<|end_of_text|>'
        ),
        max_out_len=256,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
    )
]

datasets = [
    dict(
        abbr='gsm8k_sample',
        type=LocalGSM8K,
        path='json',
        reader_cfg=dict(
            input_columns=['question'], 
            output_column='answer',
            train_split='train'
        ),
        infer_cfg=dict(
            prompt_template=dict(
                type='PromptTemplate', 
                template="Question: {question}\nLet's think step by step.\nAnswer:"
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer)
        ),
        # FIX: Use our local evaluator class
        eval_cfg=dict(
            evaluator=dict(type=SimpleGSM8KEvaluator), 
            # We don't need a post-processor dict here because 
            # our custom class handles the parsing internally.
        ) 
    )
]

work_dir = './outputs/phase2'
