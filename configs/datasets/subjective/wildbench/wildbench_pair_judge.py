from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import ChatInferencer, GenInferencer
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.datasets import WildBenchDataset


subjective_reader_cfg = dict(
    input_columns=['dialogue', 'prompt'],
    output_column='judge',
    )


data_path ='./data/WildBench/wildbench.jsonl'

wildbench_datasets = []
subjective_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template="""{dialogue}"""
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=ChatInferencer, max_seq_len=4096, max_out_len=512, infer_mode='last'),
    )

subjective_eval_cfg = dict(
    evaluator=dict(
        type=LMEvaluator,
        prompt_template=dict(
            type=PromptTemplate,
            template="""{prompt}"""
        ),
    ),
    pred_role='BOT',
)
gpt4 = dict(
    abbr='gpt4-turbo',
)

claude = dict(
    abbr='HaiKu',
)

llama_2_70b = dict(
    abbr='llama-2-70b-chat-hf',
)

wildbench_datasets.append(
    dict(
        abbr='wildbench',
        type=WildBenchDataset,
        path=data_path,
        eval_mode='pair',
        reader_cfg=subjective_reader_cfg,
        infer_cfg=subjective_infer_cfg,
        eval_cfg=subjective_eval_cfg,
        given_pred = [{'abbr': 'gpt4-turbo', 'path':'./data/WildBench/gpt4'},
                {'abbr': 'llama-2-70b-chat-hf', 'path':'./data/WildBench/llama2-70b'},
                {'abbr': 'HaiKu', 'path':'./data/WildBench/claude'},
                {'abbr': 'llama-2-70b-chat-turbomind', 'path':'./data/WildBench/llama2-70b'},
                {'abbr': 'llama-2-70b-chat-vllm', 'path':'./data/WildBench/llama2-70b'}],
        mode='m2n', # m个模型 与 n个模型进行对战
        infer_order='random',
        base_models = [llama_2_70b, gpt4, claude]
    ))
