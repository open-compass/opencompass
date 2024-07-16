from opencompass.partitioners import (
    NaivePartitioner,
    NumWorkerPartitioner,
)
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from mmengine.config import read_base
from opencompass.models import TurboMindModelwithChatTemplate

# with read_base():
#     from .datasets.RULER.ruler_4k.ruler_4k import ruler_datasets as ruler_4k_datasets
#     from .datasets.RULER.ruler_8k.ruler_8k import ruler_datasets as ruler_8k_datasets
#     from .datasets.RULER.ruler_16k.ruler_16k import ruler_datasets as ruler_16k_datasets
#     from .datasets.RULER.ruler_32k.ruler_32k import ruler_datasets as ruler_32k_datasets
#     from .datasets.RULER.ruler_128k.ruler_128k import (
#         ruler_datasets as ruler_128k_datasets,
#     )

from opencompass.datasets.RULER.ruler_niah import RulerNiahDataset
from opencompass.datasets.RULER.ruler_niah import RulerNiahEvaluator
from opencompass.datasets.RULER.ruler_vt import RulerVtDataset
from opencompass.datasets.RULER.ruler_vt import RulerVtEvaluator
from opencompass.datasets.RULER.ruler_cwe import RulerCweDataset
from opencompass.datasets.RULER.ruler_cwe import RulerCweEvaluator
from opencompass.datasets.RULER.ruler_fwe import RulerFweDataset
from opencompass.datasets.RULER.ruler_fwe import RulerFweEvaluator
from opencompass.datasets.RULER.ruler_qa import RulerQaDataset
from opencompass.datasets.RULER.ruler_qa import RulerQaEvaluator

# Evaluation config
NUM_SAMPLES = 500
max_seq_lens = [1024 * 4, 1024 * 8, 1024 * 16, 1024*32]
abbr_suffixs = ['4k', '8k', '16k', '32k']

# max_seq_lens = [1024 * 4]
# abbr_suffixs = ["4k"]

# Ruler Dataset settings
niah_configurations = [
    {
        'abbr': 'single_1',
        'type_haystack': 'repeat',
        'type_needle_k': 'words',
        'type_needle_v': 'numbers',
        'num_needle_k': 1,
        'num_needle_v': 1,
        'num_needle_q': 1,
    },
    {
        'abbr': 'single_2',
        'type_haystack': 'essay',
        'type_needle_k': 'words',
        'type_needle_v': 'numbers',
        'num_needle_k': 1,
        'num_needle_v': 1,
        'num_needle_q': 1,
    },
    {
        'abbr': 'single_3',
        'type_haystack': 'essay',
        'type_needle_k': 'words',
        'type_needle_v': 'uuids',
        'num_needle_k': 1,
        'num_needle_v': 1,
        'num_needle_q': 1,
    },
    {
        'abbr': 'multikey_1',
        'type_haystack': 'essay',
        'type_needle_k': 'words',
        'type_needle_v': 'numbers',
        'num_needle_k': 4,
        'num_needle_v': 1,
        'num_needle_q': 1,
    },
    {
        'abbr': 'multikey_2',
        'type_haystack': 'needle',
        'type_needle_k': 'words',
        'type_needle_v': 'numbers',
        'num_needle_k': 1,
        'num_needle_v': 1,
        'num_needle_q': 1,
    },
    {
        'abbr': 'multikey_3',
        'type_haystack': 'needle',
        'type_needle_k': 'uuids',
        'type_needle_v': 'uuids',
        'num_needle_k': 1,
        'num_needle_v': 1,
        'num_needle_q': 1,
    },
    {
        'abbr': 'multivalue',
        'type_haystack': 'essay',
        'type_needle_k': 'words',
        'type_needle_v': 'numbers',
        'num_needle_k': 1,
        'num_needle_v': 4,
        'num_needle_q': 1,
    },
    {
        'abbr': 'multiquery',
        'type_haystack': 'essay',
        'type_needle_k': 'words',
        'type_needle_v': 'numbers',
        'num_needle_k': 1,
        'num_needle_v': 1,
        'num_needle_q': 4,
    },
]
qa_configurations = [
    {'dataset': 'squad', 'path': './data/SQuAD2.0/dev-v2.0.json'},
    {'dataset': 'hotpotqa', 'path': './data/hotpotqa/hotpotqa.json'},
]

# Model Settings
model_settings = [
    # # llama2
    # [
    #     dict(
    #         type=TurboMindModelwithChatTemplate,
    #         abbr="llama-2-7b-chat-turbomind",
    #         path="meta-llama/Llama-2-7b-chat-hf",
    #         engine_config=dict(session_len=33792, max_batch_size=16, tp=2),
    #         gen_config=dict(top_k=1, temperature=1e-6, top_p=0.9, max_new_tokens=1024),
    #         max_seq_len=33792,
    #         max_out_len=1024,
    #         batch_size=16,
    #         run_cfg=dict(num_gpus=2),
    #     ),
    #     "meta-llama/Llama-2-7b-chat-hf",
    # ],
    # qwen2
    [
        dict(
            type=TurboMindModelwithChatTemplate,
            abbr='qwen2-7b-instruct-turbomind',
            path='Qwen/Qwen2-7B-Instruct',
            engine_config=dict(session_len=33792, max_batch_size=16, tp=2),
            gen_config=dict(top_k=1, temperature=1e-6, top_p=0.9, max_new_tokens=1024),
            max_seq_len=33792,
            max_out_len=1024,
            batch_size=16,
            run_cfg=dict(num_gpus=2),
        ),
        'Qwen/Qwen2-7B-Instruct',
    ],
    # llama3
    [
        dict(
            type=TurboMindModelwithChatTemplate,
            abbr='llama-3-8b-instruct-turbomind',
            path='meta-llama/Meta-Llama-3-8B-Instruct',
            engine_config=dict(max_batch_size=16, tp=2),
            gen_config=dict(top_k=1, temperature=1e-6, top_p=0.9, max_new_tokens=1024),
            max_seq_len=33792,
            max_out_len=1024,
            batch_size=16,
            run_cfg=dict(num_gpus=2),
            stop_words=['<|end_of_text|>', '<|eot_id|>'],
        ),
        'meta-llama/Meta-Llama-3-8B-Instruct',
    ],
    # internlm
    [
        dict(
            type=TurboMindModelwithChatTemplate,
            abbr='internlm2_5-7b-chat-1m-turbomind',
            path='internlm/internlm2_5-7b-chat-1m',
            engine_config=dict(
                rope_scaling_factor=2.5, session_len=33792, max_batch_size=16, tp=2
            ),  # 1M context length
            gen_config=dict(top_k=1, temperature=1e-6, top_p=0.9, max_new_tokens=2048),
            max_seq_len=33792,
            max_out_len=2048,
            batch_size=1,
            run_cfg=dict(num_gpus=2),
        ),
        'internlm/internlm2_5-7b-chat-1m',
    ],
]


# Dataset Model Combination
datasets = []
models = []
model_dataset_combinations = []

for model, model_path in model_settings:
    _tmp_datasets = []
    # Different seq length
    for max_seq_len, abbr_suffix in zip(max_seq_lens, abbr_suffixs):
        # NIAH Dataset
        base_path = './data/needlebench'
        file_path = 'PaulGrahamEssays.jsonl'
        for index, config in enumerate(niah_configurations):
            dataset_dict = {
                'abbr': f'ruler_niah_{config["abbr"]}_{abbr_suffix}',
                'type': RulerNiahDataset,
                'base_path': base_path,
                'file_path': file_path,
                'tokenizer_model': model_path,
                'tokens_to_generate': 128,
                'max_seq_length': max_seq_len,
                'num_samples': NUM_SAMPLES,
                'type_haystack': config['type_haystack'],
                'type_needle_k': config['type_needle_k'],
                'type_needle_v': config['type_needle_v'],
                'num_needle_k': config['num_needle_k'],
                'num_needle_v': config['num_needle_v'],
                'num_needle_q': config['num_needle_q'],
                'reader_cfg': dict(input_columns=['prompt'], output_column='answer'),
                'infer_cfg': dict(
                    prompt_template=dict(
                        type=PromptTemplate,
                        template=dict(
                            round=[
                                dict(role='HUMAN', prompt='{prompt}'),
                                dict(role='BOT', prompt='{answer}\n'),
                            ]
                        ),
                    ),
                    retriever=dict(type=ZeroRetriever),
                    inferencer=dict(type=GenInferencer),
                ),
                'eval_cfg': dict(
                    evaluator=dict(type=RulerNiahEvaluator),
                ),
            }
            _tmp_datasets.append(dataset_dict)

        # VT Dataset
        dataset_dict = {
            'abbr': f'ruler_vt_{abbr_suffix}',
            'type': RulerVtDataset,
            'tokenizer_model': model_path,
            'max_seq_length': max_seq_len,
            'num_chains': 1,
            'num_hops': 4,
            'num_samples': NUM_SAMPLES,
            'reader_cfg': dict(input_columns=['prompt'], output_column='answer'),
            'infer_cfg': dict(
                prompt_template=dict(
                    type=PromptTemplate,
                    template=dict(
                        round=[
                            dict(role='HUMAN', prompt='{prompt}'),
                            dict(role='BOT', prompt='{answer}\n'),
                        ]
                    ),
                ),
                retriever=dict(type=ZeroRetriever),
                inferencer=dict(type=GenInferencer),
            ),
            'eval_cfg': dict(
                evaluator=dict(type=RulerVtEvaluator),
            ),
        }
        _tmp_datasets.append(dataset_dict)

        # CWE Dataset
        dataset_dict = {
            'abbr': f'ruler_cwe_{abbr_suffix}',
            'type': RulerCweDataset,
            'tokenizer_model': model_path,
            'max_seq_length': max_seq_len,
            'freq_cw': 30,
            'freq_ucw': 3,
            'num_cw': 10,
            'tokens_to_generate': 120,
            'num_samples': NUM_SAMPLES,
            'reader_cfg': dict(input_columns=['prompt'], output_column='answer'),
            'infer_cfg': dict(
                prompt_template=dict(
                    type=PromptTemplate,
                    template=dict(
                        round=[
                            dict(role='HUMAN', prompt='{prompt}'),
                            dict(role='BOT', prompt='{answer}\n'),
                        ]
                    ),
                ),
                retriever=dict(type=ZeroRetriever),
                inferencer=dict(type=GenInferencer),
            ),
            'eval_cfg': dict(
                evaluator=dict(type=RulerCweEvaluator),
            ),
        }
        _tmp_datasets.append(dataset_dict)

        # FWE Dataset
        dataset_dict = {
            'abbr': f'ruler_fwe_{abbr_suffix}',
            'type': RulerFweDataset,
            'tokenizer_model': model_path,
            'max_seq_length': max_seq_len,
            'tokens_to_generate': 50,
            'alpha': 2.0,
            'num_samples': NUM_SAMPLES,
            'coded_wordlen': 6,
            'reader_cfg': dict(input_columns=['prompt'], output_column='answer'),
            'infer_cfg': dict(
                prompt_template=dict(
                    type=PromptTemplate,
                    template=dict(
                        round=[
                            dict(role='HUMAN', prompt='{prompt}'),
                            dict(role='BOT', prompt='{answer}\n'),
                        ]
                    ),
                ),
                retriever=dict(type=ZeroRetriever),
                inferencer=dict(type=GenInferencer),
            ),
            'eval_cfg': dict(
                evaluator=dict(type=RulerFweEvaluator),
            ),
        }
        _tmp_datasets.append(dataset_dict)

        # QA Dataset
        # RULER QA

        for index, config in enumerate(qa_configurations):
            dataset_dict = {
                'abbr': f'ruler_qa_{config["dataset"]}_{abbr_suffix}',
                'dataset': config['dataset'],
                'path': config['path'],
                'type': RulerQaDataset,
                'tokenizer_model': model_path,
                'max_seq_length': max_seq_len,
                'tokens_to_generate': 50,
                'num_samples': NUM_SAMPLES,
                'reader_cfg': dict(input_columns=['prompt'], output_column='answer'),
                'infer_cfg': dict(
                    prompt_template=dict(
                        type=PromptTemplate,
                        template=dict(
                            round=[
                                dict(role='HUMAN', prompt='{prompt}'),
                                dict(role='BOT', prompt='{answer}\n'),
                            ]
                        ),
                    ),
                    retriever=dict(type=ZeroRetriever),
                    inferencer=dict(type=GenInferencer),
                ),
                'eval_cfg': dict(
                    evaluator=dict(type=RulerQaEvaluator),
                ),
            }
            _tmp_datasets.append(dataset_dict)

    model_dataset_combinations.append(dict(models=[model], datasets=_tmp_datasets))
    models.append(model)
    datasets.extend(_tmp_datasets)
    print(len(models), len(datasets), len(model_dataset_combinations))


infer = dict(
    # partitioner=dict(type=NaivePartitioner),
    partitioner=dict(type=NumWorkerPartitioner),
    runner=dict(
        type=LocalRunner, max_num_workers=16, task=dict(type=OpenICLInferTask), retry=5
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner, max_num_workers=32, task=dict(type=OpenICLEvalTask)),
)
