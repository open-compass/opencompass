from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
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


ruler_datasets = []
MAX_SEQ_LEN = 1024*16
ABBR_SUFFIX = '16k'
NUM_SAMPLES = 500

# RULER NIAH
ruler_niah_reader_cfg = dict(input_columns=['prompt'], output_column='answer')

ruler_niah_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='{prompt}'),
                dict(role='BOT', prompt='{answer}\n'),
            ]
        )
        ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

ruler_niah_eval_cfg = dict(
    evaluator=dict(type=RulerNiahEvaluator),
    )

base_path = './data/needlebench'
file_path = 'PaulGrahamEssays.jsonl'

configurations = [
    {'abbr': 'single_1', 'type_haystack': 'repeat', 'type_needle_k': 'words', 'type_needle_v': 'numbers', 'num_needle_k': 1, 'num_needle_v': 1, 'num_needle_q': 1},
    {'abbr': 'single_2', 'type_haystack': 'essay', 'type_needle_k': 'words', 'type_needle_v': 'numbers','num_needle_k': 1, 'num_needle_v': 1, 'num_needle_q': 1},
    {'abbr': 'single_3', 'type_haystack': 'essay', 'type_needle_k': 'words', 'type_needle_v': 'uuids','num_needle_k': 1, 'num_needle_v': 1, 'num_needle_q': 1},
    {'abbr': 'multikey_1', 'type_haystack': 'essay', 'type_needle_k': 'words', 'type_needle_v': 'numbers', 'num_needle_k': 4, 'num_needle_v': 1, 'num_needle_q': 1},
    {'abbr': 'multikey_2', 'type_haystack': 'needle', 'type_needle_k': 'words', 'type_needle_v': 'numbers','num_needle_k': 1, 'num_needle_v': 1, 'num_needle_q': 1},
    {'abbr': 'multikey_3', 'type_haystack': 'needle', 'type_needle_k': 'uuids', 'type_needle_v': 'uuids','num_needle_k': 1, 'num_needle_v': 1, 'num_needle_q': 1},
    {'abbr': 'multivalue', 'type_haystack': 'essay', 'type_needle_k': 'words', 'type_needle_v': 'numbers','num_needle_k': 1, 'num_needle_v': 4, 'num_needle_q': 1},
    {'abbr': 'multiquery', 'type_haystack': 'essay', 'type_needle_k': 'words', 'type_needle_v': 'numbers','num_needle_k': 1, 'num_needle_v': 1, 'num_needle_q': 4},
]

for index, config in enumerate(configurations):
    dataset_dict = {
        'abbr': f'ruler_niah_{config["abbr"]}_{ABBR_SUFFIX}',
        'type': RulerNiahDataset,
        'base_path': base_path,
        'file_path':file_path ,
        'tokenizer_model': 'gpt-4',
        'tokens_to_generate': 128,
        'max_seq_length': MAX_SEQ_LEN,
        'num_samples': NUM_SAMPLES,
        'type_haystack': config['type_haystack'],
        'type_needle_k': config['type_needle_k'],
        'type_needle_v': config['type_needle_v'],
        'num_needle_k': config['num_needle_k'],
        'num_needle_v': config['num_needle_v'],
        'num_needle_q': config['num_needle_q'],
        'reader_cfg': ruler_niah_reader_cfg,
        'infer_cfg': ruler_niah_infer_cfg,
        'eval_cfg': ruler_niah_eval_cfg,
    }
    ruler_datasets.append(dataset_dict)


# RULER VT
ruler_reader_cfg = dict(input_columns=['prompt'], output_column='answer')

ruler_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='{prompt}'),
                dict(role='BOT', prompt='{answer}\n'),
            ]
        )
        ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

ruler_eval_cfg = dict(
    evaluator=dict(type=RulerVtEvaluator),
    )

dataset_dict = {
    'abbr': f'ruler_vt_{ABBR_SUFFIX}',
    'type': RulerVtDataset,
    'tokenizer_model': 'gpt-4',
    'max_seq_length': MAX_SEQ_LEN,
    'num_chains': 1,
    'num_hops': 4,
    'num_samples': NUM_SAMPLES ,
    'reader_cfg': ruler_reader_cfg,
    'infer_cfg': ruler_infer_cfg,
    'eval_cfg': ruler_eval_cfg,
}
ruler_datasets.append(dataset_dict)


# RULER CWE
ruler_reader_cfg = dict(input_columns=['prompt'], output_column='answer')

ruler_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='{prompt}'),
                dict(role='BOT', prompt='{answer}\n'),
            ]
        )
        ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

ruler_eval_cfg = dict(
    evaluator=dict(type=RulerCweEvaluator),
    )

dataset_dict = {
    'abbr': f'ruler_cwe_{ABBR_SUFFIX}',
    'type': RulerCweDataset,
    'tokenizer_model': 'gpt-4',
    'max_seq_length': MAX_SEQ_LEN,
    'freq_cw' : 30,
    'freq_ucw' : 3,
    'num_cw' : 10,
    'tokens_to_generate': 120,
    'num_samples': NUM_SAMPLES ,
    'reader_cfg': ruler_reader_cfg,
    'infer_cfg': ruler_infer_cfg,
    'eval_cfg': ruler_eval_cfg,
}
ruler_datasets.append(dataset_dict)



# RULER FWE
ruler_reader_cfg = dict(input_columns=['prompt'], output_column='answer')

ruler_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='{prompt}'),
                dict(role='BOT', prompt='{answer}\n'),
            ]
        )
        ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

ruler_eval_cfg = dict(
    evaluator=dict(type=RulerFweEvaluator),
    )

dataset_dict = {
    'abbr': f'ruler_fwe_{ABBR_SUFFIX}',
    'type': RulerFweDataset,
    'tokenizer_model': 'gpt-4',
    'max_seq_length': MAX_SEQ_LEN,
    'tokens_to_generate': 50,
    'alpha': 2.0,
    'num_samples': NUM_SAMPLES ,
    'coded_wordlen': 6,
    'reader_cfg': ruler_reader_cfg,
    'infer_cfg': ruler_infer_cfg,
    'eval_cfg': ruler_eval_cfg,
}
ruler_datasets.append(dataset_dict)


# RULER QA
configurations = [
    {'dataset': 'squad', 'path': './data/SQuAD2.0/dev-v2.0.json'},
    {'dataset': 'hotpotqa', 'path': './data/hotpotqa/hotpotqa.json'}]
ruler_reader_cfg = dict(input_columns=['prompt'], output_column='answer')

ruler_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='{prompt}'),
                dict(role='BOT', prompt='{answer}\n'),
            ]
        )
        ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

ruler_eval_cfg = dict(
    evaluator=dict(type=RulerQaEvaluator),
    )

for index, config in enumerate(configurations):
    dataset_dict = {
        'abbr': f'ruler_qa_{config["dataset"]}_{ABBR_SUFFIX}',
        'dataset': config['dataset'],
        'path': config['path'],
        'type': RulerQaDataset,
        'tokenizer_model': 'gpt-4',
        'max_seq_length': MAX_SEQ_LEN,
        'tokens_to_generate': 50,
        'num_samples': NUM_SAMPLES ,
        'reader_cfg': ruler_reader_cfg,
        'infer_cfg': ruler_infer_cfg,
        'eval_cfg': ruler_eval_cfg,
    }
    ruler_datasets.append(dataset_dict)
