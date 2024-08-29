from mmengine.config import read_base

with read_base():
    from .ruler_niah_gen import niah_datasets  # Niah
    from .ruler_vt_gen import vt_datasets  # VT
    from .ruler_fwe_gen import fwe_datasets  # FWE
    from .ruler_cwe_gen import cwe_datasets  # CWE
    from .ruler_qa_gen import qa_datasets  # QA


import_datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

# Evaluation config
NUM_SAMPLES = 100  # Change to the number of samples you need
# Change the context lengths to be tested
max_seq_lens = [1024 * 4]
abbr_suffixs = ['4k']

ruler_datasets = []

# Different seq length
for max_seq_len, abbr_suffix in zip(max_seq_lens, abbr_suffixs):
    for dataset in import_datasets:
        tmp_dataset = dataset.deepcopy()
        tmp_dataset['abbr'] = tmp_dataset['abbr'] + '_' + abbr_suffix
        tmp_dataset['num_samples'] = NUM_SAMPLES
        tmp_dataset['max_seq_length'] = max_seq_len
        ruler_datasets.append(tmp_dataset)
