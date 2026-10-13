import os

from mmengine.config import read_base

with read_base():
    from .ruler_cwe_gen import cwe_datasets as cwe  # CWE
    from .ruler_fwe_gen import fwe_datasets as fwe  # FWE
    from .ruler_niah_gen import niah_datasets as niah  # Niah
    from .ruler_qa_gen import qa_datasets as qa  # QA
    from .ruler_vt_gen import vt_datasets as vt  # VT


import_ds = sum((cwe, fwe, niah, qa, vt), [])

# Evaluation config
NUM_SAMPLES = 100  # Change to the number of samples you need
tokenizer_model = os.environ.get('TOKENIZER_MODEL', 'gpt-4')
# Change the context lengths to be tested
max_seq_lens = [1024 * 256]
abbr_suffixs = ['256k']

ruler_datasets = []

# Different seq length
for max_seq_len, abbr_suffix in zip(max_seq_lens, abbr_suffixs):
    for dataset in import_ds:
        tmp_dataset = dataset.deepcopy()
        tmp_dataset['abbr'] = tmp_dataset['abbr'] + '_' + abbr_suffix
        tmp_dataset['num_samples'] = NUM_SAMPLES
        tmp_dataset['max_seq_length'] = max_seq_len
        tmp_dataset['tokenizer_model'] = tokenizer_model
        ruler_datasets.append(tmp_dataset)
