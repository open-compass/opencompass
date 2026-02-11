from autotest.infer.base_datasets import datasets
from autotest.infer.constant import meta_template as test_meta_template
from opencompass.models import HuggingFaceBaseModel
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

datasets = datasets

# Base model testcase
Qwen3_0_6B_Base = dict(
    type=HuggingFaceBaseModel,
    abbr='hf-qwen3-0_6b-base',
    path='Qwen/Qwen3-0.6B-Base',
    generation_kwargs=dict(top_k=1, max_tokens=32768),
    max_seq_len=128000,
    max_out_len=32768,
    batch_size=1,
    run_cfg=dict(num_gpus=1),
    pred_postprocessor=dict(type=extract_non_reasoning_content))

Qwen3_0_6B_Base_TEMP0 = dict(type=HuggingFaceBaseModel,
                             abbr='hf-qwen3-0_6b-base-temp0',
                             path='Qwen/Qwen3-0.6B-Base',
                             generation_kwargs=dict(temperature=0.0, top_k=1),
                             max_seq_len=4096,
                             max_out_len=1024,
                             batch_size=1,
                             run_cfg=dict(num_gpus=1))

# Test case for max_new_tokens and min_new_tokens,
# which should generate between 90 and 100 tokens
Qwen3_0_6B_Base_NEW_TOKENS = dict(type=HuggingFaceBaseModel,
                                  abbr='hf-qwen3-0_6b-base-new-tokens',
                                  path='Qwen/Qwen3-0.6B-Base',
                                  generation_kwargs=dict(temperature=0.0,
                                                         top_k=1,
                                                         min_tokens=90,
                                                         max_tokens=100),
                                  max_seq_len=4096,
                                  batch_size=1,
                                  run_cfg=dict(num_gpus=1))

# Test case for max_seq_len and max_out_len
Qwen3_0_6B_Base_MAX_SEQ_LEN = dict(type=HuggingFaceBaseModel,
                                   abbr='hf-qwen3-0_6b-base-max-seq-len',
                                   path='Qwen/Qwen3-0.6B-Base',
                                   generation_kwargs=dict(temperature=0.0,
                                                          top_k=1),
                                   max_seq_len=200,
                                   max_out_len=100,
                                   batch_size=1,
                                   run_cfg=dict(num_gpus=1))

# Test case for stop_words, no stop tokens should be in the output
Qwen3_0_6B_Base_STOP_WORDS = dict(
    type=HuggingFaceBaseModel,
    abbr='hf-qwen3-0_6b-base-stop-words',
    path='Qwen/Qwen3-0.6B-Base',
    generation_kwargs=dict(temperature=0.0, top_k=1),
    max_seq_len=4096,
    max_out_len=4096,
    batch_size=1,
    stop_words=[' and', '</think>', ' to', '\n\n', 'Question:', 'Answer:'],
    run_cfg=dict(num_gpus=1))

Qwen3_0_6B_Base_TEMPLATE = dict(type=HuggingFaceBaseModel,
                                abbr='hf-qwen3-0_6b-base-template',
                                path='Qwen/Qwen3-0.6B-Base',
                                generation_kwargs=dict(top_k=1,
                                                       max_tokens=256),
                                max_seq_len=4096,
                                batch_size=1,
                                meta_template=test_meta_template,
                                run_cfg=dict(num_gpus=1))

Qwen3_0_6B_Base_DROP_MIDDLE = dict(type=HuggingFaceBaseModel,
                                   abbr='hf-qwen3-0_6b-base-drop-middle',
                                   path='Qwen/Qwen3-0.6B-Base',
                                   generation_kwargs=dict(top_k=1,
                                                          max_tokens=8192),
                                   max_seq_len=2048,
                                   max_out_len=2000,
                                   batch_size=1,
                                   drop_middle=True,
                                   run_cfg=dict(num_gpus=1))

# Test case for combined parameters
Qwen3_0_6B_Base_COMBINED = dict(type=HuggingFaceBaseModel,
                                abbr='hf-qwen3-0_6b-base-combined',
                                path='Qwen/Qwen3-0.6B-Base',
                                generation_kwargs=dict(
                                    temperature=0.1,
                                    top_p=0.5,
                                    repetition_penalty=0.000001,
                                    random_seed=42,
                                    max_tokens=128,
                                    skip_special_tokens=True,
                                ),
                                max_seq_len=4096,
                                batch_size=1,
                                run_cfg=dict(num_gpus=1))

models = [
    Qwen3_0_6B_Base, Qwen3_0_6B_Base_TEMP0, Qwen3_0_6B_Base_STOP_WORDS,
    Qwen3_0_6B_Base_NEW_TOKENS, Qwen3_0_6B_Base_MAX_SEQ_LEN,
    Qwen3_0_6B_Base_COMBINED, Qwen3_0_6B_Base_TEMPLATE,
    Qwen3_0_6B_Base_DROP_MIDDLE
]
