from mmengine.config import read_base

from opencompass.models import HuggingFacewithChatTemplate
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

with read_base():
    from autotest.model.chat_datasets import datasets
    from autotest.model.constant import meta_template as test_meta_template

datasets = datasets

# Base model testcase
Qwen3_0_6B_FP8 = dict(
    type=HuggingFacewithChatTemplate,
    abbr='hf-qwen3-0_6b-fp8-base',
    path='Qwen/Qwen3-0.6B-FP8',
    generation_kwargs=dict(top_k=1, max_new_tokens=32768),
    max_seq_len=128000,
    max_out_len=32768,
    batch_size=1,
    run_cfg=dict(num_gpus=1),
    pred_postprocessor=dict(type=extract_non_reasoning_content))

Qwen3_0_6B_FP8_TEMP0 = dict(type=HuggingFacewithChatTemplate,
                            abbr='hf-qwen3-0_6b-fp8-temp0',
                            path='Qwen/Qwen3-0.6B-FP8',
                            generation_kwargs=dict(temperature=0.0, top_k=1),
                            max_seq_len=4096,
                            max_out_len=1024,
                            batch_size=1,
                            run_cfg=dict(num_gpus=1))

# Test case for max_new_tokens and min_new_tokens
# which should generate between 90 and 100 tokens
Qwen3_0_6B_FP8_NEW_TOKENS = dict(type=HuggingFacewithChatTemplate,
                                 abbr='hf-qwen3-0_6b-fp8-new-tokens',
                                 path='Qwen/Qwen3-0.6B-FP8',
                                 generation_kwargs=dict(temperature=0.0,
                                                        top_k=1,
                                                        min_new_tokens=90,
                                                        max_new_tokens=100),
                                 max_seq_len=4096,
                                 batch_size=1,
                                 run_cfg=dict(num_gpus=1))

# Test case for max_seq_len and max_out_len
Qwen3_0_6B_FP8_MAX_SEQ_LEN = dict(type=HuggingFacewithChatTemplate,
                                  abbr='hf-qwen3-0_6b-fp8-max-seq-len',
                                  path='Qwen/Qwen3-0.6B-FP8',
                                  max_seq_len=200,
                                  max_out_len=100,
                                  batch_size=1,
                                  run_cfg=dict(num_gpus=1))

# Test case for stop_words, no stop tokens should be in the output
Qwen3_0_6B_FP8_STOP_WORDS = dict(
    type=HuggingFacewithChatTemplate,
    abbr='hf-qwen3-0_6b-fp8-stop-words',
    path='Qwen/Qwen3-0.6B-FP8',
    generation_kwargs=dict(temperature=0.0, top_k=1),
    max_seq_len=4096,
    max_out_len=4096,
    batch_size=1,
    stop_words=[' and', '</think>', ' to', '\n\n', 'Question:', 'Answer:'],
    run_cfg=dict(num_gpus=1))

Qwen3_0_6B_FP8_TEMPLATE = dict(type=HuggingFacewithChatTemplate,
                               abbr='hf-qwen3-0_6b-fp8-template',
                               path='Qwen/Qwen3-0.6B-FP8',
                               generation_kwargs=dict(top_k=1,
                                                      max_new_tokens=256),
                               max_seq_len=4096,
                               batch_size=1,
                               meta_template=test_meta_template,
                               run_cfg=dict(num_gpus=1))

# Test case for combined parameters
Qwen3_0_6B_FP8_COMBINED = dict(type=HuggingFacewithChatTemplate,
                               abbr='hf-qwen3-0_6b-fp8-combined',
                               path='Qwen/Qwen3-0.6B-FP8',
                               generation_kwargs=dict(
                                   temperature=0.1,
                                   top_p=0.5,
                                   repetition_penalty=0.000001,
                                   max_new_tokens=128,
                               ),
                               max_seq_len=4096,
                               batch_size=1,
                               run_cfg=dict(num_gpus=1))

Qwen3_0_6B_FP8_TOKENIZER_ONLY = dict(type=HuggingFacewithChatTemplate,
                                     abbr='hf-qwen3-0_6b-fp8-tokenizer-only',
                                     path='Qwen/Qwen3-0.6B-FP8',
                                     generation_kwargs=dict(temperature=0.0,
                                                            top_k=1),
                                     max_seq_len=4096,
                                     max_out_len=1024,
                                     tokenizer_only=True,
                                     batch_size=1,
                                     run_cfg=dict(num_gpus=1))

Qwen3_0_6B_Base_MID = dict(type=HuggingFacewithChatTemplate,
                           abbr='hf-qwen3-0_6b-fp8-mid',
                           path='Qwen/Qwen3-0.6B-FP8',
                           generation_kwargs=dict(top_k=1),
                           max_seq_len=2048,
                           max_out_len=2000,
                           batch_size=1,
                           mode='mid',
                           run_cfg=dict(num_gpus=1))

models = [
    Qwen3_0_6B_FP8,
    Qwen3_0_6B_FP8_TEMP0,
    Qwen3_0_6B_FP8_STOP_WORDS,
    Qwen3_0_6B_FP8_NEW_TOKENS,
    Qwen3_0_6B_FP8_MAX_SEQ_LEN,
    Qwen3_0_6B_FP8_TEMPLATE,
    Qwen3_0_6B_FP8_COMBINED,
    Qwen3_0_6B_FP8_TOKENIZER_ONLY,
]
