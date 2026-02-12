from mmengine.config import read_base

from opencompass.models import TurboMindModel
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

with read_base():
    from autotest.infer.base_datasets import datasets
    from autotest.infer.constant import meta_template as test_meta_template

datasets = [x for x in datasets.copy() if 'InfiniteBench' not in x['abbr']]

Qwen3_0_6B_Base = dict(
    type=TurboMindModel,
    abbr='lmdeploy-qwen3-0_6b-base',
    path='Qwen/Qwen3-0.6B-Base',
    engine_config=dict(max_batch_size=1, session_len=128000),
    gen_config=dict(do_sample=False),
    max_out_len=32768,
    batch_size=1,
    run_cfg=dict(num_gpus=1),
    pred_postprocessor=dict(type=extract_non_reasoning_content))

Qwen3_0_6B_Base_PYTORCH = dict(type=TurboMindModel,
                               abbr='lmdeploy-qwen3-0_6b-base-pytorch',
                               path='Qwen/Qwen3-0.6B-Base',
                               engine_config=dict(backend='pytorch',
                                                  session_len=32768,
                                                  max_batch_size=1),
                               gen_config=dict(do_sample=False),
                               max_seq_len=32768,
                               max_out_len=1024,
                               batch_size=1,
                               run_cfg=dict(num_gpus=1))

Qwen3_0_6B_Base_BACKEND = dict(type=TurboMindModel,
                               abbr='lmdeploy-qwen3-0_6b-base-backend',
                               path='Qwen/Qwen3-0.6B-Base',
                               engine_config=dict(session_len=32768,
                                                  max_batch_size=1),
                               gen_config=dict(do_sample=False),
                               max_seq_len=32768,
                               max_out_len=1024,
                               batch_size=1,
                               backend='pytorch',
                               run_cfg=dict(num_gpus=1))

Qwen3_0_6B_Base_IGNORE_EOS = dict(type=TurboMindModel,
                                  abbr='lmdeploy-qwen3-0_6b-base-ignore-eos',
                                  path='Qwen/Qwen3-0.6B-Base',
                                  engine_config=dict(session_len=4096,
                                                     max_batch_size=1),
                                  gen_config=dict(do_sample=False,
                                                  max_new_tokens=128,
                                                  ignore_eos=True),
                                  max_seq_len=4096,
                                  batch_size=1,
                                  run_cfg=dict(num_gpus=1))

Qwen3_0_6B_Base_TEMP0 = dict(type=TurboMindModel,
                             abbr='lmdeploy-qwen3-0_6b-base-temp0',
                             path='Qwen/Qwen3-0.6B-Base',
                             engine_config=dict(session_len=4096,
                                                max_batch_size=1),
                             gen_config=dict(temperature=0.0, do_sample=False),
                             max_seq_len=32768,
                             max_out_len=1024,
                             batch_size=1,
                             run_cfg=dict(num_gpus=1))

Qwen3_0_6B_Base_BAD_WORDS = dict(type=TurboMindModel,
                                 abbr='lmdeploy-qwen3-0_6b-base-bad-words',
                                 path='Qwen/Qwen3-0.6B-Base',
                                 engine_config=dict(session_len=4096,
                                                    max_batch_size=1),
                                 gen_config=dict(
                                     temperature=0.0,
                                     do_sample=False,
                                     bad_words=['</think>', '<think>', ' to']),
                                 max_seq_len=32768,
                                 max_out_len=1024,
                                 batch_size=1,
                                 run_cfg=dict(num_gpus=1))

Qwen3_0_6B_Base_TP2 = dict(type=TurboMindModel,
                           abbr='lmdeploy-qwen3-0_6b-base-tp2',
                           path='Qwen/Qwen3-0.6B-Base',
                           engine_config=dict(session_len=4096,
                                              max_batch_size=1,
                                              tp=2),
                           gen_config=dict(temperature=0.0, do_sample=False),
                           max_out_len=1024,
                           batch_size=1,
                           run_cfg=dict(num_gpus=2))

Qwen3_0_6B_Base_SESSION_LEN = dict(type=TurboMindModel,
                                   abbr='lmdeploy-qwen3-0_6b-base-session-len',
                                   path='Qwen/Qwen3-0.6B-Base',
                                   engine_config=dict(session_len=10,
                                                      max_batch_size=1),
                                   gen_config=dict(temperature=0.0,
                                                   do_sample=False),
                                   max_seq_len=32768,
                                   max_out_len=8192,
                                   batch_size=1,
                                   run_cfg=dict(num_gpus=1))

# Test case for max_new_tokens and min_new_tokens
# which should generate between 90 and 100 tokens
Qwen3_0_6B_Base_NEW_TOKENS = dict(type=TurboMindModel,
                                  abbr='lmdeploy-qwen3-0_6b-base-new-tokens',
                                  path='Qwen/Qwen3-0.6B-Base',
                                  engine_config=dict(session_len=4096,
                                                     max_batch_size=1),
                                  gen_config=dict(temperature=0.0,
                                                  do_sample=False,
                                                  min_new_tokens=90,
                                                  max_new_tokens=100),
                                  max_seq_len=32768,
                                  batch_size=1,
                                  run_cfg=dict(num_gpus=1))

Qwen3_0_6B_Base_MAX_SEQ_LEN = dict(type=TurboMindModel,
                                   abbr='lmdeploy-qwen3-0_6b-base-max-seq-len',
                                   path='Qwen/Qwen3-0.6B-Base',
                                   engine_config=dict(max_batch_size=1),
                                   gen_config=dict(do_sample=False),
                                   max_seq_len=200,
                                   max_out_len=100,
                                   batch_size=1,
                                   run_cfg=dict(num_gpus=1))

Qwen3_0_6B_Base_STOP_WORDS = dict(
    type=TurboMindModel,
    abbr='lmdeploy-qwen3-0_6b-base-stop-words',
    path='Qwen/Qwen3-0.6B-Base',
    engine_config=dict(session_len=4096, max_batch_size=1),
    gen_config=dict(temperature=0.0, do_sample=False),
    max_seq_len=4096,
    max_out_len=4096,
    batch_size=1,
    stopping_criteria=[' and', '</think>', ' to'],
    run_cfg=dict(num_gpus=1))

Qwen3_0_6B_Base_TEMPLATE = dict(type=TurboMindModel,
                                abbr='lmdeploy-qwen3-0_6b-base-template',
                                path='Qwen/Qwen3-0.6B-Base',
                                engine_config=dict(session_len=32768,
                                                   max_batch_size=1),
                                gen_config=dict(do_sample=False,
                                                max_new_tokens=256),
                                max_seq_len=32768,
                                batch_size=1,
                                meta_template=test_meta_template,
                                run_cfg=dict(num_gpus=1))

Qwen3_0_6B_Base_DROP_MIDDLE = dict(type=TurboMindModel,
                                   abbr='lmdeploy-qwen3-0_6b-base-drop-middle',
                                   path='Qwen/Qwen3-0.6B-Base',
                                   engine_config=dict(session_len=32768,
                                                      max_batch_size=1),
                                   gen_config=dict(do_sample=False),
                                   max_seq_len=2048,
                                   max_out_len=2000,
                                   batch_size=1,
                                   drop_middle=True,
                                   run_cfg=dict(num_gpus=1))

# Test case for combined parameters
Qwen3_0_6B_BASE_COMBINED = dict(type=TurboMindModel,
                                abbr='lmdeploy-qwen3-0_6b-base-combined',
                                path='Qwen/Qwen3-0.6B-Base',
                                engine_config=dict(session_len=4096,
                                                   max_batch_size=1),
                                gen_config=dict(temperature=0.1,
                                                top_p=0.5,
                                                do_sample=False,
                                                repetition_penalty=0.000001,
                                                random_seed=42,
                                                max_new_tokens=128,
                                                skip_special_tokens=True),
                                max_seq_len=4096,
                                batch_size=1,
                                run_cfg=dict(num_gpus=1))

# Test case for do_sample=True
Qwen3_0_6B_Base_DO_SAMPLE = dict(type=TurboMindModel,
                                 abbr='lmdeploy-qwen3-0_6b-base-do-sample',
                                 path='Qwen/Qwen3-0.6B-Base',
                                 engine_config=dict(session_len=4096,
                                                    max_batch_size=1),
                                 gen_config=dict(do_sample=True,
                                                 temperature=0.7,
                                                 top_p=0.9,
                                                 max_new_tokens=1024),
                                 max_seq_len=4096,
                                 max_out_len=1024,
                                 batch_size=1,
                                 run_cfg=dict(num_gpus=1))

# Test case for stop_token_ids, no </think> should be in the output
Qwen3_0_6B_Base_STOP_TOKEN_IDS = dict(
    type=TurboMindModel,
    abbr='lmdeploy-qwen3-0_6b-base-stop-token-ids',
    path='Qwen/Qwen3-0.6B-Base',
    engine_config=dict(session_len=4096, max_batch_size=1),
    gen_config=dict(temperature=0.0,
                    do_sample=False,
                    max_new_tokens=1024,
                    stop_token_ids=[151645, 151668]),
    max_seq_len=4096,
    max_out_len=1024,
    batch_size=1,
    run_cfg=dict(num_gpus=1))

# Test case for bad_token_ids, no </think> should be in the output
Qwen3_0_6B_Base_BAD_TOKEN_IDS = dict(
    type=TurboMindModel,
    abbr='lmdeploy-qwen3-0_6b-base-bad-token-ids',
    path='Qwen/Qwen3-0.6B-Base',
    engine_config=dict(session_len=4096, max_batch_size=1),
    gen_config=dict(temperature=0.0,
                    do_sample=False,
                    max_new_tokens=1024,
                    bad_token_ids=[151645, 151668]),
    max_seq_len=4096,
    max_out_len=1024,
    batch_size=1,
    run_cfg=dict(num_gpus=1))

# Test case for logprobs
Qwen3_0_6B_Base_LOGPROBS = dict(type=TurboMindModel,
                                abbr='lmdeploy-qwen3-0_6b-base-logprobs',
                                path='Qwen/Qwen3-0.6B-Base',
                                engine_config=dict(session_len=4096,
                                                   max_batch_size=1),
                                gen_config=dict(temperature=0.0,
                                                do_sample=False,
                                                max_new_tokens=1024,
                                                logprobs=5),
                                max_seq_len=4096,
                                max_out_len=1024,
                                batch_size=1,
                                run_cfg=dict(num_gpus=1))

Qwen3_0_6B_Base_ENDSTR = dict(type=TurboMindModel,
                              abbr='lmdeploy-qwen3-0_6b-base-end-str',
                              path='Qwen/Qwen3-0.6B-Base',
                              engine_config=dict(session_len=4096,
                                                 max_batch_size=1),
                              gen_config=dict(temperature=0.0,
                                              do_sample=False,
                                              max_new_tokens=1024),
                              max_seq_len=4096,
                              max_out_len=1024,
                              batch_size=1,
                              end_str='</think>',
                              run_cfg=dict(num_gpus=1))

models = [
    Qwen3_0_6B_Base, Qwen3_0_6B_Base_PYTORCH, Qwen3_0_6B_Base_BACKEND,
    Qwen3_0_6B_Base_DROP_MIDDLE, Qwen3_0_6B_Base_IGNORE_EOS,
    Qwen3_0_6B_Base_TP2, Qwen3_0_6B_Base_TEMP0, Qwen3_0_6B_Base_DO_SAMPLE,
    Qwen3_0_6B_Base_BAD_WORDS, Qwen3_0_6B_Base_STOP_WORDS,
    Qwen3_0_6B_Base_STOP_TOKEN_IDS, Qwen3_0_6B_Base_BAD_TOKEN_IDS,
    Qwen3_0_6B_Base_NEW_TOKENS, Qwen3_0_6B_Base_MAX_SEQ_LEN,
    Qwen3_0_6B_Base_SESSION_LEN, Qwen3_0_6B_BASE_COMBINED,
    Qwen3_0_6B_Base_LOGPROBS, Qwen3_0_6B_Base_TEMPLATE, Qwen3_0_6B_Base_ENDSTR
]
