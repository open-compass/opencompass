from mmengine.config import read_base

from opencompass.models import TurboMindModelwithChatTemplate
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

with read_base():
    from autotest.infer.base_datasets import datasets
    from autotest.infer.constant import meta_template as test_meta_template

datasets = [x for x in datasets.copy() if 'InfiniteBench' not in x['abbr']]

# Base model testcase
Qwen3_0_6B_FP8 = dict(
    type=TurboMindModelwithChatTemplate,
    abbr='lmdeploy-qwen3-0_6b-fp8-base',
    path='Qwen/Qwen3-0.6B-FP8',
    engine_config=dict(max_batch_size=1, session_len=128000),
    gen_config=dict(do_sample=False),
    max_out_len=32768,
    batch_size=1,
    run_cfg=dict(num_gpus=1),
    pred_postprocessor=dict(type=extract_non_reasoning_content))

# Test case for PyTorch backend
Qwen3_0_6B_FP8_PYTORCH = dict(type=TurboMindModelwithChatTemplate,
                              abbr='lmdeploy-qwen3-0_6b-fp8-pytorch',
                              path='Qwen/Qwen3-0.6B-FP8',
                              engine_config=dict(backend='pytorch',
                                                 session_len=128000,
                                                 max_batch_size=1),
                              gen_config=dict(do_sample=False),
                              max_seq_len=128000,
                              max_out_len=1024,
                              batch_size=1,
                              run_cfg=dict(num_gpus=1))

# Test case for backend should be same as PyTorch backend
Qwen3_0_6B_FP8_BACKEND = dict(type=TurboMindModelwithChatTemplate,
                              abbr='lmdeploy-qwen3-0_6b-fp8-backend',
                              path='Qwen/Qwen3-0.6B-FP8',
                              engine_config=dict(session_len=128000,
                                                 max_batch_size=1),
                              gen_config=dict(do_sample=False),
                              max_seq_len=128000,
                              max_out_len=1024,
                              batch_size=1,
                              backend='pytorch',
                              run_cfg=dict(num_gpus=1))

# test case for ignore_eos, which is used in some models
Qwen3_0_6B_FP8_IGNORE_EOS = dict(type=TurboMindModelwithChatTemplate,
                                 abbr='lmdeploy-qwen3-0_6b-fp8-ignore-eos',
                                 path='Qwen/Qwen3-0.6B-FP8',
                                 engine_config=dict(session_len=4096,
                                                    max_batch_size=1),
                                 gen_config=dict(do_sample=False,
                                                 max_new_tokens=128,
                                                 ignore_eos=True),
                                 max_seq_len=4096,
                                 batch_size=1,
                                 run_cfg=dict(num_gpus=1))

Qwen3_0_6B_FP8_TEMP0 = dict(type=TurboMindModelwithChatTemplate,
                            abbr='lmdeploy-qwen3-0_6b-fp8-temp0',
                            path='Qwen/Qwen3-0.6B-FP8',
                            engine_config=dict(session_len=4096,
                                               max_batch_size=1),
                            gen_config=dict(temperature=0.0, do_sample=False),
                            max_seq_len=32768,
                            max_out_len=1024,
                            batch_size=1,
                            run_cfg=dict(num_gpus=1))

Qwen3_0_6B_FP8_BAD_WORDS = dict(type=TurboMindModelwithChatTemplate,
                                abbr='lmdeploy-qwen3-0_6b-fp8-bad-words',
                                path='Qwen/Qwen3-0.6B-FP8',
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

Qwen3_0_6B_FP8_TP2 = dict(type=TurboMindModelwithChatTemplate,
                          abbr='lmdeploy-qwen3-0_6b-fp8-tp2',
                          path='Qwen/Qwen3-0.6B-FP8',
                          engine_config=dict(session_len=4096,
                                             max_batch_size=1,
                                             tp=2),
                          gen_config=dict(temperature=0.0, do_sample=False),
                          max_out_len=1024,
                          batch_size=1,
                          run_cfg=dict(num_gpus=2))

Qwen3_0_6B_FP8_SESSION_LEN = dict(type=TurboMindModelwithChatTemplate,
                                  abbr='lmdeploy-qwen3-0_6b-fp8-session-len',
                                  path='Qwen/Qwen3-0.6B-FP8',
                                  engine_config=dict(session_len=10,
                                                     max_batch_size=1),
                                  gen_config=dict(temperature=0.0,
                                                  do_sample=False),
                                  max_seq_len=32768,
                                  max_out_len=8192,
                                  batch_size=1,
                                  run_cfg=dict(num_gpus=1))

# Test case for max_new_tokens and min_new_tokens,
# which should generate between 90 and 100 tokens
Qwen3_0_6B_FP8_NEW_TOKENS = dict(type=TurboMindModelwithChatTemplate,
                                 abbr='lmdeploy-qwen3-0_6b-fp8-new-tokens',
                                 path='Qwen/Qwen3-0.6B-FP8',
                                 engine_config=dict(session_len=4096,
                                                    max_batch_size=1),
                                 gen_config=dict(temperature=0.0,
                                                 do_sample=False,
                                                 min_new_tokens=90,
                                                 max_new_tokens=100),
                                 max_seq_len=32768,
                                 batch_size=1,
                                 run_cfg=dict(num_gpus=1))

# Test case for max_seq_len and max_out_len
Qwen3_0_6B_FP8_MAX_SEQ_LEN = dict(type=TurboMindModelwithChatTemplate,
                                  abbr='lmdeploy-qwen3-0_6b-fp8-max-seq-len',
                                  path='Qwen/Qwen3-0.6B-FP8',
                                  engine_config=dict(max_batch_size=1),
                                  gen_config=dict(do_sample=False),
                                  max_seq_len=200,
                                  max_out_len=100,
                                  batch_size=1,
                                  run_cfg=dict(num_gpus=1))

# Test case for stop_words, no stop tokens should be in the output
# for example, ' to '
Qwen3_0_6B_FP8_STOP_WORDS = dict(
    type=TurboMindModelwithChatTemplate,
    abbr='lmdeploy-qwen3-0_6b-fp8-stop-words',
    path='Qwen/Qwen3-0.6B-FP8',
    engine_config=dict(session_len=4096, max_batch_size=1),
    gen_config=dict(temperature=0.0, do_sample=False),
    max_seq_len=4096,
    max_out_len=4096,
    batch_size=1,
    stop_words=[' and', '</think>', ' to', '\n\n', 'Question:', 'Answer:'],
    run_cfg=dict(num_gpus=1))

Qwen3_0_6B_FP8_TEMPLATE = dict(type=TurboMindModelwithChatTemplate,
                               abbr='lmdeploy-qwen3-0_6b-fp8-template',
                               path='Qwen/Qwen3-0.6B-FP8',
                               engine_config=dict(session_len=32768,
                                                  max_batch_size=1),
                               gen_config=dict(do_sample=False,
                                               max_new_tokens=256),
                               max_seq_len=32768,
                               batch_size=1,
                               meta_template=test_meta_template,
                               run_cfg=dict(num_gpus=1))

Qwen3_0_6B_FP8_DROP_MIDDLE = dict(type=TurboMindModelwithChatTemplate,
                                  abbr='lmdeploy-qwen3-0_6b-fp8-drop-middle',
                                  path='Qwen/Qwen3-0.6B-FP8',
                                  engine_config=dict(session_len=32768,
                                                     max_batch_size=1),
                                  gen_config=dict(do_sample=False),
                                  max_seq_len=2048,
                                  max_out_len=2000,
                                  batch_size=1,
                                  drop_middle=True,
                                  run_cfg=dict(num_gpus=1))

Qwen3_0_6B_FP8_HF_OVER = dict(
    type=TurboMindModelwithChatTemplate,
    abbr='lmdeploy-qwen3-0_6b-fp8-hf-over',
    path='Qwen/Qwen3-0.6B-FP8',
    engine_config=dict(
        session_len=32768,
        max_batch_size=1,
        hf_overrides=dict(
            rope_scaling=dict(rope_type='yarn',
                              factor=4.0,
                              original_max_position_embeddings=32768))),
    gen_config=dict(top_p=0.9,
                    temperature=0.7,
                    do_sample=False,
                    max_new_tokens=1024),
    max_seq_len=128000,
    batch_size=1,
    run_cfg=dict(num_gpus=1))

# Additional test cases for various top_k values
Qwen3_0_6B_FP8_TOPK50 = dict(type=TurboMindModelwithChatTemplate,
                             abbr='lmdeploy-qwen3-0_6b-fp8-topk50',
                             path='Qwen/Qwen3-0.6B-FP8',
                             engine_config=dict(session_len=4096,
                                                max_batch_size=1),
                             gen_config=dict(top_k=50,
                                             temperature=1.2,
                                             max_new_tokens=1024),
                             max_seq_len=4096,
                             max_out_len=1024,
                             batch_size=1,
                             run_cfg=dict(num_gpus=1))

# Test case for combined parameters
Qwen3_0_6B_FP8_COMBINED = dict(type=TurboMindModelwithChatTemplate,
                               abbr='lmdeploy-qwen3-0_6b-fp8-combined',
                               path='Qwen/Qwen3-0.6B-FP8',
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
Qwen3_0_6B_FP8_DO_SAMPLE = dict(type=TurboMindModelwithChatTemplate,
                                abbr='lmdeploy-qwen3-0_6b-fp8-do-sample',
                                path='Qwen/Qwen3-0.6B-FP8',
                                engine_config=dict(session_len=4096,
                                                   max_batch_size=1),
                                gen_config=dict(do_sample=True,
                                                max_new_tokens=1024),
                                max_seq_len=4096,
                                max_out_len=1024,
                                batch_size=1,
                                run_cfg=dict(num_gpus=1))

# Test case for stop_token_ids, no </think> should be in the output
Qwen3_0_6B_FP8_STOP_TOKEN_IDS = dict(
    type=TurboMindModelwithChatTemplate,
    abbr='lmdeploy-qwen3-0_6b-fp8-stop-token-ids',
    path='Qwen/Qwen3-0.6B-FP8',
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
Qwen3_0_6B_FP8_BAD_TOKEN_IDS = dict(
    type=TurboMindModelwithChatTemplate,
    abbr='lmdeploy-qwen3-0_6b-fp8-bad-token-ids',
    path='Qwen/Qwen3-0.6B-FP8',
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
Qwen3_0_6B_FP8_LOGPROBS = dict(type=TurboMindModelwithChatTemplate,
                               abbr='lmdeploy-qwen3-0_6b-fp8-logprobs',
                               path='Qwen/Qwen3-0.6B-FP8',
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

models = [
    Qwen3_0_6B_FP8,
    Qwen3_0_6B_FP8_PYTORCH,
    Qwen3_0_6B_FP8_BACKEND,
    Qwen3_0_6B_FP8_DROP_MIDDLE,
    Qwen3_0_6B_FP8_IGNORE_EOS,
    Qwen3_0_6B_FP8_TP2,
    Qwen3_0_6B_FP8_TEMP0,
    Qwen3_0_6B_FP8_TOPK50,
    Qwen3_0_6B_FP8_DO_SAMPLE,
    Qwen3_0_6B_FP8_BAD_WORDS,
    Qwen3_0_6B_FP8_STOP_WORDS,
    Qwen3_0_6B_FP8_STOP_TOKEN_IDS,
    Qwen3_0_6B_FP8_BAD_TOKEN_IDS,
    Qwen3_0_6B_FP8_NEW_TOKENS,
    Qwen3_0_6B_FP8_MAX_SEQ_LEN,
    Qwen3_0_6B_FP8_SESSION_LEN,
    Qwen3_0_6B_FP8_LOGPROBS,
    Qwen3_0_6B_FP8_TEMPLATE,
    Qwen3_0_6B_FP8_HF_OVER,
    Qwen3_0_6B_FP8_COMBINED,
]
