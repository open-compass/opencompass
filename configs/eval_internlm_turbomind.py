from mmengine.config import read_base
from opencompass.models.turbomind import TurboMindModel


with read_base():
    # choose a list of datasets
    from .datasets.mmlu.mmlu_gen_a484b3 import mmlu_datasets
    from .datasets.ceval.ceval_gen_5f30c7 import ceval_datasets
    from .datasets.SuperGLUE_WiC.SuperGLUE_WiC_gen_d06864 import WiC_datasets
    from .datasets.triviaqa.triviaqa_gen_2121ce import triviaqa_datasets
    from .datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    from .datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets
    # and output the results in a choosen format
    from .summarizers.medium import summarizer

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

# # config for internlm-7b model
internlm_7b = dict(
        type=TurboMindModel,
        abbr='internlm-7b-turbomind',
        path='internlm/internlm-7b',
        engine_config=dict(session_len=2048,
                           max_batch_size=32,
                           rope_scaling_factor=1.0),
        gen_config=dict(top_k=1,
                        top_p=0.8,
                        temperature=1.0,
                        max_new_tokens=100),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=32,
        concurrency=32,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )

# config for internlm-20b model
internlm_20b = dict(
        type=TurboMindModel,
        abbr='internlm-20b-turbomind',
        path='internlm/internlm-20b',
        engine_config=dict(session_len=2048,
                           max_batch_size=8,
                           rope_scaling_factor=1.0),
        gen_config=dict(top_k=1, top_p=0.8,
                        temperature=1.0,
                        max_new_tokens=100),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        concurrency=8,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )

models = [internlm_20b]
