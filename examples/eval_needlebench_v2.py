from mmengine.config import read_base
# we use mmengine.config to import other config files

with read_base():
    from opencompass.configs.models.hf_internlm.hf_internlm2_chat_7b import models as internlm2_chat_7b

    # Evaluate needlebench_32k, adjust the configuration to use 4k, 32k, 128k, 200k, or 1000k if necessary.
    # from opencompass.configs.datasets.needlebench_v2.needlebench_v2_32k.needlebench_v2_32k import needlebench_datasets
    # from opencompass.configs.summarizers.needlebench import needlebench_32k_summarizer as summarizer

    # only eval original "needle in a haystack test" in needlebench_32k
    from opencompass.configs.datasets.needlebench_v2.needlebench_v2_32k.needlebench_v2_single_32k import needlebench_zh_datasets, needlebench_en_datasets
    from opencompass.configs.summarizers.needlebench import needlebench_v2_32k_summarizer as summarizer

    # eval Ancestral Tracing Challenge(ATC)
    # from opencompass.configs.datasets.needlebench_v2.atc.atc_0shot_nocot_2_power_en import needlebench_datasets
    # ATC use default summarizer thus no need to import summarizer

datasets = sum([v for k, v in locals().items() if ('datasets' in k)], [])

for m in internlm2_chat_7b:
    m['max_seq_len'] = 32768 # Ensure InternLM2-7B model can receive the full long text; for other models, adjust according to their supported maximum sequence length.
    m['max_out_len'] = 4096

models = internlm2_chat_7b

work_dir = './outputs/needlebench'