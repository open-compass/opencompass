from mmengine.config import read_base

with read_base():
    from .arxivmath_0826_llmjudge_rawprompt_gen import \
        arxivmath_0826_datasets  # noqa: F401, F403
    from .arxivmath_0926_llmjudge_rawprompt_gen import \
        arxivmath_0926_datasets  # noqa: F401, F403
    from .arxivmath_1026_llmjudge_rawprompt_gen import \
        arxivmath_1026_datasets  # noqa: F401, F403

arxivmath_08_to_10_datasets = (arxivmath_0826_datasets + arxivmath_0926_datasets + arxivmath_1026_datasets)
