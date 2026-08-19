from mmengine.config import read_base

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.benbench import BenBenchDataset, BenbenEvaluator
from opencompass.models import HuggingFaceBaseModel

with read_base():
    from .internal.clusters.aliyun_llmeval_fattn2 import infer_num_worker as infer
    # from .internal.clusters.local import infer_num_worker as infer
    from .internal.clusters.local import eval

infer['runner']['max_num_workers'] = 80
infer['partitioner']['num_worker'] = 48
infer['partitioner']['num_split'] = 8

model_settings = [
    ['internlm2-7b-hf', 'internlm/internlm2-7b', 8, 1],
    ['internlm2-base-7b-hf', 'internlm/internlm2-base-7b', 8, 1],
    ['internlm2.5-7b-hf', 'internlm/internlm2_5-7b', 8, 1],
    ['qwen-7b-hf', 'Qwen/Qwen-7B', 8, 1],
    ['qwen1.5-7b-hf', 'Qwen/Qwen1.5-7B', 8, 1],
    ['qwen2-7b-hf', 'Qwen/Qwen2-7B', 8, 1],
]

dataset_settings = [
    ['GSM8K-origin-test', 'original/GSM8K-origin-test.jsonl'],
    ['GSM8K-origin-train', 'original/GSM8K-origin-train.jsonl'],
    ['MATH-origin-test', 'original/MATH-origin-test.jsonl'],
    ['MATH-origin-train', 'original/MATH-origin-train.jsonl'],
    ['MATH-origin-prm800k-500-test', 'original/MATH-origin-prm800k-500-test.jsonl'],
    ['MMLU-origin-dev', 'original/MMLU-origin-dev.jsonl'],
    ['MMLU-origin-val', 'original/MMLU-origin-val.jsonl'],
    ['MMLU-origin-test', 'original/MMLU-origin-test.jsonl'],
    ['GPQA_DIAMOND-origin-test', 'original/GPQA_DIAMOND-origin-test.jsonl'],
    ['HUMANEVAL-origin-test', 'original/HUMANEVAL-origin-test.jsonl'],
    ['IFEVAL-origin-test', 'original/IFEVAL-origin-test.jsonl'],
    ['TRIVIAQA-origin-test', 'original/TRIVIAQA-origin-test.jsonl'],
    ['TRIVIAQA-origin-train', 'original/TRIVIAQA-origin-train.jsonl'],

    ['GSM8K_rewritten-test-1', 'rewritten/GSM8K_rewritten-test-1.jsonl'],
    ['GSM8K_rewritten-test-2', 'rewritten/GSM8K_rewritten-test-2.jsonl'],
    ['GSM8K_rewritten-test-3', 'rewritten/GSM8K_rewritten-test-3.jsonl'],
    ['GSM8K_rewritten-train-1', 'rewritten/GSM8K_rewritten-train-1.jsonl'],
    ['GSM8K_rewritten-train-2', 'rewritten/GSM8K_rewritten-train-2.jsonl'],
    ['GSM8K_rewritten-train-3', 'rewritten/GSM8K_rewritten-train-3.jsonl'],
    ['MATH_rewritten-test-1', 'rewritten/MATH_rewritten-test-1.jsonl'],
    ['MATH_rewritten-test-2', 'rewritten/MATH_rewritten-test-2.jsonl'],
    ['MATH_rewritten-test-3', 'rewritten/MATH_rewritten-test-3.jsonl'],
    ['MATH_rewritten-train-1', 'rewritten/MATH_rewritten-train-1.jsonl'],
    ['MATH_rewritten-train-2', 'rewritten/MATH_rewritten-train-2.jsonl'],
    ['MATH_rewritten-train-3', 'rewritten/MATH_rewritten-train-3.jsonl'],
    ['MATH_rewritten-prm800k-500-test-1', 'rewritten/MATH_rewritten-prm800k-500-test-1.jsonl'],
    ['MATH_rewritten-prm800k-500-test-2', 'rewritten/MATH_rewritten-prm800k-500-test-2.jsonl'],
    ['MATH_rewritten-prm800k-500-test-3', 'rewritten/MATH_rewritten-prm800k-500-test-3.jsonl'],

]

n_gram = 20  # <------------------------------------------------------------------------------------------------------------------------------------------------------------------ n-gram
datasets = []
models = []
model_dataset_combinations = []

for model_abbr, model_path, batch_size, num_gpu in model_settings:
    _tmp_datasets = []
    for dataset_abbr, dataset_path in dataset_settings:
        _tmp_datasets.append(
            dict(
                abbr=dataset_abbr + f'-{n_gram}gram',
                type=BenBenchDataset,
                num_gram=n_gram,
                path=f'data/benbench/{dataset_path}',
                tokenizer_path=model_path,
                reader_cfg=dict(input_columns=['prompt'], output_column='reference'),
                infer_cfg=dict(
                    prompt_template=dict(type=PromptTemplate, template='{prompt}'),
                    retriever=dict(type=ZeroRetriever),
                    inferencer=dict(type=GenInferencer, max_out_len=n_gram)
                ),
                eval_cfg=dict(evaluator=dict(type=BenbenEvaluator))
            )
        )

    model = dict(
        type=HuggingFaceBaseModel,
        abbr=model_abbr,
        path=model_path,
        max_out_len=n_gram,
        batch_size=batch_size,
        run_cfg=dict(num_gpus=num_gpu),
    )

    model_dataset_combinations.append(dict(models=[model], datasets=_tmp_datasets))
    models.append(model)
    datasets.extend(_tmp_datasets)

work_dir = f'outputs/debug/benbench'

summarizer = dict(
    dataset_abbrs=[
        [f'GSM8K-origin-train-{n_gram}gram', 'exact_match'],
        [f'GSM8K-origin-test-{n_gram}gram', 'exact_match'],
        [f'GSM8K_rewritten-train-{n_gram}gram', 'exact_match'],
        [f'GSM8K_rewritten-test-{n_gram}gram', 'exact_match'],
        '',
        [f'MATH-origin-train-{n_gram}gram', 'exact_match'],
        [f'MATH-origin-test-{n_gram}gram', 'exact_match'],
        [f'MATH-origin-prm800k-500-test-{n_gram}gram', 'exact_match'],
        [f'MATH_rewritten-train-{n_gram}gram', 'exact_match'],
        [f'MATH_rewritten-test-{n_gram}gram', 'exact_match'],
        [f'MATH_rewritten-prm800k-500-test-{n_gram}gram', 'exact_match'],
        '',
        [f'MMLU-origin-dev-{n_gram}gram', 'exact_match'],
        [f'MMLU-origin-val-{n_gram}gram', 'exact_match'],
        [f'MMLU-origin-test-{n_gram}gram', 'exact_match'],
        '',
        [f'GPQA_DIAMOND-origin-test-{n_gram}gram', 'exact_match'],
        '',
        [f'HUMANEVAL-origin-test-{n_gram}gram', 'exact_match'],
        '',
        [f'IFEVAL-origin-test-{n_gram}gram', 'exact_match'],
        '',
        [f'TRIVIAQA-origin-train-{n_gram}gram', 'exact_match'],
        [f'TRIVIAQA-origin-test-{n_gram}gram', 'exact_match'],
    ],
    summary_groups=[
        {'name': f'GSM8K_rewritten-test-{n_gram}gram', 'subsets': [f'GSM8K_rewritten-test-1-{n_gram}gram', f'GSM8K_rewritten-test-2-{n_gram}gram', f'GSM8K_rewritten-test-3-{n_gram}gram']},
        {'name': f'GSM8K_rewritten-train-{n_gram}gram', 'subsets': [f'GSM8K_rewritten-train-1-{n_gram}gram', f'GSM8K_rewritten-train-2-{n_gram}gram', f'GSM8K_rewritten-train-3-{n_gram}gram']},
        {'name': f'MATH_rewritten-test-{n_gram}gram', 'subsets': [f'MATH_rewritten-test-1-{n_gram}gram', f'MATH_rewritten-test-2-{n_gram}gram', f'MATH_rewritten-test-3-{n_gram}gram']},
        {'name': f'MATH_rewritten-train-{n_gram}gram', 'subsets': [f'MATH_rewritten-train-1-{n_gram}gram', f'MATH_rewritten-train-2-{n_gram}gram', f'MATH_rewritten-train-3-{n_gram}gram']},
        {'name': f'MATH_rewritten-prm800k-500-test-{n_gram}gram', 'subsets': [f'MATH_rewritten-prm800k-500-test-1-{n_gram}gram', f'MATH_rewritten-prm800k-500-test-2-{n_gram}gram', f'MATH_rewritten-prm800k-500-test-3-{n_gram}gram']},
    ]
)

# dataset                                 version    metric       mode    internlm2-7b-hf    internlm2-base-7b-hf    internlm2.5-7b-hf    qwen-7b-hf    qwen1.5-7b-hf    qwen2-7b-hf
# --------------------------------------  ---------  -----------  ------  -----------------  ----------------------  -------------------  ------------  ---------------  -------------
# GSM8K-origin-train-20gram               3bf459     exact_match  gen     18.06              1.49                    2.29                 12.60         55.02            47.09
# GSM8K-origin-test-20gram                3bf459     exact_match  gen     3.18               1.26                    2.30                 3.67          14.97            10.71
# GSM8K_rewritten-train-20gram            -          exact_match  gen     1.53               1.01                    1.51                 3.16          4.79             5.11
# GSM8K_rewritten-test-20gram             -          exact_match  gen     1.18               0.96                    1.48                 2.85          4.21             4.48
#                                         -          -            -       -                  -                       -                    -             -                -
# MATH-origin-train-20gram                3bf459     exact_match  gen     27.75              3.93                    6.27                 33.81         64.41            52.95
# MATH-origin-test-20gram                 3bf459     exact_match  gen     8.75               3.89                    5.83                 26.42         47.13            11.51
# MATH-origin-prm800k-500-test-20gram     3bf459     exact_match  gen     8.49               3.51                    5.78                 6.95          13.66            11.14
# MATH_rewritten-train-20gram             -          exact_match  gen     9.94               3.16                    4.86                 10.99         15.07            11.91
# MATH_rewritten-test-20gram              -          exact_match  gen     5.76               3.15                    4.52                 9.52          12.08            6.31
# MATH_rewritten-prm800k-500-test-20gram  -          exact_match  gen     6.01               3.32                    4.96                 4.70          7.02             6.36
#                                         -          -            -       -                  -                       -                    -             -                -
# MMLU-origin-dev-20gram                  3bf459     exact_match  gen     0.63               0.56                    0.63                 0.49          0.56             0.21
# MMLU-origin-val-20gram                  3bf459     exact_match  gen     0.34               0.33                    0.40                 0.21          0.48             0.39
# MMLU-origin-test-20gram                 3bf459     exact_match  gen     0.30               0.22                    0.37                 0.21          0.47             0.43
#                                         -          -            -       -                  -                       -                    -             -                -
# GPQA_DIAMOND-origin-test-20gram         3bf459     exact_match  gen     2.12               2.63                    2.63                 2.32          2.32             2.42
#                                         -          -            -       -                  -                       -                    -             -                -
# HUMANEVAL-origin-test-20gram            3bf459     exact_match  gen     1.46               1.83                    2.20                 2.68          3.54             10.00
#                                         -          -            -       -                  -                       -                    -             -                -
# IFEVAL-origin-test-20gram               3bf459     exact_match  gen     1.18               1.00                    1.18                 0.78          0.96             0.85
#                                         -          -            -       -                  -                       -                    -             -                -
# TRIVIAQA-origin-train-20gram            3bf459     exact_match  gen     15.76              0.02                    13.40                0.83          10.95            5.02
# TRIVIAQA-origin-test-20gram             3bf459     exact_match  gen     11.35              0.05                    10.46                0.33          8.11             3.71

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# This configuration file uses the BenBench [1] dataset and its approach for data contamination detection. However, the experiments corresponding to this config have significant differences from BenBench.

# This config uses `n_gram=20` to more strictly determine whether a sample has been exposed to an LLM during training. Additionally, although this config measures both the original and rewritten texts, we did not focus much on their differences.

# The `prm800k-500-test` is divided according to the settings in [2].

# Based on the above results, we have the following possible conclusions:

# 1. Compared to internlm2-7b, internlm2-base-7b almost has no issues with data contamination.
# 2. internlm2-7b, qwen-7b, qwen1.5-7b, and qwen2-7b have a high probability of being trained on the GSM8K/MATH training sets. Moreover, qwen-7b and qwen1.5-7b are also suspected of being trained on the MATH test set, but this is likely due to the setting in [2].
# 3. Under the strict setting of 20-gram, we believe that directly observing without rewritten data can also determine whether a dataset has been exposed during training. For example, qwen2-7b on humaneval and internlm2-7b, internlm2.5-7b, and qwen1.5-7b on triviaqa are highly suspected of data leakage.

# Additionally, I have the following questions:

# 1. Since BenBench's division is based on tokenized token sequences, different tokenizers will result in inconsistent text lengths corresponding to the 20-gram, making it difficult to compare numbers between different models. Particularly for the GSM8K/MATH math datasets, internlm assigns a token to each number in the range $[0, 10k)$, while qwen only has tokens for characters in the range $[0, 10)$, with larger numbers obtained by concatenation. What impact might this have on the conclusions?
# 2. Even if some data was exposed during training, it might not be in the way we expected. For example, chat models will use chat_template as the format, which is inconsistent with our use of the dataset's original text. How should chat models be evaluated? Additionally, the prompt for the training data in base models may vary, so might the exact_match values we measure be underestimated due to this mismatch?

# [1] Xu, R., Wang, Z., Fan, R.-Z., & Liu, P. (2024). Benchmarking Benchmark Leakage in Large Language Models. arXiv preprint arXiv:2404.18824.
# [2] Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., & Cobbe, K. (2023). Let's Verify Step by Step. arXiv preprint arXiv:2305.20050.
