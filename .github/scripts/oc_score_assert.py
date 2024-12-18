import csv
import os

import pytest
import yaml

output_path = 'regression_result_daily'

chat_model_list = [
    'baichuan2-7b-chat-hf',
    'glm-4-9b-chat-hf',
    'glm-4-9b-chat-turbomind',
    'glm-4-9b-chat-vllm',
    'deepseek-7b-chat-hf',
    'deepseek-moe-16b-chat-hf',
    'deepseek-7b-chat-vllm',
    'gemma2-2b-it-hf',
    'gemma2-9b-it-hf',
    'gemma-2b-it-hf',
    'gemma-7b-it-hf',
    'gemma-2-9b-it-turbomind',
    'gemma-7b-it-vllm',
    'internlm2_5-7b-chat-hf',
    'internlm2_5-7b-chat-turbomind',
    'internlm2-chat-1.8b-turbomind',
    'internlm2-chat-1.8b-sft-turbomind',
    'internlm2-chat-7b-lmdeploy',
    'internlm2-chat-7b-sft-turbomind',
    'internlm2-chat-7b-vllm',
    'llama-3_1-8b-instruct-hf',
    'llama-3_2-3b-instruct-hf',
    'llama-3-8b-instruct-hf',
    'llama-3_1-8b-instruct-turbomind',
    'llama-3_2-3b-instruct-turbomind',
    'llama-3-8b-instruct-turbomind',
    'mistral-7b-instruct-v0.2-hf',
    'mistral-7b-instruct-v0.3-hf',
    'mistral-nemo-instruct-2407-hf',
    'mistral-nemo-instruct-2407-turbomind',
    'mistral-7b-instruct-v0.1-vllm',
    'mistral-7b-instruct-v0.2-vllm',
    # 'MiniCPM3-4B-hf', 'minicpm-2b-dpo-fp32-hf', 'minicpm-2b-sft-bf16-hf',
    # 'minicpm-2b-sft-fp32-hf',
    'phi-3-mini-4k-instruct-hf',
    'qwen1.5-0.5b-chat-hf',
    'qwen2-1.5b-instruct-hf',
    'qwen2-7b-instruct-hf',
    'qwen2-1.5b-instruct-turbomind',
    'qwen2-7b-instruct-turbomind',
    'qwen1.5-0.5b-chat-vllm',
    'yi-1.5-6b-chat-hf',
    'yi-1.5-9b-chat-hf',
    'deepseek-v2-lite-chat-hf',
    'internlm2_5-20b-chat-hf',
    'internlm2_5-20b-chat-turbomind',
    'mistral-small-instruct-2409-hf',
    'mistral-small-instruct-2409-turbomind',
    'qwen2.5-14b-instruct-hf',
    'qwen2.5-14b-instruct-turbomind'
]
base_model_list = [
    'glm-4-9b-hf', 'deepseek-moe-16b-base-hf', 'deepseek-7b-base-turbomind',
    'deepseek-moe-16b-base-vllm', 'gemma2-2b-hf', 'gemma2-9b-hf',
    'gemma-2b-hf', 'gemma-7b-hf', 'gemma-2b-vllm', 'gemma-7b-vllm',
    'internlm2_5-7b-hf', 'internlm2-7b-hf', 'internlm2-base-7b-hf',
    'internlm2-1.8b-turbomind', 'internlm2_5-7b-turbomind',
    'internlm2-7b-turbomind', 'internlm2-base-7b-turbomind', 'llama-2-7b-hf',
    'llama-3_1-8b-hf', 'llama-3-8b-hf', 'llama-3.1-8b-turbomind',
    'llama-3-8b-turbomind', 'mistral-7b-v0.2-hf', 'mistral-7b-v0.3-hf',
    'mistral-7b-v0.2-vllm', 'qwen2.5-7b-hf', 'qwen2.5-1.5b-turbomind',
    'qwen2.5-7b-turbomind', 'qwen1.5-moe-a2.7b-hf', 'qwen2-0.5b-hf',
    'qwen2-1.5b-hf', 'qwen2-7b-hf', 'qwen2-1.5b-turbomind',
    'qwen2-7b-turbomind', 'qwen1.5-0.5b-vllm', 'yi-1.5-6b-hf', 'yi-1.5-9b-hf',
    'deepseek-v2-lite-hf', 'internlm2-20b-hf', 'internlm2-base-20b-hf',
    'internlm2-20b-turbomind', 'qwen2.5-14b-hf'
]


@pytest.fixture()
def baseline_scores_testrange(request):
    config_path = os.path.join(
        request.config.rootdir,
        '.github/scripts/oc_score_baseline_testrange.yaml')
    with open(config_path) as f:
        config = yaml.load(f.read(), Loader=yaml.SafeLoader)
    return config


@pytest.fixture()
def baseline_scores(request):
    config_path = os.path.join(request.config.rootdir,
                               '.github/scripts/oc_score_baseline.yaml')
    with open(config_path) as f:
        config = yaml.load(f.read(), Loader=yaml.SafeLoader)
    return config


@pytest.fixture()
def baseline_scores_fullbench(request):
    config_path = os.path.join(
        request.config.rootdir,
        '.github/scripts/oc_score_baseline_fullbench.yaml')
    with open(config_path) as f:
        config = yaml.load(f.read(), Loader=yaml.SafeLoader)
    return config


@pytest.fixture()
def result_scores():
    file = find_csv_files(output_path)
    if file is None:
        return None
    return read_csv_file(file)


@pytest.mark.usefixtures('result_scores')
@pytest.mark.usefixtures('baseline_scores_testrange')
@pytest.mark.chat
class TestChat:
    """Test cases for chat model."""

    @pytest.mark.parametrize(
        'model, dataset', [(p1, p2) for p1 in chat_model_list
                           for p2 in ['gsm8k_accuracy', 'race-high_accuracy']])
    def test_model_dataset_score(self, baseline_scores_testrange,
                                 result_scores, model, dataset):
        base_score = baseline_scores_testrange.get(model).get(dataset)
        result_score = result_scores.get(model).get(dataset)
        assert_score(model, result_score, base_score)


@pytest.mark.usefixtures('result_scores')
@pytest.mark.usefixtures('baseline_scores_testrange')
@pytest.mark.base
class TestBase:
    """Test cases for base model."""

    @pytest.mark.parametrize('model, dataset', [
        (p1, p2) for p1 in base_model_list for p2 in
        ['gsm8k_accuracy', 'GPQA_diamond', 'race-high_accuracy', 'winogrande']
    ])
    def test_model_dataset_score(self, baseline_scores_testrange,
                                 result_scores, model, dataset):
        if model in ['gemma-2b-vllm', 'gemma-7b-vllm'
                     ] and dataset != 'gsm8k_accuracy':
            return
        base_score = baseline_scores_testrange.get(model).get(dataset)
        result_score = result_scores.get(model).get(dataset)
        assert_score(model, result_score, base_score)


@pytest.mark.usefixtures('result_scores')
@pytest.mark.usefixtures('baseline_scores_fullbench')
@pytest.mark.chat_obj_fullbench
class TestChatObjFullbench:
    """Test cases for chat model."""

    @pytest.mark.parametrize('model, dataset', [(p1, p2) for p1 in [
        'internlm2_5-7b-chat-hf_fullbench',
        'internlm2_5-7b-chat-turbomind_fullbench'
    ] for p2 in [
        'race-high_accuracy', 'ARC-c_accuracy', 'BoolQ_accuracy',
        'triviaqa_wiki_1shot_score', 'nq_open_1shot_score',
        'IFEval_Prompt-level-strict-accuracy', 'drop_accuracy',
        'GPQA_diamond_accuracy', 'hellaswag_accuracy', 'TheoremQA_score',
        'musr_average_naive_average', 'korbench_single_naive_average',
        'gsm8k_accuracy', 'math_accuracy', 'cmo_fib_accuracy',
        'aime2024_accuracy', 'wikibench-wiki-single_choice_cncircular_perf_4',
        'sanitized_mbpp_score', 'ds1000_naive_average',
        'lcb_code_generation_pass@1', 'lcb_code_execution_pass@1',
        'lcb_test_output_pass@1', 'bbh-logical_deduction_seven_objects_score',
        'bbh-multistep_arithmetic_two_score', 'mmlu-other_naive_average',
        'cmmlu-china-specific_naive_average', 'mmlu_pro_math_accuracy',
        'ds1000_Pandas_accuracy', 'ds1000_Numpy_accuracy',
        'ds1000_Tensorflow_accuracy', 'ds1000_Scipy_accuracy',
        'ds1000_Sklearn_accuracy', 'ds1000_Pytorch_accuracy',
        'ds1000_Matplotlib_accuracy', 'openai_mmmlu_lite_AR-XY_accuracy',
        'college_naive_average', 'college_knowledge_naive_average'
    ]])
    def test_model_dataset_score(self, baseline_scores_fullbench,
                                 result_scores, model, dataset):
        base_score = baseline_scores_fullbench.get(model).get(dataset)
        result_score = result_scores.get(model).get(dataset)
        assert_score(model, result_score, base_score)


@pytest.mark.usefixtures('result_scores')
@pytest.mark.usefixtures('baseline_scores_fullbench')
@pytest.mark.chat_sub_fullbench
class TestChatSubFullbench:
    """Test cases for chat model."""

    @pytest.mark.parametrize('model, dataset', [(p1, p2) for p1 in [
        'internlm2_5-7b-chat-hf_fullbench',
        'internlm2_5-7b-chat-turbomind_fullbench'
    ] for p2 in [
        'alignment_bench_v1_1_总分', 'alpaca_eval_total', 'arenahard_score',
        'Followbench_naive_average', 'CompassArena_naive_average',
        'mtbench101_avg', 'wildbench_average',
        'simpleqa_accuracy_given_attempted',
        'chinese_simpleqa_given_attempted_accuracy',
        'alignment_bench_v1_1_专业能力', 'alignment_bench_v1_1_数学计算',
        'alignment_bench_v1_1_基本任务', 'alignment_bench_v1_1_逻辑推理',
        'alignment_bench_v1_1_中文理解', 'alignment_bench_v1_1_文本写作',
        'alignment_bench_v1_1_角色扮演', 'alignment_bench_v1_1_综合问答',
        'alpaca_eval_helpful_base', 'compassarena_language_naive_average',
        'compassarena_knowledge_naive_average',
        'compassarena_reason_v2_naive_average',
        'compassarena_math_v2_naive_average',
        'compassarena_creationv2_zh_naive_average',
        'fofo_test_prompts_overall', 'followbench_llmeval_en_HSR_AVG',
        'followbench_llmeval_en_SSR_AVG', 'followbench_llmeval_en_HSR_L1',
        'followbench_llmeval_en_HSR_L2', 'followbench_llmeval_en_HSR_L3',
        'followbench_llmeval_en_HSR_L4', 'followbench_llmeval_en_HSR_L5',
        'followbench_llmeval_en_SSR_L1', 'followbench_llmeval_en_SSR_L2',
        'followbench_llmeval_en_SSR_L3', 'followbench_llmeval_en_SSR_L4',
        'followbench_llmeval_en_SSR_L5', 'simpleqa_f1'
    ]])
    def test_model_dataset_score(self, baseline_scores_fullbench,
                                 result_scores, model, dataset):
        base_score = baseline_scores_fullbench.get(model).get(dataset)
        result_score = result_scores.get(model).get(dataset)
        assert_score(model, result_score, base_score)


@pytest.mark.usefixtures('result_scores')
@pytest.mark.usefixtures('baseline_scores_fullbench')
@pytest.mark.base_fullbench
class TestBaseFullbench:
    """Test cases for chat model."""

    @pytest.mark.parametrize('model, dataset', [(p1, p2) for p1 in [
        'internlm2_5-7b-hf_fullbench', 'internlm2_5-7b-turbomind_fullbench'
    ] for p2 in [
        'race-high_accuracy', 'ARC-c_accuracy', 'BoolQ_accuracy',
        'triviaqa_wiki_1shot_score', 'nq_open_1shot_score', 'drop_accuracy',
        'GPQA_diamond_accuracy', 'hellaswag_accuracy', 'TheoremQA_score',
        'winogrande_accuracy', 'gsm8k_accuracy',
        'GaokaoBench_2010-2022_Math_II_MCQs_score',
        'GaokaoBench_2010-2022_Math_II_Fill-in-the-Blank_score',
        'math_accuracy', 'wikibench-wiki-single_choice_cncircular_perf_4',
        'sanitized_mbpp_score', 'dingo_en_192_score', 'dingo_zh_170_score',
        'mmlu-other_accuracy', 'cmmlu-china-specific_accuracy',
        'mmlu_pro_math_accuracy', 'bbh-logical_deduction_seven_objects_score',
        'bbh-multistep_arithmetic_two_score', 'college_naive_average',
        'college_knowledge_naive_average'
    ]])
    def test_model_dataset_score(self, baseline_scores_fullbench,
                                 result_scores, model, dataset):
        base_score = baseline_scores_fullbench.get(model).get(dataset)
        result_score = result_scores.get(model).get(dataset)
        assert_score(model, result_score, base_score)


@pytest.mark.usefixtures('result_scores')
@pytest.mark.usefixtures('baseline_scores')
@pytest.mark.api
class TestApibench:
    """Test cases for chat model."""

    @pytest.mark.parametrize('model, dataset',
                             [('lmdeploy-api-test', 'race-middle_accuracy'),
                              ('lmdeploy-api-test', 'race-high_accuracy'),
                              ('lmdeploy-api-test', 'gsm8k_accuracy')])
    def test_api(self, baseline_scores, result_scores, model, dataset):
        base_score = baseline_scores.get(model).get(dataset)
        result_score = result_scores.get(model).get(dataset)
        assert_score(model + '_batch', result_score, base_score)


@pytest.mark.usefixtures('result_scores')
@pytest.mark.usefixtures('baseline_scores_fullbench')
@pytest.mark.volc_fullbench
class TestVolcFullbench:
    """Test cases for chat model."""

    @pytest.mark.parametrize('model, dataset', [(
        p1, p2
    ) for p1 in ['internlm2_5-7b-chat-turbomind'] for p2 in [
        'race-high_accuracy', 'ARC-c_accuracy', 'BoolQ_accuracy',
        'triviaqa_wiki_1shot_score', 'nq_open_1shot_score',
        'mmmlu_lite_naive_average', 'IFEval_Prompt-level-strict-accuracy',
        'drop_accuracy', 'bbh_naive_average', 'GPQA_diamond_accuracy',
        'hellaswag_accuracy', 'TheoremQA_score', 'musr_average_naive_average',
        'korbench_single_naive_average',
        'ARC_Prize_Public_Evaluation_accuracy', 'gsm8k_accuracy',
        'GaokaoBench_weighted_average', 'math_accuracy', 'cmo_fib_accuracy',
        'aime2024_accuracy', 'Mathbench_naive_average',
        'wikibench-wiki-single_choice_cncircular_perf_4',
        'cmmlu_naive_average', 'mmlu_naive_average', 'mmlu_pro_naive_average',
        'openai_humaneval_humaneval_pass@1', 'sanitized_mbpp_score',
        'humanevalx_naive_average', 'ds1000_naive_average',
        'lcb_code_generation_pass@1', 'lcb_code_execution_pass@1',
        'lcb_test_output_pass@1', 'bigcodebench_hard_instruct_pass@1',
        'bigcodebench_hard_complete_pass@1', 'teval_naive_average',
        'qa_dingo_cn_score', 'mmlu-stem_naive_average',
        'mmlu-social-science_naive_average', 'mmlu-humanities_naive_average',
        'mmlu-other_naive_average', 'cmmlu-stem_naive_average',
        'cmmlu-social-science_naive_average', 'cmmlu-humanities_naive_average',
        'cmmlu-other_naive_average', 'cmmlu-china-specific_naive_average',
        'mmlu_pro_biology_accuracy', 'mmlu_pro_business_accuracy',
        'mmlu_pro_chemistry_accuracy', 'mmlu_pro_computer_science_accuracy',
        'mmlu_pro_economics_accuracy', 'mmlu_pro_engineering_accuracy',
        'mmlu_pro_health_accuracy', 'mmlu_pro_history_accuracy',
        'mmlu_pro_law_accuracy', 'mmlu_pro_math_accuracy',
        'mmlu_pro_philosophy_accuracy', 'mmlu_pro_physics_accuracy',
        'mmlu_pro_psychology_accuracy', 'mmlu_pro_other_accuracy',
        'humanevalx-python_pass@1', 'humanevalx-cpp_pass@1',
        'humanevalx-go_pass@1', 'humanevalx-java_pass@1',
        'humanevalx-js_pass@1', 'ds1000_Pandas_accuracy',
        'ds1000_Numpy_accuracy', 'ds1000_Tensorflow_accuracy',
        'ds1000_Scipy_accuracy', 'ds1000_Sklearn_accuracy',
        'ds1000_Pytorch_accuracy', 'ds1000_Matplotlib_accuracy',
        'openai_mmmlu_lite_AR-XY_accuracy', 'openai_mmmlu_lite_BN-BD_accuracy',
        'openai_mmmlu_lite_DE-DE_accuracy', 'openai_mmmlu_lite_ES-LA_accuracy',
        'openai_mmmlu_lite_FR-FR_accuracy', 'openai_mmmlu_lite_HI-IN_accuracy',
        'openai_mmmlu_lite_ID-ID_accuracy', 'openai_mmmlu_lite_IT-IT_accuracy',
        'openai_mmmlu_lite_JA-JP_accuracy', 'openai_mmmlu_lite_KO-KR_accuracy',
        'openai_mmmlu_lite_PT-BR_accuracy', 'openai_mmmlu_lite_SW-KE_accuracy',
        'openai_mmmlu_lite_YO-NG_accuracy', 'openai_mmmlu_lite_ZH-CN_accuracy',
        'college_naive_average', 'high_naive_average', 'middle_naive_average',
        'primary_naive_average', 'arithmetic_naive_average',
        'mathbench-a (average)_naive_average',
        'college_knowledge_naive_average', 'high_knowledge_naive_average',
        'middle_knowledge_naive_average', 'primary_knowledge_naive_average',
        'mathbench-t (average)_naive_average'
    ]])
    @pytest.mark.chat_objective
    def test_chat_objective(self, baseline_scores_fullbench, result_scores,
                            model, dataset):
        base_score = baseline_scores_fullbench.get(model).get(dataset)
        result_score = result_scores.get(model).get(dataset)
        assert_score(model + '_batch', result_score, base_score)

    @pytest.mark.parametrize(
        'model, dataset',
        [(p1, p2) for p1 in ['internlm2_5-7b-chat-turbomind']
         for p2 in [
             'alignment_bench_v1_1_总分', 'alpaca_eval_total', 'arenahard_score',
             'Followbench_naive_average', 'CompassArena_naive_average',
             'FoFo_naive_average', 'mtbench101_avg', 'wildbench_average',
             'simpleqa_accuracy_given_attempted',
             'chinese_simpleqa_given_attempted_accuracy',
             'alignment_bench_v1_1_专业能力', 'alignment_bench_v1_1_数学计算',
             'alignment_bench_v1_1_基本任务', 'alignment_bench_v1_1_逻辑推理',
             'alignment_bench_v1_1_中文理解', 'alignment_bench_v1_1_文本写作',
             'alignment_bench_v1_1_角色扮演', 'alignment_bench_v1_1_综合问答',
             'alpaca_eval_helpful_base', 'alpaca_eval_koala',
             'alpaca_eval_oasst', 'alpaca_eval_selfinstruct',
             'alpaca_eval_vicuna', 'compassarena_language_naive_average',
             'compassarena_knowledge_naive_average',
             'compassarena_reason_v2_naive_average',
             'compassarena_math_v2_naive_average',
             'compassarena_creationv2_zh_naive_average',
             'fofo_test_prompts_overall', 'fofo_test_prompts_cn_overall',
             'followbench_llmeval_en_HSR_AVG',
             'followbench_llmeval_en_SSR_AVG', 'followbench_llmeval_en_HSR_L1',
             'followbench_llmeval_en_HSR_L2', 'followbench_llmeval_en_HSR_L3',
             'followbench_llmeval_en_HSR_L4', 'followbench_llmeval_en_HSR_L5',
             'followbench_llmeval_en_SSR_L1', 'followbench_llmeval_en_SSR_L2',
             'followbench_llmeval_en_SSR_L3', 'followbench_llmeval_en_SSR_L4',
             'followbench_llmeval_en_SSR_L5', 'simpleqa_f1'
         ]])
    @pytest.mark.chat_subjective
    def test_chat_subjective(self, baseline_scores_fullbench, result_scores,
                             model, dataset):
        base_score = baseline_scores_fullbench.get(model).get(dataset)
        result_score = result_scores.get(model).get(dataset)
        assert_score(model + '_batch', result_score, base_score)

    @pytest.mark.parametrize('model, dataset', [(
        p1, p2
    ) for p1 in ['internlm2_5-7b-turbomind'] for p2 in [
        'race-high_accuracy', 'ARC-c_accuracy', 'BoolQ_accuracy',
        'triviaqa_wiki_1shot_score', 'nq_open_1shot_score', 'drop_accuracy',
        'bbh_naive_average', 'GPQA_diamond_accuracy', 'hellaswag_accuracy',
        'TheoremQA_score', 'winogrande_accuracy', 'gsm8k_accuracy',
        'GaokaoBench_weighted_average', 'math_accuracy',
        'Mathbench_naive_average',
        'wikibench-wiki-single_choice_cncircular_perf_4',
        'cmmlu_naive_average', 'mmlu_naive_average', 'mmlu_pro_naive_average',
        'openai_humaneval_humaneval_pass@1',
        'openai_humaneval_v2_humaneval_pass@1', 'sanitized_mbpp_score',
        'dingo_en_192_score', 'dingo_zh_170_score', 'mmlu-stem_naive_average',
        'mmlu-social-science_naive_average', 'mmlu-humanities_naive_average',
        'mmlu-other_naive_average', 'cmmlu-stem_naive_average',
        'cmmlu-social-science_naive_average', 'cmmlu-humanities_naive_average',
        'cmmlu-other_naive_average', 'cmmlu-china-specific_naive_average',
        'mmlu_pro_biology_accuracy', 'mmlu_pro_business_accuracy',
        'mmlu_pro_chemistry_accuracy', 'mmlu_pro_computer_science_accuracy',
        'mmlu_pro_economics_accuracy', 'mmlu_pro_engineering_accuracy',
        'mmlu_pro_health_accuracy', 'mmlu_pro_history_accuracy',
        'mmlu_pro_law_accuracy', 'mmlu_pro_math_accuracy',
        'mmlu_pro_philosophy_accuracy', 'mmlu_pro_physics_accuracy',
        'mmlu_pro_psychology_accuracy', 'mmlu_pro_other_accuracy',
        'college_naive_average', 'high_naive_average', 'middle_naive_average',
        'primary_naive_average', 'arithmetic_naive_average',
        'mathbench-a (average)_naive_average',
        'college_knowledge_naive_average', 'high_knowledge_naive_average',
        'middle_knowledge_naive_average', 'primary_knowledge_naive_average',
        'mathbench-t (average)_naive_average'
    ]])
    @pytest.mark.base_objective
    def test_base_objective(self, baseline_scores_fullbench, result_scores,
                            model, dataset):
        base_score = baseline_scores_fullbench.get(model).get(dataset)
        result_score = result_scores.get(model).get(dataset)
        assert_score(model + '_batch', result_score, base_score)

    @pytest.mark.parametrize(
        'model, dataset',
        [(p1, p2) for p1 in ['internlm2_5-7b-turbomind']
         for p2 in [
             'Single-Needle-Retrieval(S-RT)-32000_naive_average',
             'Single-Needle-Retrieval-EN-32000_naive_average',
             'Single-Needle-Retrieval-ZH-32000_naive_average',
             'Single-Needle-Retrieval(S-RT)-100000_naive_average',
             'Single-Needle-Retrieval-EN-100000_naive_average',
             'Single-Needle-Retrieval-ZH-100000_naive_average',
             'Single-Needle-Retrieval(S-RT)-200000_naive_average',
             'Single-Needle-Retrieval-EN-200000_naive_average',
             'Single-Needle-Retrieval-ZH-200000_naive_average',
             'longbench_naive_average', 'longbench_zh_naive_average',
             'longbench_en_naive_average',
             'longbench_single-document-qa_naive_average',
             'longbench_multi-document-qa_naive_average',
             'longbench_summarization_naive_average',
             'longbench_few-shot-learning_naive_average',
             'longbench_synthetic-tasks_naive_average',
             'longbench_code-completion_naive_average'
         ]])
    @pytest.mark.base_long_context
    def test_base_long_context(self, baseline_scores_fullbench, result_scores,
                               model, dataset):
        base_score = baseline_scores_fullbench.get(model).get(dataset)
        result_score = result_scores.get(model).get(dataset)
        assert_score(model + '_batch', result_score, base_score)

    @pytest.mark.parametrize(
        'model, dataset',
        [(p1, p2) for p1 in ['internlm2_5-7b-chat-1m-turbomind']
         for p2 in [
             'ruler_8k_naive_average', 'ruler_32k_naive_average',
             'ruler_128k_naive_average',
             'NeedleBench-Overall-Score-8K_weighted_average',
             'NeedleBench-Overall-Score-32K_weighted_average',
             'NeedleBench-Overall-Score-128K_weighted_average',
             'longbench_naive_average', 'longbench_zh_naive_average',
             'longbench_en_naive_average', 'babilong_0k_naive_average',
             'babilong_4k_naive_average', 'babilong_16k_naive_average',
             'babilong_32k_naive_average', 'babilong_128k_naive_average',
             'babilong_256k_naive_average',
             'longbench_single-document-qa_naive_average',
             'longbench_multi-document-qa_naive_average',
             'longbench_summarization_naive_average',
             'longbench_few-shot-learning_naive_average',
             'longbench_synthetic-tasks_naive_average',
             'longbench_code-completion_naive_average'
         ]])
    @pytest.mark.chat_long_context
    def test_chat_long_context(self, baseline_scores_fullbench, result_scores,
                               model, dataset):
        base_score = baseline_scores_fullbench.get(model).get(dataset)
        result_score = result_scores.get(model).get(dataset)
        assert_score(model + '_batch', result_score, base_score)


@pytest.mark.usefixtures('result_scores')
@pytest.mark.usefixtures('baseline_scores')
class TestCmdCase:

    @pytest.mark.case1
    @pytest.mark.parametrize('model, dataset',
                             [('internlm2_5-7b-hf', 'race-middle_accuracy'),
                              ('internlm2_5-7b-hf', 'race-high_accuracy'),
                              ('internlm2_5-7b-hf', 'demo_gsm8k_accuracy'),
                              ('internlm2-1.8b-hf', 'race-middle_accuracy'),
                              ('internlm2-1.8b-hf', 'race-high_accuracy'),
                              ('internlm2-1.8b-hf', 'demo_gsm8k_accuracy')])
    def test_cmd_case1(self, baseline_scores, result_scores, model, dataset):
        base_score = baseline_scores.get(model).get(dataset)
        result_score = result_scores.get(model).get(dataset)
        assert_score(model, result_score, base_score)

    @pytest.mark.case2
    @pytest.mark.parametrize(
        'model, dataset',
        [('internlm2_5-7b-chat-lmdeploy', 'race-middle_accuracy'),
         ('internlm2_5-7b-chat-lmdeploy', 'race-high_accuracy'),
         ('internlm2_5-7b-chat-lmdeploy', 'demo_gsm8k_accuracy'),
         ('internlm2-chat-1.8b-lmdeploy', 'race-middle_accuracy'),
         ('internlm2-chat-1.8b-lmdeploy', 'race-high_accuracy'),
         ('internlm2-chat-1.8b-lmdeploy', 'demo_gsm8k_accuracy')])
    def test_cmd_case2(self, baseline_scores, result_scores, model, dataset):
        base_score = baseline_scores.get(model).get(dataset)
        result_score = result_scores.get(model).get(dataset)
        assert_score(model + '_batch', result_score, base_score)

    @pytest.mark.case3
    @pytest.mark.parametrize('model, dataset',
                             [('internlm2_5-7b_hf', 'race-middle_accuracy'),
                              ('internlm2_5-7b_hf', 'race-high_accuracy'),
                              ('internlm2_5-7b_hf', 'demo_gsm8k_accuracy')])
    def test_cmd_case3(self, baseline_scores, result_scores, model, dataset):
        base_score = baseline_scores.get(model).get(dataset)
        result_score = result_scores.get(model).get(dataset)
        assert_score(model, result_score, base_score)

    @pytest.mark.case4
    @pytest.mark.parametrize(
        'model, dataset', [('internlm2_5-7b-chat_hf', 'race-middle_accuracy'),
                           ('internlm2_5-7b-chat_hf', 'race-high_accuracy'),
                           ('internlm2_5-7b-chat_hf', 'demo_gsm8k_accuracy')])
    def test_cmd_case4(self, baseline_scores, result_scores, model, dataset):
        base_score = baseline_scores.get(model).get(dataset)
        result_score = result_scores.get(model).get(dataset)
        assert_score(model, result_score, base_score)


THRESHOLD = 3


def assert_score(model_type, score, baseline):
    if score is None or score == '-':
        assert False, 'value is none'

    if 'batch' not in model_type:
        if float(score) <= (baseline + 0.01) and float(score) >= (baseline -
                                                                  0.01):
            print(' '.join([score, 'is equal', str(baseline)]))
            assert True
        else:
            print(' '.join([score, 'is not equal', str(baseline)]))
            assert False, ' '.join([score, 'is not equal', str(baseline)])
    else:
        if float(score) <= (baseline + THRESHOLD) and float(score) >= (
                baseline - THRESHOLD):
            print(' '.join([
                score, 'is between',
                str(baseline - THRESHOLD), 'and',
                str(baseline + THRESHOLD)
            ]))
            assert True
        else:
            print(' '.join([
                score, 'is not etween',
                str(baseline - THRESHOLD), 'and',
                str(baseline + THRESHOLD)
            ]))
            assert False, ' '.join([
                score, 'is not etween',
                str(baseline - THRESHOLD), 'and',
                str(baseline + THRESHOLD)
            ])


def find_csv_files(directory):
    csv_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv') and file.startswith('summary'):
                csv_files.append(os.path.join(root, file))

    csv_files_with_time = {f: os.path.getctime(f) for f in csv_files}
    sorted_csv_files = sorted(csv_files_with_time.items(), key=lambda x: x[1])
    latest_csv_file = sorted_csv_files[-1][0]
    return latest_csv_file


def read_csv_file(file_path):
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        filtered_data = []
        for row in reader:
            if row['metric'] is not None and 'bpb' not in row[
                    'metric'] and '_' != row['metric']:
                filtered_row = row
                filtered_row['dataset'] = row['dataset'] + '_' + row['metric']
                del filtered_row['version']
                del filtered_row['metric']
                del filtered_row['mode']
                filtered_data.append(filtered_row)

    result = {}
    for data in filtered_data:
        dataset = data.get('dataset')
        for key in data.keys():
            if key == 'dataset':
                continue
            else:
                if key in result.keys():
                    result.get(key)[dataset] = data.get(key)
                else:
                    result[key] = {dataset: data.get(key)}
    return result
