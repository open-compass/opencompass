from mmengine.config import read_base

from opencompass.models import (HuggingFacewithChatTemplate,
                                TurboMindModelwithChatTemplate)
from opencompass.summarizers import DefaultSubjectiveSummarizer
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

with read_base():
    # read hf models - chat models
    # Dataset
    from opencompass.configs.datasets.chinese_simpleqa.chinese_simpleqa_gen import \
        csimpleqa_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.SimpleQA.simpleqa_gen_0283c3 import \
        simpleqa_datasets  # noqa: F401, E501; noqa: F401, E501
    from opencompass.configs.datasets.subjective.alignbench.alignbench_v1_1_judgeby_critiquellm_new import \
        alignbench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.subjective.alpaca_eval.alpacav2_judgeby_gpt4_new import \
        alpacav2_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.subjective.arena_hard.arena_hard_compare_new import \
        arenahard_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.subjective.compassarena.compassarena_compare_new import \
        compassarena_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.subjective.followbench.followbench_llmeval_new import \
        followbench_llmeval_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.subjective.multiround.mtbench101_judge_new import \
        mtbench101_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.subjective.wildbench.wildbench_pair_judge_new import \
        wildbench_datasets  # noqa: F401, E501

    from ...rjob import infer, sub_eval  # noqa: F401, E501

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')
                and 'mtbench101' not in k and 'wildbench' not in k), [])
datasets += mtbench101_datasets  # noqa: F401, E501
datasets += wildbench_datasets  # noqa: F401, E501

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
    reserved_roles=[dict(role='SYSTEM', api_role='SYSTEM')],
)

hf_model = dict(type=HuggingFacewithChatTemplate,
                abbr='qwen-3-8b-hf-fullbench',
                path='Qwen/Qwen3-8B',
                max_out_len=8192,
                batch_size=8,
                run_cfg=dict(num_gpus=1),
                pred_postprocessor=dict(type=extract_non_reasoning_content))

tm_model = dict(type=TurboMindModelwithChatTemplate,
                abbr='qwen-3-8b-fullbench',
                path='Qwen/Qwen3-8B',
                engine_config=dict(session_len=32768, max_batch_size=1, tp=1),
                gen_config=dict(do_sample=False, enable_thinking=True),
                max_seq_len=32768,
                max_out_len=32768,
                batch_size=1,
                run_cfg=dict(num_gpus=1),
                pred_postprocessor=dict(type=extract_non_reasoning_content))

models = [hf_model, tm_model]

judge_models = [
    dict(type=TurboMindModelwithChatTemplate,
         abbr='qwen-3-8b-judger',
         path='Qwen/Qwen3-8B',
         engine_config=dict(session_len=46000, max_batch_size=1, tp=1),
         gen_config=dict(do_sample=False, enable_thinking=False),
         max_seq_len=46000,
         max_out_len=46000,
         batch_size=1,
         run_cfg=dict(num_gpus=1),
         pred_postprocessor=dict(type=extract_non_reasoning_content))
]

sub_eval['partitioner']['judge_models'] = judge_models
eval = sub_eval

summary_groups = []
summary_groups.append({
    'name': 'compassarena_language',
    'subsets': [
        ['compassarena_language', '内容总结'],
    ],
})
summary_groups.append({
    'name': 'compassarena_knowledge',
    'subsets': [
        ['compassarena_knowledge', '生活常识_ZH'],
    ],
})
summary_groups.append({
    'name': 'compassarena_reason_v2',
    'subsets': [
        ['compassarena_reason_v2', 'reasoning'],
    ],
})
summary_groups.append({
    'name': 'compassarena_math_v2',
    'subsets': [
        ['compassarena_math_v2', '高等数学_ZH'],
    ],
})
summary_groups.append({
    'name': 'compassarena_creationv2_zh',
    'subsets': [
        ['compassarena_creationv2_zh', '内容扩写_ZH'],
    ],
})
summary_groups.append({
    'name':
    'CompassArena',
    'subsets': [
        'compassarena_language',
        'compassarena_knowledge',
        'compassarena_reason_v2',
        'compassarena_math_v2',
        'compassarena_creationv2_zh',
    ],
})
summary_groups.append({
    'name':
    'FoFo',
    'subsets': [['fofo_test_prompts', 'overall'],
                ['fofo_test_prompts_cn', 'overall']],
})
summary_groups.append({
    'name':
    'Followbench',
    'subsets': [
        ['followbench_llmeval_en', 'HSR_AVG'],
        ['followbench_llmeval_en', 'SSR_AVG'],
    ],
})

# Summarizer
summarizer = dict(
    dataset_abbrs=[
        ['alignment_bench_v1_1', '总分'],
        ['alpaca_eval', 'total'],
        ['arenahard', 'score'],
        ['Followbench', 'naive_average'],
        ['CompassArena', 'naive_average'],
        ['FoFo', 'naive_average'],
        ['mtbench101', 'avg'],
        ['wildbench', 'average'],
        ['simpleqa', 'accuracy_given_attempted'],
        ['chinese_simpleqa', 'given_attempted_accuracy'],
        '',
        ['alignment_bench_v1_1', '专业能力'],
        ['alignment_bench_v1_1', '数学计算'],
        ['alignment_bench_v1_1', '基本任务'],
        ['alignment_bench_v1_1', '逻辑推理'],
        ['alignment_bench_v1_1', '中文理解'],
        ['alignment_bench_v1_1', '文本写作'],
        ['alignment_bench_v1_1', '角色扮演'],
        ['alignment_bench_v1_1', '综合问答'],
        ['alpaca_eval', 'helpful_base'],
        ['alpaca_eval', 'koala'],
        ['alpaca_eval', 'oasst'],
        ['alpaca_eval', 'selfinstruct'],
        ['alpaca_eval', 'vicuna'],
        ['compassarena_language', 'naive_average'],
        ['compassarena_knowledge', 'naive_average'],
        ['compassarena_reason_v2', 'naive_average'],
        ['compassarena_math_v2', 'naive_average'],
        ['compassarena_creationv2_zh', 'naive_average'],
        ['fofo_test_prompts', 'overall'],
        ['fofo_test_prompts_cn', 'overall'],
        ['followbench_llmeval_en', 'HSR_AVG'],
        ['followbench_llmeval_en', 'SSR_AVG'],
        ['followbench_llmeval_en', 'HSR_L1'],
        ['followbench_llmeval_en', 'HSR_L2'],
        ['followbench_llmeval_en', 'HSR_L3'],
        ['followbench_llmeval_en', 'HSR_L4'],
        ['followbench_llmeval_en', 'HSR_L5'],
        ['followbench_llmeval_en', 'SSR_L1'],
        ['followbench_llmeval_en', 'SSR_L2'],
        ['followbench_llmeval_en', 'SSR_L3'],
        ['followbench_llmeval_en', 'SSR_L4'],
        ['followbench_llmeval_en', 'SSR_L5'],
        ['simpleqa', 'f1'],
    ],
    type=DefaultSubjectiveSummarizer,
    summary_groups=summary_groups,
)
