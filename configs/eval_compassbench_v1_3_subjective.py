from mmengine.config import read_base

with read_base():
    from .datasets.subjective.compassbench.compassbench_checklist import (
        checklist_datasets,
    )

    # INTERNAL SETTINGS
    from .internal.aliyun_llmeval_xiaolinchen import infer_num_worker as infer
    from .internal.opensource_model_compass_bench_v1_3 import models as _origin_models
    from .internal.closesource_model_compass_bench_v1_3 import (
        ai_360gpt_pro,
        minimax_abab6_5_chat,
        baichuan4_api,
        step_1_8k,
        moonshot_v1_8k,
        qwen_max_0428,
        gpt4o_20240513,
        gpt_4o_mini_20240718,
        ernie_4_0_8k_preview_0518,
        glm4_0520,
        mistral_large,
        yi_large_api,
        xunfei_spark_v4_ultra,
        claude3_5_sonnet_alles,
        doubao_pro_4k_240515,
        hunyuan,
        mistral_large_instruct_2407,
        deepseek_api,
        gpt4_1106,
        xunfei_spark_v3_5_max,
    )
from opencompass.partitioners import (
    SizePartitioner,
    NaivePartitioner,
    NumWorkerPartitioner,
)
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask
from opencompass.summarizers.subjective.compassbench_v13 import CompassBenchSummarizer

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
    reserved_roles=[dict(role='SYSTEM', api_role='SYSTEM')],
)

_origin_models = sum(
    [v for k, v in locals().items() if k.endswith('_model') or k == '_origin_models'],
    [],
)

######### API Models #########
_origin_models += [minimax_abab6_5_chat]  # minimax_abab6_5_chat
_origin_models += [baichuan4_api]  # baichuan4_api
_origin_models += [step_1_8k]  # step_1_8k
_origin_models += [moonshot_v1_8k]  # moonshot_v1_8k
_origin_models += [qwen_max_0428]  # qwen_max_0428
_origin_models += [gpt4o_20240513]  # gpt4o_20240513
_origin_models += [gpt_4o_mini_20240718]  # gpt_4o_mini_20240718
_origin_models += [ernie_4_0_8k_preview_0518]  # ernie_4_0_8k_preview_0518
_origin_models += [mistral_large]  # mistral_large
_origin_models += [yi_large_api]  # yi_large_api
_origin_models += [claude3_5_sonnet_alles]  # claude3_5_sonnet_alles
_origin_models += [doubao_pro_4k_240515]  # doubao_pro_4k_240515
_origin_models += [hunyuan]  # hunyuan
_origin_models += [mistral_large_instruct_2407]
_origin_models += [deepseek_api]
_origin_models += [glm4_0520]  # glm4_0520D
_origin_models += [xunfei_spark_v4_ultra]  # xunfei_spark_v4_ultra
_origin_models += [xunfei_spark_v3_5_max]  # xunfei_spark_v3_5_max

###### Reference model ######
_origin_models += [gpt4_1106]


models = _origin_models
# -------------Inference Stage ----------------------------------------
# For subjective evaluation, we often set do sample for models
datasets = [*checklist_datasets]
######## LOCAL RUNNER #########
infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=8),
    runner=dict(type=LocalRunner, task=dict(type=OpenICLInferTask)),
)

infer['partitioner']['num_worker'] = 4
infer['runner']['max_num_workers'] = 32
infer['runner']['retry'] = 0

# -------------Evalation Stage ----------------------------------------
## ------------- JudgeLLM Configuration
judge_models = [
    # True GPT4-O
    gpt4o_20240513
]
## ------------- Evaluation Configuration
eval = dict(
    partitioner=dict(
        type=SubjectiveNaivePartitioner,
        models=models,
        judge_models=judge_models,
    ),
    runner=dict(
        type=LocalRunner, max_num_workers=16, task=dict(type=SubjectiveEvalTask)
    ),
)
summarizer = dict(type=CompassBenchSummarizer)
work_dir = 'outputs/compassbench_v1_3/subjective'
