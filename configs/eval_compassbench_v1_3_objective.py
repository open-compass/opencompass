from copy import deepcopy
from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import (
    NaivePartitioner,
    NumWorkerPartitioner,
)
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    from .datasets.compassbench_v1_3.compassbench_v1_3_math import (
        compassbench_math_datasets,
    )
    from .datasets.compassbench_v1_3.compassbench_v1_3_knowledge import (
        compassbench_knowledge_datasets,
    )
    from .datasets.compassbench_v1_3.compassbench_v1_3_code_gen_c8c3aa import (
        compassbench_v1_3_code_datasets,
    )
    from .datasets.compassbench_20_v1_1.agent.mus_teval_gen_105c48 import (
        plugin_eval_datasets,
    )
    from .datasets.compassbench_v1_3.compassbench_v1_3_prompt import (
        FORCE_STOP_PROMPT_EN,
        FEWSHOT_INSTRUCTION,
        IPYTHON_INTERPRETER_DESCRIPTION,
    )
    from .summarizers.compassbench_v1_3_objective import summarizer

    # INTERNAL SETTINGS
    from .internal.aliyun_llmeval_xiaolinchen import infer_num_worker as infer
    from .internal.opensource_model_compass_bench_v1_3 import models as _origin_models
    from .internal.closesource_model_compass_bench_v1_3 import (
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
        xunfei_spark_v3_5_max,
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

work_dir = 'outputs/compassbench_v1_3/objective'

####### LOCAL RUNNER #########
infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=8),
    runner=dict(type=LocalRunner, task=dict(type=OpenICLInferTask)),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner, n=4),
    runner=dict(type=LocalRunner, task=dict(type=OpenICLEvalTask)),
)
infer['partitioner']['num_worker'] = 4
infer['runner']['max_num_workers'] = 32
infer['runner']['retry'] = 0
eval['runner']['retry'] = 0

MUS_PLUGINEVAL_DATASET_NAMES = ['plugin_eval_datasets']

# ---------------------------------------- VANILLA BEGIN ----------------------------------------
# remove system round
_naive_datasets = sum(
    [v for k, v in locals().items() if (k.endswith('_datasets') or k == 'datasets')],
    [],
)
_naive_models = []
for m in _origin_models:
    m = deepcopy(m)
    if 'meta_template' in m and 'round' in m['meta_template']:
        round = m['meta_template']['round']
        if any(r['role'] == 'SYSTEM' for r in round):
            new_round = [r for r in round if r['role'] != 'SYSTEM']
            print(
                f'WARNING: remove SYSTEM round in meta_template for {m.get("abbr", None)}'
            )
            m['meta_template']['round'] = new_round
    _naive_models.append(m)

model_dataset_combinations, models, datasets = [], [], []
if _naive_datasets:
    model_dataset_combinations.append(
        dict(models=_naive_models, datasets=_naive_datasets)
    )
    models.extend(_naive_models)
    datasets.extend(_naive_datasets)
