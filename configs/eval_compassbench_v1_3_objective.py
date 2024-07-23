from copy import deepcopy
from mmengine.config import read_base
from lagent.agents.react import ReActProtocol
from opencompass.models.lagent import CodeAgent
from opencompass.lagent.actions.ipython_interpreter import IPythonInterpreter
from opencompass.lagent.agents.react import CIReAct

from opencompass.runners import LocalRunner
from opencompass.partitioners import (
    SizePartitioner,
    NaivePartitioner,
    InferTimePartitioner,
    NumWorkerPartitioner,
)
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask


with read_base():
    from .models.qwen.lmdeploy_qwen2_7b_instruct import (
        models as qwen2_7b_instruct_model,
    )  # Qwen2-7B-Instruct
    from .datasets.compassbench_v1_3.compassbench_v1_3_objective_gen_068af0 import (
        compassbench_aug_datasets,
    )
    # from .datasets.compassbench_20_v1_1.agent.cibench_template_gen_e6b12a import (
    #     cibench_datasets,
    # )
    # from .datasets.compassbench_20_v1_1.agent.mus_teval_gen_105c48 import (
    #     plugin_eval_datasets,
    # )

_origin_models = sum(
    [v for k, v in locals().items() if k.endswith("_model") or k == "_origin_models"],
    [],
)

work_dir = 'outputs/compassbench_v1_3/objective'
# infer['partitioner']['num_worker'] = 16
# # infer['runner']['max_num_workers'] = 32
# infer['runner']['retry'] = 0
# eval['runner']['retry'] = 0

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=8),
    runner=dict(type=LocalRunner, task=dict(type=OpenICLInferTask)),
)
# infer = dict(
#     partitioner=dict(type=SizePartitioner),
#     runner=dict(
#         type=LocalRunner,
#         task=dict(type=OpenICLInferTask),
#     ),
# )
eval = dict(
    partitioner=dict(type=NaivePartitioner, n=8),
    runner=dict(type=LocalRunner, task=dict(type=OpenICLEvalTask)),
)

CIBENCH_DATASET_NAMES = ["cibench_datasets"]
MUS_PLUGINEVAL_DATASET_NAMES = ["plugin_eval_datasets"]

# ---------------------------------------- VANILLA BEGIN ----------------------------------------
# remove system round
_naive_datasets = sum(
    [
        v
        for k, v in locals().items()
        if (k.endswith("_datasets") or k == "datasets")
        and (k not in CIBENCH_DATASET_NAMES)
    ],
    [],
)
_naive_models = []
for m in _origin_models:
    m = deepcopy(m)
    if "meta_template" in m and "round" in m["meta_template"]:
        round = m["meta_template"]["round"]
        if any(r["role"] == "SYSTEM" for r in round):
            new_round = [r for r in round if r["role"] != "SYSTEM"]
            print(
                f'WARNING: remove SYSTEM round in meta_template for {m.get("abbr", None)}'
            )
            m["meta_template"]["round"] = new_round
    _naive_models.append(m)
# ---------------------------------------- VANILLA END ----------------------------------------


model_dataset_combinations, models, datasets = [], [], []
if _naive_datasets:
    model_dataset_combinations.append(
        dict(models=_naive_models, datasets=_naive_datasets)
    )
    models.extend(_naive_models)
    datasets.extend(_naive_datasets)
# if _cibench_agent_datasets:
#     model_dataset_combinations.append(dict(models=_cibench_agent_models, datasets=_cibench_agent_datasets))
#     models.extend(_cibench_agent_models)
#     datasets.extend(_cibench_agent_datasets)
