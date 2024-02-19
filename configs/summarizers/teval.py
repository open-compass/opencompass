from mmengine.config import read_base

with read_base():
    from .groups.teval import teval_summary_groups

summarizer = dict(
    dataset_abbrs=[
        ['teval', 'naive_average'],
        ['teval-instruct_v1', 'string_metric'],  # 指令跟随能力-string格式
        ['teval-instruct_v1', 'json_metric'],  # 指令跟随能力-json格式
        ['teval-plan_str_v1', 'f1_score'],  # 规划能力-string格式
        ['teval-plan_json_v1', 'f1_score'],  # 规划能力-json格式
        ['teval-reason_str_v1', 'thought'],  # 推理能力-string格式
        ['teval-reason_retrieve_understand_json_v1', 'thought'],  # 推理能力-json格式
        ['teval-retrieve_str_v1', 'name'],  # 检索能力-string格式
        ['teval-reason_retrieve_understand_json_v1', 'name'],  # 检索能力-json格式
        ['teval-understand_str_v1', 'args'],  # 理解能力-string格式
        ['teval-reason_retrieve_understand_json_v1', 'args'],  # 理解能力-json格式
        ['teval-review_str_v1', 'review_quality'],   # 反思能力-string格式

        ['teval_zh', 'naive_average'],
        ['teval-instruct_v1_zh', 'string_metric'],
        ['teval-instruct_v1_zh', 'json_metric'],
        ['teval-plan_str_v1_zh', 'f1_score'],
        ['teval-plan_json_v1_zh', 'f1_score'],
        ['teval-reason_str_v1_zh', 'thought'],
        ['teval-reason_retrieve_understand_json_v1_zh', 'thought'],
        ['teval-retrieve_str_v1_zh', 'name'],
        ['teval-reason_retrieve_understand_json_v1_zh', 'name'],
        ['teval-understand_str_v1_zh', 'args'],
        ['teval-reason_retrieve_understand_json_v1_zh', 'args'],
        ['teval-review_str_v1_zh', 'review_quality'],
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], [])
)
