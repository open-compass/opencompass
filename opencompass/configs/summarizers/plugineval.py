from mmengine.config import read_base

with read_base():
    from .groups.plugineval import plugineval_summary_groups

summarizer = dict(
    dataset_abbrs=[
        ['plugin_eval', 'naive_average'],
        ['plugin_eval-instruct_v1', 'string_metric'],  # 指令跟随能力-string格式
        ['plugin_eval-instruct_v1', 'json_metric'],  # 指令跟随能力-json格式
        ['plugin_eval-plan_str_v1', 'f1_score'],  # 规划能力-string格式
        ['plugin_eval-plan_json_v1', 'f1_score'],  # 规划能力-json格式
        ['plugin_eval-reason_str_v1', 'thought'],  # 推理能力-string格式
        ['plugin_eval-reason_retrieve_understand_json_v1', 'thought'],  # 推理能力-json格式
        ['plugin_eval-retrieve_str_v1', 'name'],  # 检索能力-string格式
        ['plugin_eval-reason_retrieve_understand_json_v1', 'name'],  # 检索能力-json格式
        ['plugin_eval-understand_str_v1', 'args'],  # 理解能力-string格式
        ['plugin_eval-reason_retrieve_understand_json_v1', 'args'],  # 理解能力-json格式
        ['plugin_eval-review_str_v1', 'review_quality'],   # 反思能力-string格式

        ['plugin_eval_zh', 'naive_average'],
        ['plugin_eval-instruct_v1_zh', 'string_metric'],
        ['plugin_eval-instruct_v1_zh', 'json_metric'],
        ['plugin_eval-plan_str_v1_zh', 'f1_score'],
        ['plugin_eval-plan_json_v1_zh', 'f1_score'],
        ['plugin_eval-reason_str_v1_zh', 'thought'],
        ['plugin_eval-reason_retrieve_understand_json_v1_zh', 'thought'],
        ['plugin_eval-retrieve_str_v1_zh', 'name'],
        ['plugin_eval-reason_retrieve_understand_json_v1_zh', 'name'],
        ['plugin_eval-understand_str_v1_zh', 'args'],
        ['plugin_eval-reason_retrieve_understand_json_v1_zh', 'args'],
        ['plugin_eval-review_str_v1_zh', 'review_quality'],
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], [])
)
