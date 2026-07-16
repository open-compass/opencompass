# ============================================================
# 被测模型配置
# ============================================================
# API 模式模型配置 OpenAISDK | OpenAISDKStreaming
models = [
    dict(
        abbr="intern-s1-pro",
        type="OpenAISDK",
        path="intern-s1-pro",
        openai_api_base="https://chat.intern-ai.org.cn/api/v1/",
        key="sk-y9WPswfdYUG6thXhrL0kzkyUk2pJWotv7SZarLf63MAh5chC",
        query_per_second=1,
        batch_size=2,
        temperature=0.7,
        max_out_len=4096,
        max_seq_len=4096,
        retry=10,
        tokenizer_path="gpt-4"
    )
]

# ==================== 自定义模块导入 ====================
# 导入自定义 evaluator，使其能被注册器找到
# custom_imports = dict(
#     imports=['fake-alignment.fake_alignment_evaluator'],
#     allow_failed_imports=False
# )

# ============================================================
# 自定义数据集配置（由用户提供）
# ============================================================
# 通过 config_str 传入，用户需要提供完整的 datasets 定义
from opencompass.datasets import CustomDataset
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer

# 读取配置
fake_alignment_reader_cfg = dict(
    input_columns=['question'],
    output_column='category'
)

# 推理配置
fake_alignment_infer_cfg = dict(
    prompt_template=dict(type=PromptTemplate, template='{question}'),  # 直接输出问题，让模型自由回答
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=2048)
)

# 评估配置 - 使用自定义评判器
fake_alignment_eval_cfg = dict(
    evaluator=dict(
        type='FakeAlignmentJudgeEvaluator',  # 自定义评估器
        judge_model_cfg=dict(
            type="OpenAISDK",
            path="internlm/Intern-S1",
            openai_api_base="https://chat.intern-ai.org.cn/api/v1/",
            key="sk-3BpA3oDui9rGfsdkNDYq9DtbgPbwyls9BNHPjqruzJ219ln9",
            temperature=0.0,  # 评判任务使用低温度保证稳定性
            max_out_len=512,
            max_seq_len=8192,
            batch_size=8,
            query_per_second=2,
            retry=5,
            meta_template=dict(
                round=[
                    dict(role='HUMAN', api_role='HUMAN'),
                    dict(role='BOT', api_role='BOT', generate=True),
                ],
            ),
        )
    )
)

# 数据集列表
datasets = [
    dict(
        abbr='fake_safety',
        type=CustomDataset,
        path='./data/fake-alignment/safety.jsonl',
        reader_cfg=fake_alignment_reader_cfg,
        infer_cfg=fake_alignment_infer_cfg,
        eval_cfg=fake_alignment_eval_cfg),
    dict(
        abbr='dna_training_set',
        type=CustomDataset,
        path='./data/fake-alignment/dna_training_set.jsonl',
        reader_cfg=fake_alignment_reader_cfg,
        infer_cfg=fake_alignment_infer_cfg,
        eval_cfg=fake_alignment_eval_cfg)
]

# ===================== 评测执行调度配置 =====================
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=8),
    runner=dict(type=LocalRunner, task=dict(type=OpenICLInferTask), max_num_workers=64),
)
eval = dict(
    partitioner=dict(type=NaivePartitioner, n=8),
    runner=dict(
        type=LocalRunner, task=dict(type=OpenICLEvalTask), max_num_workers=32
    ),
)

# ======================== 结果输出 ============================
work_dir = "./outputs/fake-alignment/"