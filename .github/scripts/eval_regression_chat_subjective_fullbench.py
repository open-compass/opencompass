from mmengine.config import read_base

from opencompass.models import OpenAISDK
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.summarizers import SubjectiveSummarizer
from opencompass.tasks.subjective_eval import SubjectiveEvalTask

with read_base():
    # read hf models - chat models
    # Dataset
    from opencompass.configs.datasets.subjective.alignbench.alignbench_v1_1_judgeby_critiquellm import \
        alignbench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.subjective.alpaca_eval.alpacav2_judgeby_gpt4 import \
        alpacav2_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.subjective.arena_hard.arena_hard_compare import \
        arenahard_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.subjective.compassarena.compassarena_compare import \
        compassarena_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.subjective.fofo.fofo_bilingual_judge import \
        fofo_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.subjective.followbench.followbench_llmeval import \
        followbench_llmeval_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.subjective.multiround.mtbench101_judge import \
        mtbench101_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.subjective.wildbench.wildbench_pair_judge import \
        wildbench_datasets  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_5_7b_chat import \
        models as lmdeploy_internlm2_5_7b_chat_model  # noqa: F401, E501

summarizer = dict(type=SubjectiveSummarizer, function='subjective')

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')
                and 'mtbench101' not in k and 'wildbench' not in k), [])
datasets += mtbench101_datasets  # noqa: F401, E501
datasets += wildbench_datasets  # noqa: F401, E501

for d in datasets:
    d['reader_cfg']['test_range'] = '[0:20]'

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
    reserved_roles=[dict(role='SYSTEM', api_role='SYSTEM')],
)

models = [(dict(
    abbr='lmdeploy-api-test',
    type=OpenAISDK,
    key='EMPTY',
    openai_api_base='http://localhost:23333/v1',
    path='compass_judger_internlm2_5-7b-chat',
    tokenizer_path='internlm/internlm2_5-7b-chat',
    rpm_verbose=True,
    meta_template=api_meta_template,
    query_per_second=50,
    max_out_len=1024,
    max_seq_len=4096,
    temperature=0.01,
    batch_size=128,
    retry=3,
))]

judge_models = models
eval = dict(
    partitioner=dict(
        type=SubjectiveNaivePartitioner,
        models=models,
        judge_models=judge_models,
    ),
    runner=dict(type=LocalRunner,
                max_num_workers=16,
                task=dict(type=SubjectiveEvalTask)),
)
