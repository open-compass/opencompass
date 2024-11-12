from copy import deepcopy

from mmengine.config import read_base

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
    from opencompass.configs.models.hf_internlm.hf_internlm2_5_7b_chat import \
        models as hf_internlm2_5_7b_chat_model  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_5_7b_chat import \
        models as lmdeploy_internlm2_5_7b_chat_model  # noqa: F401, E501

summarizer = dict(type=SubjectiveSummarizer, function='subjective')

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

models = sum([v for k, v in locals().items() if k.endswith('_model')], [])
for m in models:
    m['abbr'] = m['abbr'] + '_fullbench'
    if 'turbomind' in m['abbr'] or 'lmdeploy' in m['abbr']:
        m['engine_config']['max_batch_size'] = 1
        m['batch_size'] = 1

models = sorted(models, key=lambda x: x['run_cfg']['num_gpus'])

judge_models = deepcopy([models[1]])
judge_models[0]['abbr'] = judge_models[0]['abbr'] + '-judge'

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
