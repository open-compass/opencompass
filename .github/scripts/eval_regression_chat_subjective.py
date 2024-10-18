from mmengine.config import read_base

from opencompass.models import OpenAISDK
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks.subjective_eval import SubjectiveEvalTask
from opencompass.summarizers import SubjectiveSummarizer

with read_base():
    # choose a list of datasets
    from opencompass.configs.datasets.subjective.compassbench.compassbench_datasets import \
        compassarena_datasets
    from opencompass.configs.datasets.subjective.fofo.fofo_bilingual_judge import \
        fofo_datasets
    from opencompass.configs.datasets.subjective.multiround.mtbench101_judge import \
        mtbench101_datasets
    # read hf models - chat models
    from opencompass.configs.models.hf_internlm.hf_internlm2_5_7b_chat import \
        models as hf_internlm2_5_7b_chat_model  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_5_7b_chat import \
        models as lmdeploy_internlm2_5_7b_chat_model  # noqa: F401, E501
    

models = sum([v for k, v in locals().items() if k.endswith('_model')], [])
datasets = sum([v for k, v in locals().items() if k.endswith('_datasets')], [])

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
    reserved_roles=[dict(role='SYSTEM', api_role='SYSTEM')],
)

for d in datasets:
    d['reader_cfg']['test_range'] = '[0:50]'


api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
    reserved_roles=[dict(role='SYSTEM', api_role='SYSTEM')],
)


judge_models = [dict(
        abbr='lmdeploy-api-test',
        type=OpenAISDK,
        key='EMPTY',
        openai_api_base='http://judgemodel:10001/v1',
        path='compass_judger_internlm2_102b_0508',
        tokenizer_path='internlm/internlm2_5-20b-chat',
        rpm_verbose=True,
        meta_template=api_meta_template,
        query_per_second=50,
        max_out_len=1024,
        max_seq_len=4096,
        temperature=0.01,
        batch_size=128,
        retry=3,
    )]

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

summarizer = dict(type=SubjectiveSummarizer, function='subjective')
