from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import Judge_COREV2_Dataset, COREV2Evaluator

base_model_and_result = [{'model':'internlm7b', 'path':'/mnt/petrelfs/caomaosong/opencompass/subject/test/7b/20231130_134927/predictions/internlm-chat-7b-hf-v11/._data_subject_corev2.json'}
]

compare_model_and_result = [{'model':'internlm20b', 'path':'/mnt/petrelfs/caomaosong/opencompass/subject/test/20b/20231130_135138/predictions/internlm-chat-20b-hf/._data_subject_corev2.json'}
]

corev2_reader_cfg = dict(
    input_columns=["prompt"],
    output_column='judge'
    )

corev2_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt="{prompt}"
            ),
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)


judge_corev2_datasets = []
for base in base_model_and_result:
    for compare in compare_model_and_result:
        corev2_eval_cfg = dict(evaluator=dict(type=COREV2Evaluator, base_model=base['model'], compare_model=compare['model'], metric='win_rate'))
        judge_corev2_datasets.append(dict(type=Judge_COREV2_Dataset,
                                          path=base['path'],
                                          path2=compare['path'],
                                          model1=base['model'],
                                          model2=compare['model'],
                                          reader_cfg=corev2_reader_cfg,
                                          infer_cfg=corev2_infer_cfg,
                                          eval_cfg=corev2_eval_cfg)
                                    )
