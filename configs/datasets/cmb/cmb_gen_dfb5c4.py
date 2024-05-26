from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import CMBDataset
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.utils.text_postprocessors import multiple_select_postprocess


cmb_datasets = []
for split in ['val', 'test']:
    cmb_reader_cfg = dict(
        input_columns=['exam_type', 'exam_class', 'question_type', 'question', 'option_str'],
        output_column='answer',
        train_split=split,
        test_split=split,
    )

    cmb_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(
                        role='HUMAN',
                        prompt=f'以下是中国{{exam_type}}中{{exam_class}}考试的一道{{question_type}}，不需要做任何分析和解释，直接输出答案选项。\n{{question}}\n{{option_str}} \n 答案: ',
                    ),
                    dict(role='BOT', prompt='{answer}'),
                ],
            ),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer, max_out_len=10),
    )

    cmb_eval_cfg = dict(
        evaluator=dict(type=AccEvaluator),
        pred_postprocessor=dict(type=multiple_select_postprocess),
    )

    cmb_datasets.append(
        dict(
            abbr='cmb' if split == 'val' else 'cmb_test',
            type=CMBDataset,
            path='./data/CMB/',
            reader_cfg=cmb_reader_cfg,
            infer_cfg=cmb_infer_cfg,
            eval_cfg=cmb_eval_cfg,
        )
    )
