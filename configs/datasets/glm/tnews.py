from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GLMChoiceInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import TNewsDataset

tnews_reader_cfg = dict(input_columns='sentence', output_column='label_desc2')

tnews_labels = [
    '农业新闻', '旅游新闻', '游戏新闻', '科技类别公司新闻', '体育类别新闻', '初升高教育新闻', '娱乐圈新闻', '投资资讯',
    '军事类别常识', '车辆新闻', '楼市新闻', '环球不含中国类别新闻', '书籍文化历史类别新闻', '故事类别新闻', '股票市场类别新闻'
]

tnews_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template={lb: f'</E></S>这篇新闻属于：{lb}'
                  for lb in tnews_labels},
        column_token_map={'sentence': '</S>'},
        ice_token='</E>'),
    prompt_template=dict(
        type=PromptTemplate,
        template='</E></S>\n以上这篇新闻属于',
        column_token_map={'sentence': '</S>'},
        ice_token='</E>'),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GLMChoiceInferencer, choices=tnews_labels))

tnews_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

tnews_datasets = [
    dict(
        type=TNewsDataset,
        path='json',
        abbr='tnews',
        data_files='./data/FewCLUE/tnews/test_public.json',
        split='train',
        reader_cfg=tnews_reader_cfg,
        infer_cfg=tnews_infer_cfg,
        eval_cfg=tnews_eval_cfg)
]
