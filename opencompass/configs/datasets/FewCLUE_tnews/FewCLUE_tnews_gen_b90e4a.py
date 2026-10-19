from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import TNewsDatasetV2
from opencompass.utils.text_postprocessors import first_capital_postprocess

tnews_reader_cfg = dict(
    input_columns='sentence',
    output_column='label_desc2',
)

tnews_labels = [
    '农业新闻',  # news_agriculture
    '旅游新闻',  # news_travel
    '游戏新闻',  # news_game
    '科技类别公司新闻',  # news_tech
    '体育类别新闻',  # news_sports
    '初升高教育新闻',  # news_edu
    '娱乐圈新闻',  # news_entertainment
    '投资资讯',  # news_finance
    '军事类别常识',  # news_military
    '车辆新闻',  # news_car
    '楼市新闻',  # news_house
    '环球不含中国类别新闻',  # news_world
    '书籍文化历史类别新闻',  # news_culture
    '故事类别新闻',  # news_story
    '股票市场类别新闻',  # news_stock
]
_tnews_options_list_str = '\n'.join(f'{chr(ord("A") + i)}. {tnews_labels[i]}'
                                    for i in range(len(tnews_labels)))
_tnews_options_range_str = '，'.join(f'“{chr(ord("A") + i)}”'
                                    for i in range(len(tnews_labels)))

tnews_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt=
                f'{{sentence}}\n请判断上述内容属于什么新闻？\n{_tnews_options_list_str}\n请从{_tnews_options_range_str}中进行选择。\n答：',
            ),
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

tnews_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_capital_postprocess),
)

tnews_datasets = [
    dict(
        abbr='tnews-dev',
        type=TNewsDatasetV2,
        path='./data/FewCLUE/tnews/dev_few_all.json',
        reader_cfg=tnews_reader_cfg,
        infer_cfg=tnews_infer_cfg,
        eval_cfg=tnews_eval_cfg,
    ),
    dict(
        abbr='tnews-test',
        type=TNewsDatasetV2,
        path='./data/FewCLUE/tnews/test_public.json',
        reader_cfg=tnews_reader_cfg,
        infer_cfg=tnews_infer_cfg,
        eval_cfg=tnews_eval_cfg,
    ),
]

del _tnews_options_list_str, _tnews_options_range_str
