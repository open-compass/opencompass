import copy

from opencompass.datasets import WikiBenchDataset
from opencompass.openicl.icl_evaluator import AccEvaluator, CircularEvaluator
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

single_choice_prompts = {
    'single_choice_cn': [
        dict(role='HUMAN',
             prompt='问题: 白色念珠菌常被用作哪种生物的研究模式？\nA. 病毒\nB. 细菌\nC. 真菌\nD. 寄生虫'),
        dict(role='BOT', prompt='回答: C'),
        dict(
            role='HUMAN',
            prompt='问题: 星期五广场（荷兰语：Vrijdagmarkt；荷兰语发音： ）是比利时根特老城的一个城市广场。 星期五广场下方有一个什么设施？\nA. 游乐场\nB. 地下停车场\nC. 公园\nD. 地下商场' # noqa: E501
        ),
        dict(role='BOT', prompt='回答: B'),
        dict(
            role='HUMAN',
            prompt='问题: 尔迪雷·巴斯杜克代表土耳其国家队出场的次数？\nA. 60次\nB. 35次\nC. 49次\nD. 20次'
        ),
        dict(role='BOT', prompt='回答: C'),
        dict(
            role='HUMAN',
            prompt='问题: 陈酆被任命为漳州刺史是因为什么原因？\nA. 朝廷认为他有能力担任该职务\nB. 漳州人怀念陈元光、陈伯珙的政绩\nC. 他是陈伯珙的儿子\nD. 他是陈元光的孙子' # noqa: E501
        ),
        dict(role='BOT', prompt='回答: B'),
        dict(role='HUMAN',
             prompt='问题: 丹徒县在1928年改名为什么？\nA. 苏州市\nB. 润州县\nC. 镇江县\nD. 丹阳县'),
        dict(role='BOT', prompt='回答: C'),
        dict(role='HUMAN', prompt='问题: {question}'),
        dict(role='BOT', prompt='回答: {answer}'),
    ]
}

wikibench_sets = {
    'wiki': ['single_choice_cn'],
}

do_circular = True

wikibench_datasets = []

for _split in list(wikibench_sets.keys()):
    for _name in wikibench_sets[_split]:
        template = {}
        for answer in ['A', 'B', 'C', 'D']:
            one_template_round = copy.deepcopy(single_choice_prompts[_name])
            one_template_round[-1]['prompt'] = one_template_round[-1][
                'prompt'].format(answer=answer)
            template[answer] = dict(round=one_template_round)
        wikibench_infer_cfg = dict(
            prompt_template=dict(type=PromptTemplate, template=template),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=PPLInferencer),
        )
        wikibench_eval_cfg = dict(evaluator=dict(
            type=CircularEvaluator if do_circular else AccEvaluator), )
        wikibench_datasets.append(
            dict(
                type=WikiBenchDataset,
                path='opencompass/WikiBench',
                filename=f'{_name}.jsonl',
                name='circular_' + _name if do_circular else _name,
                abbr='wikibench-' + _split + '-' + _name +
                'circular' if do_circular else '',
                reader_cfg=dict(
                    input_columns=['question'],
                    output_column='answer',
                ),
                infer_cfg=wikibench_infer_cfg,
                eval_cfg=wikibench_eval_cfg,
            ))
