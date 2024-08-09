from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import CircularEvaluator, AccEvaluator
from opencompass.datasets import WikiBenchDataset
from opencompass.utils.text_postprocessors import first_option_postprocess

# ----------------------- Prompt Content----------------------- #
zero_shot_prompts = {
    'single_choice_prompts': [
        dict(role='HUMAN', prompt='以下是一道单项选择题，请你根据你了解的知识给出正确的答案选项。\n下面是你要回答的题目：：\n{question}\n答案选项：'),
        dict(role='BOT', prompt='{answer}')
    ]
}
few_shot_prompts = {
    'single_choice_prompts': {
        'single_choice_cn': [
            dict(role='HUMAN', prompt='题目：“一丝不苟”中的“丝”的本意是（  ）。\nA. 计量单位\nB. 丝线\nC. 丝绸\nD. 发丝'),
            dict(role='BOT', prompt='答案：A'),
            dict(role='HUMAN', prompt='题目：五华县体育场位于哪个省？\nA. 湖南省\nB. 浙江省\nC. 广东省\nD. 江苏省'),
            dict(role='BOT', prompt='答案：C'),
            dict(role='HUMAN', prompt='题目：“西施犬的原产地是哪里？\nA. 印度\nB. 中国\nC. 西藏\nD. 台湾'),
            dict(role='BOT', prompt='答案：C'),
            dict(role='HUMAN', prompt='题目：四库全书的四库是指什么？\nA. 易、书、诗、礼\nB. 经、史、子、音\nC. 诗、书、音、律\nD. 经、史、子、集'),
            dict(role='BOT', prompt='答案：D'),
            dict(role='HUMAN', prompt='题目：{question}'),
        ]}
}


# ----------------------- Prompt Template----------------------- #

# Use Zero-Shot or not
with_few_shot = True

# Max for this dataset is 4, should be set with `with_few_shot`
few_shot_samples = 4

# Use circular evaluation or not
with_circular_eval = True

single_choice_prompts = zero_shot_prompts['single_choice_prompts'] if not with_few_shot else few_shot_prompts['single_choice_prompts']

# Set few shot prompt number
if with_few_shot:
    assert few_shot_samples > 0
    for _name in list(single_choice_prompts.keys()):
        single_choice_prompts[_name] = single_choice_prompts[_name][- few_shot_samples * 2 - 2:]

compassbench_v1_knowledge_sets = {
    'common_knowledge': ['single_choice_cn'],
    'humanity': ['single_choice_cn'],
    'natural_science': ['single_choice_cn'],
    'social_science': ['single_choice_cn'],
}


# ----------------------- Dataset Config----------------------- #
compassbench_v1_knowledge_datasets = []

for _split in list(compassbench_v1_knowledge_sets.keys()):
    for _name in compassbench_v1_knowledge_sets[_split]:
        compassbench_v1_knowledge_reader_cfg = dict(input_columns=['question'], output_column='answer')

        compassbench_v1_knowledge_infer_cfg = dict(
            ice_template=dict(
                type=PromptTemplate,
                template=dict(
                    begin='</E>',
                    round=single_choice_prompts[_name]
                ),
                ice_token='</E>',
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer),
        )
        compassbench_v1_knowledge_eval_cfg = dict(
            evaluator=dict(type=CircularEvaluator if with_circular_eval else AccEvaluator),
            pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'),
        )

        compassbench_v1_knowledge_datasets.append(
            dict(
                type=WikiBenchDataset,
                abbr='compassbench_v1_knowledge-' + _split + '-' + _name + '_' + 'circular' if with_circular_eval else '',
                path=f'data/compassbench_v1.1/knowledge/{_split}/{_name}.jsonl',
                name=_name + '_circular' if with_circular_eval else _name,
                reader_cfg=compassbench_v1_knowledge_reader_cfg,
                infer_cfg=compassbench_v1_knowledge_infer_cfg,
                eval_cfg=compassbench_v1_knowledge_eval_cfg,
            )
        )


from opencompass.datasets import TriviaQADatasetV3, TriviaQAEvaluator

triviaqa_and_nq_reader_cfg = dict(input_columns=['question'], output_column='answer')

triviaqa_and_nq_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='Question: When do stores stop selling alcohol in indiana?'),
                dict(role='BOT', prompt='Answer: 3 a.m.'),
                dict(role='HUMAN', prompt='Question: Edinburgh of the Seven Seas is the capital of which group of islands?'),
                dict(role='BOT', prompt='Answer: Tristan da Cunha'),
                dict(role='HUMAN', prompt='Question: Which book of the Christian Bible\'s new testament comprises a letter from St Paul to members of a church that he had founded at Macedonia?'),
                dict(role='BOT', prompt='Answer: Philippians'),
                dict(role='HUMAN', prompt='Question: The Hindu deity Hanuman appears in the form of which animal?'),
                dict(role='BOT', prompt='Answer: A monkey'),
                dict(role='HUMAN', prompt='Question: Who hosts the ITV quiz show The Chase?'),
                dict(role='BOT', prompt='Answer: Bradley Walsh'),
                dict(role='HUMAN', prompt='Question: {question}'),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=50, stopping_criteria=['Question:']),
)

triviaqa_and_nq_eval_cfg = dict(evaluator=dict(type=TriviaQAEvaluator), pred_role='BOT')

compassbench_v1_knowledge_datasets.append(
    dict(
        type=TriviaQADatasetV3,
        abbr='compassbench_v1_knowledge-mixed-cloze_en',
        path='data/compassbench_v1.1/knowledge/mixed/cloze_en.jsonl',
        reader_cfg=triviaqa_and_nq_reader_cfg,
        infer_cfg=triviaqa_and_nq_infer_cfg,
        eval_cfg=triviaqa_and_nq_eval_cfg
    )
)
