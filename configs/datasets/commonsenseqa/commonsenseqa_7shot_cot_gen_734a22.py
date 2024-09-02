from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import commonsenseqaDataset
from opencompass.utils.text_postprocessors import (
    match_answer_pattern,
)

commonsenseqa_reader_cfg = dict(
    input_columns=['question', 'A', 'B', 'C', 'D', 'E'],
    output_column='answerKey',
    test_split='validation',
)

_ice_template = dict(
    type=PromptTemplate,
    template=dict(
        begin='</E>',
        round=[
            dict(
                role='HUMAN',
                prompt='Q: What do people use to absorb extra ink from a fountain pen? Answer Choices: A.shirt pocket B.calligrapher’s hand C.inkwell D.desk drawer E.blotter',
            ),
            dict(
                role='BOT',
                prompt='A: The answer must be an item that can absorb ink. Of the above choices, only blotters are used to absorb ink. So the answer is E.',
            ),
            dict(
                role='HUMAN',
                prompt='Q: What home entertainment equipment requires cable?Answer Choices: A.radio shack B.substation C.television D.cabinet',
            ),
            dict(
                role='BOT',
                prompt='A: The answer must require cable. Of the above choices, only television requires cable. So the answer is C.',
            ),
            dict(
                role='HUMAN',
                prompt='Q: The fox walked from the city into the forest, what was it looking for? Answer Choices: A.pretty flowers B.hen house C.natural habitat D.storybook',
            ),
            dict(
                role='BOT',
                prompt='A: The answer must be something in the forest. Of the above choices, only natural habitat is in the forest. So the answer is B.',
            ),
            dict(
                role='HUMAN',
                prompt='Q: Sammy wanted to go to where the people were. Where might he go? Answer Choices: A.populated areas B.race track C.desert D.apartment E.roadblock',
            ),
            dict(
                role='BOT',
                prompt='A: The answer must be a place with a lot of people. Of the above choices, only populated areas have a lot of people. So the answer is A.',
            ),
            dict(
                role='HUMAN',
                prompt='Q: Where do you put your grapes just before checking out? Answer Choices: A.mouth B.grocery cart Csuper market D.fruit basket E.fruit market',
            ),
            dict(
                role='BOT',
                prompt='A: The answer should be the place where grocery items are placed before checking out. Of the above choices, grocery cart makes the most sense for holding grocery items. So the answer is B.',
            ),
            dict(
                role='HUMAN',
                prompt='Q: Google Maps and other highway and street GPS services have replaced what? Answer Choices: A.united states B.mexico C.countryside D.atlas',
            ),
            dict(
                role='BOT',
                prompt='A: The answer must be something that used to do what Google Maps and GPS services do, which is to give directions. Of the above choices, only atlases are used to give directions. So the answer is D.',
            ),
            dict(
                role='HUMAN',
                prompt='Q: Before getting a divorce, what did the wife feel who was doing all the work? Answer Choices: A.harder B.anguish C.bitterness D.tears E.sadness',
            ),
            dict(
                role='BOT',
                prompt='A: The answer should be the feeling of someone getting divorced who was doing all the work. Of the above choices, the closest feeling is bitterness. So the answer is C.',
            ),
            dict(
                role='HUMAN',
                prompt='Q:{question}  Answer Choices: A. {A}\nB. {B}\nC. {C}\nD. {D}\nE. {E}\nA:',
            ),
            dict(
                role='BOT',
                prompt='{answerKey}',
            ),
        ],
    ),
    ice_token='</E>',
)

commonsenseqa_infer_cfg = dict(
    ice_template=_ice_template,
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

commonsenseqa_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_postprocessor=dict(
        type=match_answer_pattern, answer_pattern=r'(?i)so the answer is\s*([A-P])'
    ),
)


commonsenseqa_datasets = [
    dict(
        abbr='commonsense_qa',
        type=commonsenseqaDataset,
        path='opencompass/commonsense_qa',
        reader_cfg=commonsenseqa_reader_cfg,
        infer_cfg=commonsenseqa_infer_cfg,
        eval_cfg=commonsenseqa_eval_cfg,
    )
]

del _ice_template
