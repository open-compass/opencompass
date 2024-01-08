from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import EMEvaluator
from opencompass.datasets import dropDataset

drop_reader_cfg = dict(
    input_columns=['prompt', 'question'],
    output_column='answers',
    train_split='validation',
    test_split='validation',
)

drop_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='''\
Text: In the county, the population was spread out with 23.50% under the age of 18, 8.70% from 18 to 24, 29.70% from 25 to 44, 24.70% from 45 to 64, and 13.30% who were 65 years of age or older.
Question: How many more percent are under the age of 18 compared to the 18 to 24 group?
Answer: According to the text, 23.5% are under the age of 18, and 8.7% are from ages 18 to 24. 23.5%-8.7%=14.8%. So the answer is 14.8.

Text: Playing in their second straight Thanksgiving game, the Eagles struggled especially on defense, where they were unable to stop the much-hyped Lions offense. The worst of it all was how unproven rookie Eric Rowe was tasked with covering wide receiver Calvin Johnson, leading to Johnson catching 3 touchdowns. Stafford’s five passing touchdowns, including three of them to Johnson was too much for the Eagles to overcome and for the second consecutive time this season, the Eagles gave up 45 points in a game. With the loss, the Eagles drop to 4-7 on the season and 6-1 when playing on Thanksgiving.
Question: How many TD passes did Stafford throw other than to Johnson?
Answer: According to the text, Stafford threw 5 TD passes, 3 of which were to Johnson. 5-3=2. So the answer is 2.

Text: {prompt}
Question: {question}
Answer:'''),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

drop_eval_cfg = dict(
    evaluator=dict(type=EMEvaluator), pred_postprocessor=dict(
        type='gsm8k'))  # use the same processor to find answer

drop_datasets = [
    dict(
        abbr='drop',
        type=dropDataset,
        path='./data/drop/drop_dataset_dev.json',
        reader_cfg=drop_reader_cfg,
        infer_cfg=drop_infer_cfg,
        eval_cfg=drop_eval_cfg)
]
