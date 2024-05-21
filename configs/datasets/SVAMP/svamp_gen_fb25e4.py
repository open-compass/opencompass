from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import SVAMPDataset, gsm8k_postprocess, Gsm8kEvaluator

svamp_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt="Question: There are 87 oranges and 290 bananas in Philip's collection. If the bananas are organized into 2 groups and oranges are organized into 93 groups How big is each group of bananas?\nLet's think step by step\nAnswer:"),
                dict(role='BOT', prompt='To find the size of each group of bananas, we divide the total number of bananas (290) by the number of groups (2): 290 / 2 = 145. Therefore, each group of bananas contains 145 bananas. The answer is 145.\n'),
                dict(role='HUMAN', prompt="Question: Marco and his dad went strawberry picking. Marco's dad's strawberries weighed 11 pounds. If together their strawberries weighed 30 pounds. How much did Marco's strawberries weigh?\nLet's think step by step\nAnswer:"),
                dict(role='BOT', prompt="To find Marco's strawberries' weight, we subtract his dad's strawberries' weight (11 pounds) from the total weight of their strawberries (30 pounds): 30 - 11 = 19. Therefore, Marco's strawberries weighed 19 pounds. The answer is 19.\n"),
                dict(role='HUMAN', prompt="Question: Edward spent $ 6 to buy 2 books each book costing him the same amount of money. Now he has $ 12. How much did each book cost?\nLet's think step by step\nAnswer:"),
                dict(role='BOT', prompt='To find the cost of each book, we subtract the initial amount of money Edward had ($6) from the current amount of money he has ($12) and divide it by the number of books (2): (12 - 6) / 2 = 6 / 2 = 3 Therefore, each book cost $3. The answer is 3.\n'),
                dict(role='HUMAN', prompt="Question: Frank was reading through his favorite book. The book had 3 chapters, each with the same number of pages. It has a total of 594 pages. It took Frank 607 days to finish the book. How many pages are in each chapter?\nLet's think step by step\nAnswer:"),
                dict(role='BOT', prompt='To find the number of pages in each chapter, we divide the total number of pages in the book (594) by the number of chapters (3): 594 / 3 = 198. Therefore, each chapter has 198 pages. The answer is 198.\n'),
                dict(role='HUMAN', prompt="Question: {question}\nLet's think step by step\nAnswer:"),
            ],
        )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512))

svamp_eval_cfg = dict(evaluator=dict(type=Gsm8kEvaluator),
                      pred_postprocessor=dict(type=gsm8k_postprocess))

svamp_datasets = [
    dict(
        abbr='svamp',
        type=SVAMPDataset,
        path='./data/svamp/test.jsonl',
        reader_cfg=dict(input_columns=['question'], output_column='answer'),
        infer_cfg=svamp_infer_cfg,
        eval_cfg=svamp_eval_cfg)
]
