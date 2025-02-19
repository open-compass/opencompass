from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import CoinFlipDataset, coinflip_pred_postprocess

coinflip_reader_cfg = dict(
    input_columns=['question'],
    output_column='answer',
    train_split='test',
    test_split='test'
)

coinflip_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='Question: A coin is heads up. Ka flips the coin. Sherrie flips the coin. Is the coin still heads up?\nAnswer:'),
                dict(role='BOT', prompt='The coin was flipped by Ka and Sherrie. So the coin was flipped 2 times, which is an even number. The coin started heads up, so after an even number of flips, it will still be heads up.\nSo the answer is yes.\n'),
                dict(role='HUMAN', prompt='Question: A coin is heads up. Jamey flips the coin. Teressa flips the coin. Is the coin still heads up?\nAnswer:'),
                dict(role='BOT', prompt='The coin was flipped by Jamey and Teressa. So the coin was flipped 2 times, which is an even number. The coin started heads up, so after an even number of flips, it will still be heads up.\nSo the answer is yes.\n'),
                dict(role='HUMAN', prompt='Question: A coin is heads up. Maybelle flips the coin. Shalonda does not flip the coin. Is the coin still heads up?\nAnswer:'),
                dict(role='BOT', prompt='The coin was flipped by Maybelle. So the coin was flipped 1 time, which is an odd number. The coin started heads up, so after an odd number of flips, it will be tails up.\nSo the answer is no.\n'),
                dict(role='HUMAN', prompt='Question: A coin is heads up. Millicent does not flip the coin. Conception flips the coin. Is the coin still heads up?\nAnswer:'),
                dict(role='BOT', prompt='The coin was flipped by Conception. So the coin was flipped 1 time, which is an odd number. The coin started heads up, so after an odd number of flips, it will be tails up.\nSo the answer is no.\n'),
                dict(role='HUMAN', prompt='Question: A coin is heads up. Sal flips the coin. Raymond does not flip the coin. Is the coin still heads up?\nAnswer:'),
                dict(role='BOT', prompt='The coin was flipped by Sal. So the coin was flipped 1 time, which is an odd number. The coin started heads up, so after an odd number of flips, it will be tails up.\nSo the answer is no.\n'),
                dict(role='HUMAN', prompt='Question: A coin is heads up. Conception flips the coin. Kristian does not flip the coin. Is the coin still heads up?\nAnswer:'),
                dict(role='BOT', prompt='The coin was flipped by Conception. So the coin was flipped 1 time, which is an odd number. The coin started heads up, so after an odd number of flips, it will be tails up.\nSo the answer is no.\n'),
                dict(role='HUMAN', prompt='Question: A coin is heads up. Inga does not flip the coin. Elanor does not flip the coin. Is the coin still heads up?\nAnswer:'),
                dict(role='BOT', prompt='The coin was flipped by no one. So the coin was flipped 0 times. The coin started heads up, and it was not flipped, so it is still heads up.\nSo the answer is yes.\n'),
                dict(role='HUMAN', prompt='Question: A coin is heads up. Ryan flips the coin. Shaunda flips the coin. Is the coin still heads up?\nAnswer:'),
                dict(role='BOT', prompt='The coin was flipped by Ryan and Shaunda. So the coin was flipped 2 times, which is an even number. The coin started heads up, so after an even number of flips, it will still be heads up.\nSo the answer is yes.\n'),
                dict(role='HUMAN', prompt='Question: {question}\nAnswer:'),
            ]
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512),
)

coinflip_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_postprocessor=dict(type=coinflip_pred_postprocess),
)

coinflip_datasets = [
    dict(
        abbr='coinflip',
        type=CoinFlipDataset,
        path='coin_flip',
        reader_cfg=coinflip_reader_cfg,
        infer_cfg=coinflip_infer_cfg,
        eval_cfg=coinflip_eval_cfg
    )
]