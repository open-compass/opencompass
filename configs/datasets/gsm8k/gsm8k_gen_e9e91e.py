# GONNA BE DEPRECATED, DON'T USE IT
# The postprocessor has the assumption that the prompt is in the format of "Question:blabla"
# This config does not follow the above assumption, thus deprecated

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator
gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

gsm8k_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?'),
                dict(role='BOT', prompt='A: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6.\n'),
                dict(role='HUMAN', prompt='Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?'),
                dict(role='BOT', prompt='A: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.\n'),
                dict(role='HUMAN', prompt='Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?'),
                dict(role='BOT', prompt="A: Leah had 32 chocolates and Leah's sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.\n"),
                dict(role='HUMAN', prompt='Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?'),
                dict(role='BOT', prompt='A: Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is 8.\n'),
                dict(role='HUMAN', prompt='Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?'),
                dict(role='BOT', prompt='A: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9.\n'),
                dict(role='HUMAN', prompt='Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?'),
                dict(role='BOT', prompt='A: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29.\n'),
                dict(role='HUMAN', prompt='Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?'),
                dict(role='BOT', prompt='A: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.\n'),
                dict(role='HUMAN', prompt='Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?'),
                dict(role='BOT', prompt='A: She bought 5 bagels for $3 each. This means she spent 5 * $3 = $15 on the bagels. She had $23 in beginning, so now she has $23 - $15 = $8. The answer is 8.\n'),
                dict(role='HUMAN', prompt='Q: {question}'),
                dict(role='BOT', prompt='A: {answer}\n'),
            ],
        )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

gsm8k_eval_cfg = dict(evaluator=dict(type=Gsm8kEvaluator),
                      pred_role='BOT',
                      pred_postprocessor=dict(type=gsm8k_postprocess),
                      dataset_postprocessor=dict(type=gsm8k_dataset_postprocess))

gsm8k_datasets = [
    dict(
        abbr='gsm8k',
        type=GSM8KDataset,
        path='opencompass/gsm8k',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg)
]
