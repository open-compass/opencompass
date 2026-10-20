from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import MathBenchDataset, Math401Evaluator, mathbench_postprocess

cloze_prompt = [
                dict(role='HUMAN', prompt='Q: Calculate 2.9-0.11.'),
                dict(role='BOT', prompt='A: Let\'s think step by step, 2.9 - 0.11 equals 2.7900. The answer is 2.7900.\n'),
                dict(role='HUMAN', prompt='Q: Calculate 0.15-0.032.'),
                dict(role='BOT', prompt='A: Let\'s think step by step, 0.15 - 0.032 equals 0.1180. The answer is 0.1180.\n'),
                dict(role='HUMAN', prompt='Q: Calculate 78*64.'),
                dict(role='BOT', prompt='A: Let\'s think step by step, 78 multiplied by 64 equals 4992. The answer is 4992.\n'),
                dict(role='HUMAN', prompt='Q: Calculate 62×42.'),
                dict(role='BOT', prompt='A: Let\'s think step by step, 62 multiplied by 42 equals 2604. The answer is 2604.\n'),
                dict(role='HUMAN', prompt='Q: Calculate {question}'),
                dict(role='BOT', prompt='A: {answer}\n')]

math401_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=cloze_prompt,
            ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512),
)

math401_eval_cfg = dict(
    evaluator=dict(type=Math401Evaluator),
    pred_postprocessor=dict(type=mathbench_postprocess, name='en'))

math401_datasets = [
    dict(
        abbr='math401',
        type=MathBenchDataset,
        path=f'./data/math401/',
        with_circular=False,
        name='cloze_en',
        reader_cfg=dict(
            input_columns=['question'],
            output_column='answer'
            ),
        infer_cfg=math401_infer_cfg,
        eval_cfg=math401_eval_cfg,
    )]
