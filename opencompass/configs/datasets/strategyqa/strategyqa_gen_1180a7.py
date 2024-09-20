from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import StrategyQADataset, strategyqa_pred_postprocess, strategyqa_dataset_postprocess

strategyqa_reader_cfg = dict(
    input_columns=['question'],
    output_column='answer',
    train_split='test',
    test_split='test')

strategyqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt=
                    'Question: Do hamsters provide food for any animals?\nAnswer:'
                ),
                dict(
                    role='BOT',
                    prompt=
                    'Hamsters are prey animals. Prey are food for predators. Thus, hamsters provide food for some animals.\nSo the answer is yes\n'
                ),
                dict(
                    role='HUMAN',
                    prompt=
                    'Question: Could Brooke Shields succeed at University of Pennsylvania?\nAnswer:'
                ),
                dict(
                    role='BOT',
                    prompt=
                    'Brooke Shields went to Princeton University. Princeton University is about as academically rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania.\nSo the answer is yes\n'
                ),
                dict(
                    role='HUMAN',
                    prompt=
                    'Question: Hydrogen\'s atomic number squared exceeds number of Spice Girls?\nAnswer:'
                ),
                dict(
                    role='BOT',
                    prompt=
                    'Hydrogen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. Thus, Hydrogen\'s atomic number squared is less than 5.\nSo the answer is no\n'
                ),
                dict(
                    role='HUMAN',
                    prompt=
                    'Question: Is it common to see frost during some college commencements?\nAnswer:'
                ),
                dict(
                    role='BOT',
                    prompt=
                    'College commencement ceremonies can happen in December, May, and June. December is in the winter, so there can be frost. Thus, there could be frost at some commencements.\nSo the answer is yes\n'
                ),
                dict(
                    role='HUMAN',
                    prompt=
                    'Question: Yes or no: Could a llama birth twice during War in Vietnam (1945-46)?\nAnswer:'
                ),
                dict(
                    role='BOT',
                    prompt=
                    'The War in Vietnam was 6 months. The gestation period for a llama is 11 months, which is more than 6 months. Thus, a llama could not give birth twice during the War in Vietnam.\nSo the answer is no\n'
                ),
                dict(
                    role='HUMAN',
                    prompt='Question: Would a pear sink in water?\nAnswer:'),
                dict(
                    role='BOT',
                    prompt=
                    'The density of a pear is about 0.6g/cm3, which is less than water. Objects less dense than water float. Thus, a pear would float.\nSo the answer is no\n'
                ),
                dict(role='HUMAN', prompt='Question: {question}\nAnswer:'),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512))

strategyqa_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_postprocessor=dict(type=strategyqa_pred_postprocess),
    dataset_postprocessor=dict(type=strategyqa_dataset_postprocess))

strategyqa_datasets = [
    dict(
        abbr='strategyqa',
        type=StrategyQADataset,
        path='opencompass/strategy_qa',
        reader_cfg=strategyqa_reader_cfg,
        infer_cfg=strategyqa_infer_cfg,
        eval_cfg=strategyqa_eval_cfg)
]
