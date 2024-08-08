from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import ChemBenchDataset
from opencompass.utils.text_postprocessors import first_capital_postprocess


chembench_reader_cfg = dict(
    input_columns=['input', 'A', 'B', 'C', 'D'],
    output_column='target',
    train_split='dev')

chembench_all_sets = [
    'Name_Conversion',
    'Property_Prediction',
    'Mol2caption',
    'Caption2mol',
    'Product_Prediction',
    'Retrosynthesis',
    'Yield_Prediction',
    'Temperature_Prediction',
    'Solvent_Prediction'
]


chembench_datasets = []
for _name in chembench_all_sets:
    # _hint = f'There is a single choice question about {_name.replace("_", " ")}. Answer the question by replying A, B, C or D.'
    _hint = f'There is a single choice question about chemistry. Answer the question by replying A, B, C or D.'

    chembench_infer_cfg = dict(
        ice_template=dict(
            type=PromptTemplate,
            template=dict(round=[
                dict(
                    role='HUMAN',
                    prompt=
                    f'{_hint}\nQuestion: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nAnswer: '
                ),
                dict(role='BOT', prompt='{target}\n')
            ]),
        ),
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin='</E>',
                round=[
                    dict(
                        role='HUMAN',
                        prompt=
                        f'{_hint}\nQuestion: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nAnswer: '
                    ),
                ],
            ),
            ice_token='</E>',
        ),
        retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),
        inferencer=dict(type=GenInferencer),
    )

    chembench_eval_cfg = dict(
        evaluator=dict(type=AccEvaluator),
        pred_postprocessor=dict(type=first_capital_postprocess))

    chembench_datasets.append(
        dict(
            abbr=f'ChemBench_{_name}',
            type=ChemBenchDataset,
            path='opencompass/ChemBench',
            name=_name,
            reader_cfg=chembench_reader_cfg,
            infer_cfg=chembench_infer_cfg,
            eval_cfg=chembench_eval_cfg,
        ))

del _name, _hint
