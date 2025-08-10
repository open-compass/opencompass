from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import WinograndeDatasetV2
from opencompass.utils.text_postprocessors import first_option_postprocess

winogrande_reader_cfg = dict(
    input_columns=['opt1', 'opt2'],
    output_column='answer',
)

winogrande_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_option_postprocess, options='AB'),
)

_winogrande_prompt = dict(
    prompt_1='Which of the following is a good sentence:\nA. {opt1}\nB. {opt2}\nAnswer:',
    prompt_2='Which is a good sentence out of the following:\nA. {opt1}\nB. {opt2}\nAnswer:',
    prompt_3='Can you identify a good sentence from the following:\nA. {opt1}\nB. {opt2}\nAnswer:',
)

winogrande_datasets = []
for _choice in _winogrande_prompt:
    winogrande_datasets.append(
        dict(
            abbr='winogrande_'+_choice,
            type=WinograndeDatasetV2,
            path='opencompass/winogrande',
            reader_cfg=winogrande_reader_cfg,
            infer_cfg=dict(
                prompt_template=dict(
                    type=PromptTemplate,
                    template=dict(round=[
                        dict(
                            role='HUMAN',
                            prompt=_winogrande_prompt[_choice]
                        ),
                    ]),
                ),
                retriever=dict(type=ZeroRetriever),
                inferencer=dict(type=GenInferencer),
            ),
            eval_cfg=winogrande_eval_cfg),
    )

del _choice
