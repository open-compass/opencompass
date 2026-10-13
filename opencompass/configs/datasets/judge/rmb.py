from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import RMBEvaluator
from opencompass.datasets import RMBDataset


subjective_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='judge',
    )

data_path = './data/judgeeval/rmb_dataset'
subjective_all_sets = ['rmb_dataset.json']
get_rmb_dataset = []


prompt_choice_prefix = """
Please act as an impartial judge to evaluate the responses provided by two AI assistants to the user question below. Your evaluation should focus on the following criteria: helpfulness, relevance, accuracy, depth, creativity, and level of detail.

- Do not let the order of presentation, response length, or assistant names influence your judgment.
- Base your decision solely on how well each response addresses the user’s question and adheres to the instructions.

Your final reply must be structured in the following format:
{
  "Choice": "[Model A or Model B]"
}
"""

prompt_choice_en = """User Question: {question}

Model A's Response: {answerA}

Model B's Response: {answerB}

Now it's your turn. Please provide selection result as required:
"""

for _name in subjective_all_sets:
    subjective_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(
                        role='HUMAN',
                        prompt=prompt_choice_prefix + prompt_choice_en
                    ),
                ]),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer, max_out_len=4096),
        )

    rmb_eval_cfg = dict(
        evaluator=dict(
            type=RMBEvaluator,
        ),
    )

    get_rmb_dataset.append(
        dict(
            abbr=f'{_name.split(".")[0]}',
            type=RMBDataset,
            path=data_path,
            name=_name,
            reader_cfg=subjective_reader_cfg,
            infer_cfg=subjective_infer_cfg,
            eval_cfg=rmb_eval_cfg,
            mode='singlescore',
        ))
