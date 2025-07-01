from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import (
    SRbenchDataset,SRbenchDatasetEvaluator,mydataset_postprocess
)

INFER_TEMPLATE = f'''
            You will be provided with a set of input-output pairs. Based on these data, infer the mathematical relationship between y and multiple input variables. Please note that the possible mathematical operations include: +, -, *, /, exp, sqrt, sin, arcsin, and constant terms.
            The input sample data are as follows:
            {{prompt1}}
            Based on the above data, please infer the possible formula. Ensure that your inference applies to all the provided data points, and consider both linear and nonlinear combinations.
            Verify whether your formula applies to the following new data point and adjust it to ensure accuracy:
            {{prompt2}}
            Finally, please output only the formula string you inferred (e.g. z=x_0 * x_1), without any additional information.
        '''
SYSTEM_PROMPT=f'''You are an exceptional symbolic regression assistant. Your specialty lies in analyzing numerical relationships among data and variables. When provided with mathematical questions or data from humans, you carefully comprehend the essence of the problem, methodically clarify relationships among variables, and meticulously derive the solution step by step. Ultimately, you output a precise, concise, and interpretable mathematical formula. Each step of your reasoning should be clear and explicit, aiding humans in gaining a deeper understanding of the problem.'''
srbench_reader_cfg = dict(input_columns=['prompt1','prompt2'], output_column='formula')

srbench_datasets=[]

path = ['Constant.json','Keijzer.json','Feynman.json','Nguyen.json','R.json']
for task in path:
    srbench_infer_cfg = dict(
        prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                        begin=[
                        dict(role='SYSTEM', fallback_role='HUMAN', prompt=SYSTEM_PROMPT)
                    ],
                        round=[
                            dict(
                                role='HUMAN',
                                prompt=INFER_TEMPLATE)
                        ]
                    ),
                ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer),
            )

    srbench_eval_cfg = dict(
            evaluator=dict(type=SRbenchDatasetEvaluator,path=task),
            pred_role='BOT',
            pred_postprocessor=dict(type=mydataset_postprocess),
            num_gpus=1
            )

    srbench_datasets.append(
            dict(
                abbr='srbench',
                type=SRbenchDataset,
                path=task,
                reader_cfg=srbench_reader_cfg,
                infer_cfg=srbench_infer_cfg,
                eval_cfg=srbench_eval_cfg,
            )
        )

