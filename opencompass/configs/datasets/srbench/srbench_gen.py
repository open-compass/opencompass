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
            Finally, please output only the formula string you inferred (e.g. y=x_0 * x_1), without any additional information.
        '''

srbench_reader_cfg = dict(input_columns=['prompt1','prompt2'], output_column='Formula')

srbench_datasets = []

srbench_infer_cfg = dict(
    prompt_template=dict(
            type=PromptTemplate,
            template=dict(
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
        evaluator=dict(type=SRbenchDatasetEvaluator, path='opencompass/srbench'),
        pred_postprocessor=dict(type=mydataset_postprocess),
        pred_role='BOT',
        )

srbench_datasets.append(
        dict(
            abbr='srbench',
            type=SRbenchDataset,
            path='opencompass/srbench',
            reader_cfg=srbench_reader_cfg,
            infer_cfg=srbench_infer_cfg,
            eval_cfg=srbench_eval_cfg,
        )
    )

