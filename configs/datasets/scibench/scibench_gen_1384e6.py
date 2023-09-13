import os
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import ScibenchDataset, ScibenchEvaluator

scibench_reader_cfg = dict(input_columns=['question'], output_column='answer')

_path_prefix = "./data/scibench"

scibench_subsets = [
    "atkins",
    "calculus",
    "chemmc",
    "class",
    "diff",
    "fund",
    "matter",
    "quan",
    "stat",
    "thermo"
]

scibench_datasets = []
for prompt_type in ["zs", "zs-cot", "fs", "fs-cot"]:
    for _name in scibench_subsets:
        if prompt_type == "zs":
            scibench_infer_cfg = dict(
                prompt_template=dict(
                    type=PromptTemplate,
                    template=dict(round=[
                        dict(
                            role="SYSTEM",
                            prompt="Please provide a clear and step-by-step solution for a scientific problem in the categories of Chemistry, Physics, or Mathematics. The problem will specify the unit of measurement, which should not be included in the answer. Express the final answer as a decimal number with three digits after the decimal point. Conclude the answer by stating 'Therefore, the answer is \\boxed[ANSWER].'"
                        ),
                        dict(
                            role="HUMAN",
                            prompt=f"Problem: {{question}}\nAnswer:"
                        )
                    ])),
                retriever=dict(type=ZeroRetriever),
                inferencer=dict(type=GenInferencer, max_out_len=512))


        elif prompt_type == "zs-cot":
            scibench_infer_cfg = dict(
                prompt_template=dict(
                    type=PromptTemplate,
                    template=dict(round=[
                        dict(
                            role="SYSTEM",
                            prompt="Please provide a clear and step-by-step solution for a scientific problem in the categories of Chemistry, Physics, or Mathematics. The problem will specify the unit of measurement, which should not be included in the answer. Express the final answer as a decimal number with three digits after the decimal point. Conclude the answer by stating 'Therefore, the answer is \\boxed[ANSWER].'"
                        ),
                        dict(
                            role="HUMAN",
                            prompt=f"Problem: {{question}}\nAnswer:Let’s think step by step."
                        )
                    ])),
                retriever=dict(type=ZeroRetriever),
                inferencer=dict(type=GenInferencer, max_out_len=512))
        

        elif prompt_type == "fs":
            _hint = None
            _path = f"{_path_prefix}/{_name}_prompt.txt"
            if os.path.exists(_path):
                _hint = open(_path, "r").read()

            scibench_infer_cfg = dict(
                prompt_template=dict(
                    type=PromptTemplate,
                    template=dict(round=[
                            dict(
                                role="HUMAN",
                                prompt=f"{_hint}\n\nProblem 6: {{question}}\nAnswer: "
                            )
                        ])),
                retriever=dict(type=ZeroRetriever),
                inferencer=dict(type=GenInferencer, max_out_len=512))


        elif prompt_type == "fs-cot":
            _hint = None
            _path = f"{_path_prefix}/{_name}_sol.txt"
            if os.path.exists(_path):
                _hint = open(_path, "r").read()

            scibench_infer_cfg = dict(
                prompt_template=dict(
                    type=PromptTemplate,
                    template=dict(round=[
                            dict(
                                role="HUMAN",
                                prompt=f"{_hint}\n\nProblem 6: {{question}}\nExplanation for Problem 6: "
                            )
                        ])),
                retriever=dict(type=ZeroRetriever),
                inferencer=dict(type=GenInferencer, max_out_len=512))

        scibench_eval_cfg = dict(evaluator=dict(type=ScibenchEvaluator))

        scibench_datasets.append(
            dict(
                type=ScibenchDataset,
                path=f"{_path_prefix}/",
                name=_name,
                abbr=f"scibench-{_name}_{prompt_type}",
                reader_cfg=scibench_reader_cfg,
                infer_cfg=scibench_infer_cfg.copy(),
                eval_cfg=scibench_eval_cfg.copy())
            )


