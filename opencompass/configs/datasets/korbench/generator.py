import os
import json

# Define tasks and modes
tasks_with_modes = {
    "cipher": ["zero-shot"],
    "counterfactual": ["zero-shot"],
    "logic": ["zero-shot"],
    "operation": ["zero-shot"],
    "puzzle": ["zero-shot"],
}

# Base path for datasets and output
base_path = "/home/epsilon/miniforge3/my_opencompass_project/opencompass"
datasets_base_path = os.path.join(base_path, "data", "korbench")
output_base_path = os.path.join(base_path, "outputs", "matrix_scripts")

# Ensure output directory exists
os.makedirs(output_base_path, exist_ok=True)

# Template for Python scripts
script_template = """
import os
import json
from opencompass.datasets.korbench.{task} import korbench{task}Dataset
from opencompass.datasets.korbench.{task} import korbench{task}Evaluator
from opencompass.openicl.icl_inferencer import korbench_GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

# Configuration
dataset_path = "{dataset_path}"
output_path = "{output_path}"
metadata_file = "{metadata_file}"
mode = "{mode}"

# Ensure output directories exist
os.makedirs(output_path, exist_ok=True)

# Prompt template
prompt_template = dict(
    type=PromptTemplate,
    template=dict(
        begin=[dict(role="HUMAN", prompt="{system_message}")],
        round=[dict(role="HUMAN", prompt="{prompt_template}")]
    )
)

# Reader configuration
reader_cfg = dict(
    input_columns=["prompt"],
    output_column="answer",
)

# Inference configuration
infer_cfg = dict(
    prompt_template=prompt_template,
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=korbench_GenInferencer, max_out_len=1024),
)

# Evaluation configuration
eval_cfg = dict(
    evaluator=dict(type=korbench{task}Evaluator),
    pred_role="BOT",
)

try:
    # Load dataset
    dataset = korbench{task}Dataset.load(dataset_path)

    # Evaluate dataset
    evaluator = korbench{task}Evaluator(metadata_file=metadata_file)
    results = evaluator.score()

    # Save results
    results_file = os.path.join(output_path, "results_{task}_{mode}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {{results_file}}")

except Exception as e:
    print(f"An error occurred: {{e}}")
"""

# Iterate over tasks and modes to generate scripts
for task, modes in tasks_with_modes.items():
    task = task.lower()
    for mode in modes:
        # Dynamically generate system messages and prompt templates
        system_message = f"You are an expert in {task}. Solve the following {task} problem."
        # round=[dict(role="HUMAN", prompt="### Cipher Task:\n{prompt}\n### Answer:")] round=[dict(role="HUMAN", prompt="### Cipher Task:\n{prompt}\n### Answer:")]
        prompt_template = f"### {task.capitalize()} Task:\n{{prompt}}\n### Answer:"

        # Define paths
        dataset_path = os.path.join(datasets_base_path, task, f"{mode}.jsonl")
        output_path = os.path.join(output_base_path, task, mode)
        metadata_file = os.path.join(base_path, "outputs", "metadata", f"metadata_{task}_{mode}.json")

        # Fill the template with task-specific details
        script_content = script_template.format(
            task=task,
            dataset_path=dataset_path,
            output_path=output_path,
            mode=mode,
            system_message=system_message,
            prompt_template=prompt_template,
            metadata_file=metadata_file,
        )

        # Write the script to a file
        script_file = os.path.join(output_base_path, f"korbench_{task}_{mode}_gen.py")
        with open(script_file, "w") as f:
            f.write(script_content)

        print(f"Script generated: {script_file}")
