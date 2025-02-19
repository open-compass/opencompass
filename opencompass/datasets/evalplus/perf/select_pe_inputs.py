"""Select the most performance-exercising inputs from pe_inputs obtained from `sampling.py`.
"""

import json
from statistics import median

from tqdm import tqdm

from evalplus.config import PERF_CURATE_TIMEOUT_SECOND
from evalplus.data import get_human_eval_plus, get_mbpp_plus
from evalplus.data.mbpp import mbpp_deserialize_inputs, mbpp_serialize_inputs
from evalplus.perf.profile import are_profiles_broken, profile


def script(solutions: str, output_profiled_solutions: str, pe_inputs: str = None):
    assert solutions.endswith(".jsonl")
    assert pe_inputs is None or pe_inputs.endswith(".jsonl")
    assert output_profiled_solutions.endswith(".jsonl")

    evalplus = get_human_eval_plus(noextreme=True)
    mbppplus = get_mbpp_plus(noextreme=True)
    tasks = {**evalplus, **mbppplus}

    # assume each line's format is: {
    # "task_id": task's id,
    # "inputs": a list of inputs,
    inputs_dict = None

    if pe_inputs is not None:
        print("Loading performance-exercising inputs...")
        with open(pe_inputs, "r") as f:
            inputs_dict = {
                task["task_id"]: task["inputs"] for l in f for task in [json.loads(l)]
            }

    # Notably, the solutions are already validated and cleaned.
    with open(solutions, "r") as f:
        solutions = {}
        for l in f:
            solution = json.loads(l)
            solutions[solution["task_id"]] = solution["solution"]

    for task_id, task in tqdm(tasks.items()):
        if inputs_dict:
            inputs = (
                mbpp_deserialize_inputs(task_id, inputs_dict[task_id])
                if "Mbpp/" in task_id
                else inputs_dict[task_id]
            )
        else:
            inputs = task["base_input"] + list(task["plus_input"])

        input_costs = []

        if task_id.startswith("HumanEval"):
            canonical_solution = task["prompt"] + task["canonical_solution"]
        else:
            canonical_solution = task["canonical_solution"]

        for inp in inputs:
            costs = profile(
                canonical_solution,
                task["entry_point"],
                [inp],
                timeout_second_per_test=PERF_CURATE_TIMEOUT_SECOND,
            )
            if are_profiles_broken(costs):
                continue
            input_costs.append((median(costs), inp))
        input_costs.sort(reverse=True, key=lambda x: x[0])

        for _, pe_input in input_costs:
            solution_costs = []

            for solution in solutions[task_id]:
                costs = profile(
                    solution,
                    task["entry_point"],
                    [pe_input],
                    timeout_second_per_test=PERF_CURATE_TIMEOUT_SECOND,
                )
                if not are_profiles_broken(costs):
                    solution_costs.append(costs)
                    continue

                # stop once we find the first also the most performance-exercising input
                break

            # This means no timeouts happen for the input, so we use it.
            if len(solution_costs) == len(solutions[task_id]):
                break

        # If no satisfied input found, we don't save any profiled data.
        if len(input_costs) == 0 or len(solution_costs) != len(solutions[task_id]):
            print(f"Skipping {task_id}...")
            pe_input = None
            solution_costs = None
        else:
            pe_input = (
                mbpp_serialize_inputs(task_id, [pe_input])
                if task_id.startswith("Mbpp/")
                else [pe_input]
            )

        with open(output_profiled_solutions, "a") as f:
            f.write(
                json.dumps(
                    {
                        "task_id": task_id,
                        "pe_input": pe_input,
                        "solutions": solutions[task_id],
                        "counter_profile": solution_costs,
                    }
                )
                + "\n"
            )


def main():
    from fire import Fire

    Fire(script)


if __name__ == "__main__":
    main()
