"""Generate a .jsonl file where each line is a json object
representing a programming problem with a task ID ("task_id")
and a list of enhanced inputs ("inputs") for that task.
"""

import argparse
import json
import os

from opencompass.datasets.evalplus.data.mbpp import mbpp_serialize_inputs
from opencompass.datasets.evalplus.gen.chatgpt_gen import ChatGPTGen
from opencompass.datasets.evalplus.gen.type_mut import TypedMutGen

class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


# Used for MBPP as MBPP's prompt is not a formal function signature
def insert_contract_into_code(entry_point, code, contract):
    lines = code.split("\n")
    index = lines.index(
        next(line for line in lines if line.startswith(f"def {entry_point}"))
    )
    lines.insert(index + 1, contract)
    return "\n".join(lines)


def input_generation(args, problems):
    with open(args.output, "w") as file:
        for problem in problems.values():
            new_input = {}
            task_id = problem["task_id"]
            print(f"generating inputs for {task_id} ...")
            # by default we do not include constraints in the prompt (code)
            code = problem["prompt"] + problem["canonical_solution"]
            # but we use c_code to include contract which checks input validity at execution time
            if args.dataset == "humaneval":
                c_code = (
                    problem["prompt"]
                    + problem["contract"]
                    + problem["canonical_solution"]
                )
            elif args.dataset == "mbpp":
                c_code = problem["prompt"] + insert_contract_into_code(
                    entry_point=problem["entry_point"],
                    code=problem["canonical_solution"],
                    contract=problem["contract"],
                )

            # first generate chatgpt
            input_gen = ChatGPTGen(
                problem["base_input"], problem["entry_point"], c_code, code
            ).generate(args.chatgpt_len)
            # generate mutation next

            if input_gen is None or len(input_gen) == 0:
                new_input["task_id"] = task_id
                new_input["inputs"] = {}
                file.write(json.dumps(new_input, cls=SetEncoder) + "\n")
                continue

            input_gen.extend(
                TypedMutGen(input_gen, problem["entry_point"], c_code).generate(
                    args.mut_len
                )
            )
            print(f"generated {len(input_gen)} inputs")
            new_input["task_id"] = task_id
            if args.dataset == "mbpp":
                new_input["inputs"] = mbpp_serialize_inputs(task_id, input_gen)
            new_input["inputs"] = input_gen
            file.write(json.dumps(new_input, cls=SetEncoder) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", required=True, type=str, choices=["humaneval", "mbpp"]
    )
    parser.add_argument("--chatgpt_len", required=True, type=int)
    parser.add_argument("--mut_len", required=True, type=int)
    parser.add_argument("--output", type=str, help="Output .jsonl path")
    args = parser.parse_args()

    problems = None
    if args.dataset == "humaneval":
        from evalplus.data import get_human_eval_plus

        # Allow it to be incomplete
        problems = get_human_eval_plus(err_incomplete=False)
        args.output = args.output or "HumanEvalPlusInputs.jsonl"

    if args.dataset == "mbpp":
        from evalplus.data import get_mbpp_plus

        problems = get_mbpp_plus(err_incomplete=False)
        args.output = args.output or "MbppPlusInput.jsonl"

    assert not os.path.isfile(args.output), f"{args.output} already exists!"
    input_generation(args, problems)


if __name__ == "__main__":
    main()
