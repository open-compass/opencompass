"""Analyze the variance of PE and their time cost.
Filter those with high variance and low time cost.
"""

import json
import math
from datetime import datetime
from typing import List

import numpy as np
from rich.console import Console
from rich.syntax import Syntax
from termcolor import colored

from evalplus.config import PREF_CURATE_MIN_INSTRUCTION


def cv(time_costs: List[float]) -> float:
    """
    We use Coefficient of Variation (CV) to as the variance of PE.
    CV = 100 * standard deviation / mean
    """
    if len(time_costs) == 0:
        raise ValueError("time_costs is empty.")
    return 100 * np.std(time_costs) / np.mean(time_costs)


def filter_by_profile_size(task2profile: dict, threshold: int = 10):
    to_remove = []
    for task_id, profile in task2profile.items():
        if (
            profile is None
            or len(profile) < threshold
            or any(None in p for p in profile)
        ):
            print(colored(f"⚠️ {task_id} skipped: #profile < {threshold}", "red"))
            to_remove.append(task_id)
    for task_id in to_remove:
        del task2profile[task_id]
    return task2profile


def filter_by_compute_cost(
    task2profile: dict, thresh: float = PREF_CURATE_MIN_INSTRUCTION
):
    """Filter out tasks that can be solved using less than threshold #instruction."""
    to_remove = []
    for task_id, profile in task2profile.items():
        if (
            min(np.mean(p) for p in profile) < thresh
        ):  # filter if some solution is too fast
            print(
                colored(
                    f"⚠️ {task_id} skipped: some solution is faster than {thresh} #instruction",
                    "red",
                )
            )
            to_remove.append(task_id)
    for task_id in to_remove:
        del task2profile[task_id]
    return task2profile


def filter_by_cv(task2profile: dict, thresh: float, percentile: int = 95):
    to_remove = []
    for task_id, profile in task2profile.items():
        mean_var = np.percentile([cv(p) for p in profile], percentile)
        if mean_var > thresh:
            print(
                colored(
                    f"⚠️ {task_id} skipped: P{percentile} CV = {mean_var:.1f}% > {thresh}%",
                    "red",
                )
            )
            to_remove.append(task_id)
    for task_id in to_remove:
        del task2profile[task_id]
    return task2profile


# smaller time, larger threshold
def thresh_fn(base_thresh, x, weight=0.002):
    return base_thresh + math.sqrt(weight / x)


def adaptive_seg1d(arr1d, base_thresh=0.10):
    # sort from large to small
    arr1d = np.sort(arr1d)[::-1]
    # relative distance
    relative_distance = -np.diff(arr1d) / arr1d[:-1]

    splitter_idx = []
    for i, rel in enumerate(relative_distance):
        if rel > thresh_fn(base_thresh, arr1d[i], weight=PREF_CURATE_MIN_INSTRUCTION):
            splitter_idx.append(i + 1)

    # [9, 8, 7, |-> 3, 2 1]
    # splitter_idx points to the slowest in each cluster
    return np.split(arr1d, splitter_idx)


def filter_by_clustering(task2profile: dict, base_threshold=0.2, min_clusters=3):
    to_remove = []
    for task_id, profile in task2profile.items():
        if len(adaptive_seg1d(np.mean(profile, axis=1), base_threshold)) < min_clusters:
            print(
                colored(
                    f"⚠️ {task_id} skipped: #Cluster = 0 with {base_threshold=}%",
                    "red",
                )
            )
            to_remove.append(task_id)
    for task_id in to_remove:
        del task2profile[task_id]
    return task2profile


def brief_list_repr(lst, head_count=4, tail_count=4):
    if len(lst) <= head_count + tail_count:
        return f"{lst}"
    else:
        head = ", ".join(str(x) for x in lst[:head_count])
        tail = ", ".join(str(x) for x in lst[-tail_count:])
        return f"[{head}, ..., {tail}]"


def script(
    profiled_solutions: str,
    output_dataset: str = f"evalperf-{datetime.now():%Y%m%d}.jsonl",
    debug_tasks: List[str] = [],
    min_clusters=4,
):
    assert profiled_solutions.endswith(".jsonl")
    assert output_dataset.endswith(".jsonl")

    # read jsonl
    with open(profiled_solutions, "r") as f:
        profiled_solutions = [json.loads(l) for l in f if l.strip()]

    console = Console()

    task2profile = {d["task_id"]: d["counter_profile"] for d in profiled_solutions}
    print(f"Loaded {len(task2profile)} tasks.")

    # * Criteria 1: Profile cannot be empty
    task2profile = filter_by_profile_size(task2profile)
    print(f"{len(task2profile)} tasks with profile.")

    # * Criteria 2: Solutions should run more than MIN_SLOWEST_INSTRUCTION_COUNT
    task2profile = filter_by_compute_cost(task2profile)
    print(
        f"{len(task2profile)} tasks with slowest mean time > {PREF_CURATE_MIN_INSTRUCTION}s."
    )

    # * Criteria 3: P99-CV should be less than 5%
    final_thresh = 5
    percentile = 99
    task2profile = filter_by_cv(
        task2profile, thresh=final_thresh, percentile=percentile
    )
    print(f"{len(task2profile)} tasks with CV <= {final_thresh}%.")

    # * Criteria 4: Cluster should be more than 1
    task2profile = filter_by_clustering(
        task2profile, base_threshold=0.2, min_clusters=min_clusters
    )
    print(f"{len(task2profile)} tasks with #Cluster >= {min_clusters}.")

    # export dataset
    task2solution = {d["task_id"]: d for d in profiled_solutions}
    # each item is {"task_id": "xxx", "solutions": [...], "percentile": [...]}
    export_dataset = []
    total_clusters = 0
    for task_id, profile in task2profile.items():
        print(colored(f"-========== {task_id} ==========-", "green"))
        if task_id in debug_tasks:
            print(colored(f"Debugging {task_id}", "red"))
        mean_runtime = [np.mean(p) for p in profile]
        clusters = adaptive_seg1d(mean_runtime)  # descend
        print(colored(f"#seg = {len(clusters)}", "green"))

        accumulative_ratio = []
        ref_idx = []
        for i, cluster in enumerate(clusters):
            prior_ar = 0 if i == 0 else accumulative_ratio[-1]
            ratio = 100 * len(cluster) / len(mean_runtime)
            acc_ratio = prior_ar + ratio
            brief_list_str = brief_list_repr([round(1000 * v) for v in cluster])
            print(
                f"#{i} |{len(cluster):<3}| ({acc_ratio:<4.1f}) @cv {cv(cluster):.1f}: {brief_list_str}"
            )
            accumulative_ratio.append(acc_ratio)
            ref_idx.append(np.where(mean_runtime == cluster[0])[0][0])

            if task_id in debug_tasks:
                # print solutions
                solution_text = task2solution[task_id]["solutions"][ref_idx[-1]]
                # remove empty lines
                solution_text = "\n".join(
                    line for line in solution_text.split("\n") if line.strip()
                )
                console.print(Syntax(solution_text, "python"))
                print(colored("-" * 32, "green"))

        total_clusters += len(clusters)

        # add reference solution and check consistency
        for i in range(len(ref_idx)):
            if i == 0:
                continue
            # prior runtime must be larger than current
            assert mean_runtime[ref_idx[i - 1]] > mean_runtime[ref_idx[i]]

        reference = [task2solution[task_id]["solutions"][idx] for idx in ref_idx]

        assert len(reference) == len(clusters)
        assert len(accumulative_ratio) == len(reference)
        item = {
            "task_id": task_id,
            "reference": reference,
            "pe_input": task2solution[task_id]["pe_input"],
            "scores": accumulative_ratio,
        }
        export_dataset.append(item)

    print(f"Total clusters: {total_clusters}")

    with open(output_dataset, "w") as f:
        for item in export_dataset:
            f.write(json.dumps(item) + "\n")


def main():
    from fire import Fire

    Fire(script)


if __name__ == "__main__":
    main()
