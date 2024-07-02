import os

charm_memory_tasks = [
    'Chinese_Anachronisms_Judgment',
    'Chinese_Movie_and_Music_Recommendation',
    'Chinese_Sport_Understanding',
    'Chinese_Time_Understanding',
]

dataset_path = 'data/CHARM/memorization'

system_prompt_template = """Please act as an impartial judge, comparing the responses of the AI assistants to the reference answer and determining if the answers are correct.
You will receive the reference answer provided by a human and the responses of the AI assistants.
Your task is to judge whether the AI assistant's answers is correct.
{task_specific_prompt}
After providing your explanation, strictly output your final judgment in the following format: “[正确]” if the AI assistant's response is correct, “[错误]” if the AI assistant's response is incorrect.
"""

task_specific_prompts = {
    'Chinese_Anachronisms_Judgment':
    "If the provided reference answer is a list, the model's prediction is considered correct if it matches any item in the list.",
    'Chinese_Time_Understanding':
    "When evaluating the AI assistant's response regarding Chinese solar terms, as long as the AI assistant's response falls within the time frame provided in the reference answer, consider it correct.",
    'Chinese_Sport_Understanding':
    "If the provided reference answer is a list, the model's prediction is considered correct if it matches any item in the list."
}

judge_system_prompts = {
    k: system_prompt_template.format(task_specific_prompt=v)
    for k, v in task_specific_prompts.items()
}
