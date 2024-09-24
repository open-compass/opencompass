PROMPT_TEMPLATE = {
    'xFinder-qwen1505':
    """<|System|>:{system}
<|User|>:{input}
<|Bot|>:""",
    'xFinder-llama38it':
    """<|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

""",
}
