def _convert_vlmeval_content(items):
    if not isinstance(items, list):
        raise TypeError('VLMEvalKit message content must be a list.')
    content = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise TypeError(f'VLMEvalKit content item {i} must be a dict.')
        item_type = item.get('type')
        if item_type == 'text':
            content.append({'type': 'text', 'text': item['value']})
        elif item_type == 'image':
            content.append({'type': 'image', 'image_url': item['value']})
        else:
            raise ValueError(
                f'Unsupported VLMEvalKit prompt type: {item_type!r}. '
                'This bridge currently supports text and image blocks only.')
    return content


def convert_vlmeval_prompt(prompt):
    if not isinstance(prompt, list) or not prompt:
        raise TypeError(
            'VLMEvalKit build_prompt() must return a non-empty list.')
    if all(isinstance(item, dict) and 'type' in item for item in prompt):
        messages = []
        for item in prompt:
            role = item.get('role', 'user')
            if role not in {'system', 'user', 'assistant'}:
                raise ValueError(
                    f'Unsupported VLMEvalKit prompt role: {role!r}.')
            content = _convert_vlmeval_content([item])[0]
            if messages and messages[-1]['role'] == role:
                messages[-1]['content'].append(content)
            else:
                messages.append({'role': role, 'content': [content]})
        return messages
    if all(
            isinstance(item, dict) and 'role' in item and 'content' in item
            for item in prompt):
        messages = []
        for item in prompt:
            role = item['role']
            if role not in {'system', 'user', 'assistant'}:
                raise ValueError(
                    f'Unsupported VLMEvalKit prompt role: {role!r}.')
            messages.append({
                'role':
                role,
                'content':
                _convert_vlmeval_content(item['content'])
            })
        return messages
    raise TypeError(
        'VLMEvalKit build_prompt() must return flat content blocks or '
        'role/content messages.')
