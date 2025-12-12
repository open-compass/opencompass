def generate_chat(input_text, output_text=None, prefix_chat=None):
    chat = [
        {
            'role': 'user',
            'content': input_text
        },
    ]
    if output_text is not None:
        chat.append({'role': 'assistant', 'content': output_text})
    if prefix_chat is not None:
        chat = prefix_chat + chat
    return chat
