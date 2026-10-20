def get_chat_content(conversation, tokenize=False):
    if tokenize:
        raise NotImplementedError
    available_roles = ('user', 'assistant')
    content = ''
    for idx, item in enumerate(conversation):
        role = item['role']
        assert role in available_roles, role
        if idx % 2 == 0:
            assert role == 'user'
            content += '<s>'
            item_content = '[INST] %s [/INST]' % item['content']
            content += item_content
        else:
            assert role == 'assistant'
            item_content = ' %s</s>' % item['content']
            content += item_content
    return content


class GeneralPrompter(object):

    def __init__(self, apply_chat_template_func, response_split='[/INST]'):
        self.apply_chat_template_func = apply_chat_template_func
        self.response_split = response_split

    def generate_prompt(self, chat, tokenize=False, *args, **kargs) -> str:
        res = self.apply_chat_template_func(chat,
                                            tokenize=tokenize,
                                            *args,
                                            **kargs)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.response_split)[-1].strip()
