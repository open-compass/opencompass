meta_template_dict = dict(
    internlm = [
        dict(role='system', begin='<|System|>:', end='\n'),
        dict(role='user', begin='<|User|>:', end='\n'),
        dict(
            role='assistant',
            begin='<|Bot|>:',
            end='<eoa>\n',
            generate=True)
    ],
)
