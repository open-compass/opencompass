from opencompass.models import MiniMaxChatCompletionV2

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
)

minimax_text_01 = [
    dict(abbr='MiniMax-Text-01',
         type=MiniMaxChatCompletionV2,
         path='MiniMax-Text-01',
         url='https://api.minimax.chat/v1/text/chatcompletion_v2',
         key='eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJHcm91cE5hbWUiOiLlmLvlmLvlk4jlk4giLCJVc2VyTmFtZSI6IuWYu-WYu-WTiOWTiCIsIkFjY291bnQiOiIiLCJTdWJqZWN0SUQiOiIxNzQzNTAzNjg0MDUxOTMxMjk0IiwiUGhvbmUiOiIxODEwMDE3NjQ5OCIsIkdyb3VwSUQiOiIxNzQzNTAzNjg0MDQzNTQyNjg2IiwiUGFnZU5hbWUiOiIiLCJNYWlsIjoiIiwiQ3JlYXRlVGltZSI6IjIwMjUtMDMtMDcgMTY6MzU6MjQiLCJUb2tlblR5cGUiOjEsImlzcyI6Im1pbmltYXgifQ.p4QeQoT6Sk7hOCUxIfPpPJqFstT61DoKy7RGh5BcfhyLDO-l_0WFcbfyA212fCwxbCJc-RWCwM2H8q0nfMvKricNY9cXoFcQp2wCqcW11Rq6fhIpE8FzQz4HTDSlHEec9mwGDdeTOqOXUALhgqYho2anH2VP8aoARuXOSY8He_KyBxHRvODucarRkYWOjMUd20DRni7SGm8n_Gi2B_DacGW1ie60U8t2Aahna5h7pGFqudP0r-_YUtDabuqPX0Vo_EKPu1ZyVrY7jP_YT4FVx6AtYBrZcTzcq-KTm_F86-1sioUXzz9oPo3JFFJudhwbVQYrySB5jqJUJNoXP8OzDA',
         # You need to set your own API key
         meta_template=api_meta_template,
         query_per_second=3,
         retry=1,
         max_out_len=4096, max_seq_len=32768, batch_size=2),
]
