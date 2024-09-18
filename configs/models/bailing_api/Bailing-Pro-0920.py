from opencompass.models import BaiLingAPI

api_meta_template = dict(
    round=[
        dict(role="HUMAN", api_role="HUMAN"),
        dict(role="BOT", api_role="BOT", generate=False),
    ],
    reserved_roles=[dict(role="SYSTEM", api_role="SYSTEM")],
)

models = [
    dict(
        path="Bailing-Pro-0920",
        token="",  # set your token
        url="https://bailingchat.alipay.com/chat/completions",
        type=BaiLingAPI,
        meta_template=api_meta_template,
        query_per_second=1,
        max_out_len=1024,
        batch_size=1,
    ),
]
