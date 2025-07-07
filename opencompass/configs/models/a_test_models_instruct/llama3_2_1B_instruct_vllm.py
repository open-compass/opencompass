from opencompass.models import VLLMwithChatTemplate
meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<HUMAN>: ', end='<eoh>\n', api_role='HUMAN'),
        dict(role='BOT',begin='<BOT>: ', end = '<eob>\n',api_role='BOT', generate=True),
    ],
    reserved_roles=[dict(role='SYSTEM', begin='<SYSTEM>: ',api_role='SYSTEM', end='<eosys>\n')],
)

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr="llama-3.2-1b-instruct-vllm",
        path="chengxiang/datas/models/llama3.2-1B-instruct",
        model_kwargs=dict(tensor_parallel_size=1,max_model_len=32768,gpu_memory_utilization=0.9),
        max_out_len=4096,
        max_seq_len=32768,
        batch_size=10000,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=1),
        meta_template=meta_template
    )]
