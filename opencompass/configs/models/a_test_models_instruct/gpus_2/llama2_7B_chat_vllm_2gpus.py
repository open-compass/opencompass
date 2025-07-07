from opencompass.models import VLLMwithChatTemplate

# gpu = torch.cuda.device_count()
meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<HUMAN>: ', end='<eoh>\n', api_role='HUMAN'),
        dict(role='BOT', begin='<BOT>: ', end='<eob>\n',
             api_role='BOT', generate=True),
    ],
    reserved_roles=[dict(role='SYSTEM', begin='<SYSTEM>: ',
                         api_role='SYSTEM', end='<eosys>\n')],
)

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr="llama-2-7b-chat-vllm",
        path="chengxiang/datas/models/llama2-7B-chat",
        model_kwargs=dict(tensor_parallel_size=2, max_model_len=4096,gpu_memory_utilization=0.9),
        max_out_len=2048,
        max_seq_len=4096,
        batch_size=10000,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=2),
        meta_template=meta_template
    )]
