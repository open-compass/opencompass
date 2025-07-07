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
        abbr="gemma-2-9b-instruct-vllm",
        path="/mnt/pfs-gv8sxa/tts/dhg/workspace-ruc/chengxiang/datas/models/gemma2-9B-it",
        model_kwargs=dict(tensor_parallel_size=8,max_model_len=8192,gpu_memory_utilization=0.9),
        max_out_len=4096,
        max_seq_len=8192,
        batch_size=10000,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=8),
        meta_template=meta_template
)]
