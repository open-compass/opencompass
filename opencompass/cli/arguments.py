# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, asdict, field
from typing import Optional, Union


@dataclass
class Arguments:
    accelerator: str = None
    aliyun_cfg: str = '~/.aliyun.cfg'
    batch_size: int = 8
    config: str = None
    config_dir: str = 'configs'
    custom_dataset_data_type: str = None
    custom_dataset_infer_method: str = None
    custom_dataset_meta_path: str = None
    custom_dataset_path: str = None
    datasets: list = field(default_factory=list)
    debug: bool = False
    dlc: bool = False
    dry_run: bool = False
    dump_eval_details: bool = False
    generation_kwargs: Optional[dict] = field(default_factory=dict)
    hf_path: str = None
    hf_type: str = 'chat'
    lark: bool = False
    max_num_workers: int = 1
    max_out_len: int = 256
    max_seq_len: int = None
    max_workers_per_gpu: int = 1
    min_out_len: int = 1
    mode: str = 'all'
    model_kwargs: Optional[dict] = field(default_factory=dict)
    models: list = field(default_factory=list)
    num_gpus: int = 1
    pad_token_id: int = None
    partition: str = None
    peft_kwargs: Optional[dict] = field(default_factory=dict)
    peft_path: str = None
    qos: str = None
    quotatype: str = None
    retry: int = 2
    reuse: str = None
    slurm: bool = False
    stop_words: Optional[list] = field(default_factory=list)
    summarizer: str = None
    tokenizer_kwargs: Optional[dict] = field(default_factory=dict)
    tokenizer_path: str = None
    work_dir: str = 'outputs/default'

    # refer to: test_range in `opencompass.openicl.icl_dataset_reader.DatasetReader`
    limit: Optional[Union[int, float, str]] = None

    def __post_init__(self):
        ...


@dataclass
class ApiModelConfig:
    abbr: str       # The abbreviation of the model, e.g. 'Qwen-7B-Chat'
    path: str       # The path of the model, e.g. 'qwen/Qwen-7B-Chat', you can set it to the value of abbr in the format of OpenAI.
    openai_api_base: str   # The base URL of the OpenAI API, e.g. `http://127.0.0.1:8000/v1/chat/completions`

    meta_template: Union[str, dict] = None
    type: str = 'opencompass.models.OpenAIExtra'
    key: str = 'EMPTY'    # No need for APIs in the format of OpenAI.
    query_per_second: int = 1
    max_out_len: int = 2048
    max_seq_len: int = 4096
    batch_size: int = 8
    run_cfg: dict = field(default_factory=lambda: {"num_gpus": 0})
    temperature: float = 0.0        # It means the do_sample is False in OpenAI API.
    is_chat: bool = True


if __name__ == '__main__':
    task = dict(
        config='./configs/eval_openai_format_task_teval_v2_qwen.py',
        models=['Qwen-7B-Chat', 'Baichuan-7B-Chat'],
        generation_kwargs=dict(
            do_sample=True,
            temperature=0.3,
        )
    )

    args = Arguments(**task)
    print(asdict(args))
