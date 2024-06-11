# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from dataclasses import dataclass, asdict, field


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
    generation_kwargs: dict = field(default_factory=dict)
    hf_path: str = None
    hf_type: str = 'chat'
    lark: bool = False
    max_num_workers: int = 1
    max_out_len: int = 256
    max_seq_len: int = None
    max_workers_per_gpu: int = 1
    min_out_len: int = 1
    mode: str = 'all'
    model_kwargs: dict = field(default_factory=dict)
    models: list = field(default_factory=list)
    num_gpus: int = 1
    pad_token_id: int = None
    partition: str = None
    peft_kwargs: dict = field(default_factory=dict)
    peft_path: str = None
    qos: str = None
    quotatype: str = None
    retry: int = 2
    reuse: str = None
    slurm: bool = False
    stop_words: list = field(default_factory=list)
    summarizer: str = None
    tokenizer_kwargs: dict = field(default_factory=dict)
    tokenizer_path: str = None
    work_dir: str = None

    def __post_init__(self):
        ...
