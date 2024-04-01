api_meta_template=dict(
    reserved_roles=[
        dict(api_role='SYSTEM',
            role='SYSTEM'),
        ],
    round=[
        dict(api_role='HUMAN',
            role='HUMAN'),
        dict(api_role='BOT',
            generate=True,
            role='BOT'),
        ])
baichuan=dict(
    abbr='baichuan',
    batch_size=1,
    generation_kwargs=dict(
        do_sample=True),
    max_out_len=2048,
    max_seq_len=4096,
    meta_template=dict(
        reserved_roles=[
            dict(api_role='SYSTEM',
                role='SYSTEM'),
            ],
        round=[
            dict(api_role='HUMAN',
                role='HUMAN'),
            dict(api_role='BOT',
                generate=True,
                role='BOT'),
            ]),
    model_kwargs=dict(
        device_map='auto',
        trust_remote_code=True),
    path='THUDM/chatglm3-6b',
    run_cfg=dict(
        num_gpus=1,
        num_procs=1),
    tokenizer_kwargs=dict(
        padding_side='left',
        truncation_side='left',
        trust_remote_code=True),
    tokenizer_path='THUDM/chatglm3-6b',
    type='opencompass.models.HuggingFaceChatGLM3')
chatglm3=dict(
    abbr='chatglm3-6b-32k-hf',
    batch_size=1,
    generation_kwargs=dict(
        do_sample=True),
    max_out_len=2048,
    max_seq_len=4096,
    meta_template=dict(
        reserved_roles=[
            dict(api_role='SYSTEM',
                role='SYSTEM'),
            ],
        round=[
            dict(api_role='HUMAN',
                role='HUMAN'),
            dict(api_role='BOT',
                generate=True,
                role='BOT'),
            ]),
    model_kwargs=dict(
        device_map='auto',
        trust_remote_code=True),
    path='THUDM/chatglm3-6b',
    run_cfg=dict(
        num_gpus=1,
        num_procs=1),
    tokenizer_kwargs=dict(
        padding_side='left',
        truncation_side='left',
        trust_remote_code=True),
    tokenizer_path='THUDM/chatglm3-6b',
    type='opencompass.models.HuggingFaceChatGLM3')
datasets=[
    dict(abbr='creationv2_zh_test',
        eval_cfg=dict(
            evaluator=dict(
                infer_order='double',
                prompt_template=dict(
                    template=dict(
                        round=[
                            dict(prompt='\n请根据提供的 评分要求，用户问题 以及 相应的两个回答（回答1，回答2），判断两个回答中哪一个更好。\n评分要求（重要性依次递减）:\n1. 好的回答必须首先符合用户问题里的各种需求，不能跑题 \n2. 好的回答必须具有逻辑连贯性，围绕一个中心进行回答\n3. 好的回答必须具有创造性的词语和表达丰富度\n\n[用户问题]\n{question}\n\n\n[回答1开始]\n{prediction}\n[回答1结束]\n\n[回答2开始]\n{prediction2}\n[回答2结束]\n\n根据评分要求，在以下 3 个选项中做出选择:\nA. 回答1更好\nB. 回答2更好\nC. 回答1、2平局\n并提供你的解释原因。\n\n如果你认为回答1更好，你的输出应形如：\n选择：A\n原因：blahblah blahblah\n\n\n如果你认为回答2更好，你的输出应形如：\n选择：B\n原因：blahblah blahblah\n\n\n如果你认为回答1、2打成平手，你的输出应形如：\n选择：C\n原因：blahblah blahblah\n\n',
                                role='HUMAN'),
                            ]),
                    type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
                type='opencompass.openicl.icl_evaluator.LMEvaluator'),
            pred_role='BOT'),
        infer_cfg=dict(
            inferencer=dict(
                max_out_len=2048,
                max_seq_len=4096,
                type='opencompass.openicl.icl_inferencer.GenInferencer'),
            prompt_template=dict(
                template=dict(
                    round=[
                        dict(prompt='{question}',
                            role='HUMAN'),
                        ]),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                type='opencompass.openicl.icl_retriever.ZeroRetriever')),
        name='creationv2_zh_test',
        path='data/subjective/compass_arena',
        reader_cfg=dict(
            input_columns=[
                'question',
                'ref',
                ],
            output_column='judge'),
        type='opencompass.datasets.CompassArenaDataset'),
    ]
eval=dict(
    partitioner=dict(
        base_models=[
            dict(abbr='baichuan',
                batch_size=1,
                generation_kwargs=dict(
                    do_sample=True),
                max_out_len=2048,
                max_seq_len=4096,
                meta_template=dict(
                    reserved_roles=[
                        dict(api_role='SYSTEM',
                            role='SYSTEM'),
                        ],
                    round=[
                        dict(api_role='HUMAN',
                            role='HUMAN'),
                        dict(api_role='BOT',
                            generate=True,
                            role='BOT'),
                        ]),
                model_kwargs=dict(
                    device_map='auto',
                    trust_remote_code=True),
                path='THUDM/chatglm3-6b',
                run_cfg=dict(
                    num_gpus=1,
                    num_procs=1),
                tokenizer_kwargs=dict(
                    padding_side='left',
                    truncation_side='left',
                    trust_remote_code=True),
                tokenizer_path='THUDM/chatglm3-6b',
                type='opencompass.models.HuggingFaceChatGLM3'),
            ],
        compare_models=[
            dict(abbr='chatglm3-6b-32k-hf',
                batch_size=1,
                generation_kwargs=dict(
                    do_sample=True),
                max_out_len=2048,
                max_seq_len=4096,
                meta_template=dict(
                    reserved_roles=[
                        dict(api_role='SYSTEM',
                            role='SYSTEM'),
                        ],
                    round=[
                        dict(api_role='HUMAN',
                            role='HUMAN'),
                        dict(api_role='BOT',
                            generate=True,
                            role='BOT'),
                        ]),
                model_kwargs=dict(
                    device_map='auto',
                    trust_remote_code=True),
                path='THUDM/chatglm3-6b',
                run_cfg=dict(
                    num_gpus=1,
                    num_procs=1),
                tokenizer_kwargs=dict(
                    padding_side='left',
                    truncation_side='left',
                    trust_remote_code=True),
                tokenizer_path='THUDM/chatglm3-6b',
                type='opencompass.models.HuggingFaceChatGLM3'),
            dict(abbr='qwen',
                batch_size=1,
                generation_kwargs=dict(
                    do_sample=True),
                max_out_len=2048,
                max_seq_len=4096,
                meta_template=dict(
                    reserved_roles=[
                        dict(api_role='SYSTEM',
                            role='SYSTEM'),
                        ],
                    round=[
                        dict(api_role='HUMAN',
                            role='HUMAN'),
                        dict(api_role='BOT',
                            generate=True,
                            role='BOT'),
                        ]),
                model_kwargs=dict(
                    device_map='auto',
                    trust_remote_code=True),
                path='THUDM/chatglm3-6b',
                run_cfg=dict(
                    num_gpus=1,
                    num_procs=1),
                tokenizer_kwargs=dict(
                    padding_side='left',
                    truncation_side='left',
                    trust_remote_code=True),
                tokenizer_path='THUDM/chatglm3-6b',
                type='opencompass.models.HuggingFaceChatGLM3'),
            dict(abbr='baichuan',
                batch_size=1,
                generation_kwargs=dict(
                    do_sample=True),
                max_out_len=2048,
                max_seq_len=4096,
                meta_template=dict(
                    reserved_roles=[
                        dict(api_role='SYSTEM',
                            role='SYSTEM'),
                        ],
                    round=[
                        dict(api_role='HUMAN',
                            role='HUMAN'),
                        dict(api_role='BOT',
                            generate=True,
                            role='BOT'),
                        ]),
                model_kwargs=dict(
                    device_map='auto',
                    trust_remote_code=True),
                path='THUDM/chatglm3-6b',
                run_cfg=dict(
                    num_gpus=1,
                    num_procs=1),
                tokenizer_kwargs=dict(
                    padding_side='left',
                    truncation_side='left',
                    trust_remote_code=True),
                tokenizer_path='THUDM/chatglm3-6b',
                type='opencompass.models.HuggingFaceChatGLM3'),
            ],
        judge_models=[
            dict(abbr='chatglm3-6b-32k-hf',
                batch_size=1,
                generation_kwargs=dict(
                    do_sample=True),
                max_out_len=2048,
                max_seq_len=4096,
                meta_template=dict(
                    reserved_roles=[
                        dict(api_role='SYSTEM',
                            role='SYSTEM'),
                        ],
                    round=[
                        dict(api_role='HUMAN',
                            role='HUMAN'),
                        dict(api_role='BOT',
                            generate=True,
                            role='BOT'),
                        ]),
                model_kwargs=dict(
                    device_map='auto',
                    trust_remote_code=True),
                path='THUDM/chatglm3-6b',
                run_cfg=dict(
                    num_gpus=1,
                    num_procs=1),
                tokenizer_kwargs=dict(
                    padding_side='left',
                    truncation_side='left',
                    trust_remote_code=True),
                tokenizer_path='THUDM/chatglm3-6b',
                type='opencompass.models.HuggingFaceChatGLM3'),
            dict(abbr='baichuan',
                batch_size=1,
                generation_kwargs=dict(
                    do_sample=True),
                max_out_len=2048,
                max_seq_len=4096,
                meta_template=dict(
                    reserved_roles=[
                        dict(api_role='SYSTEM',
                            role='SYSTEM'),
                        ],
                    round=[
                        dict(api_role='HUMAN',
                            role='HUMAN'),
                        dict(api_role='BOT',
                            generate=True,
                            role='BOT'),
                        ]),
                model_kwargs=dict(
                    device_map='auto',
                    trust_remote_code=True),
                path='THUDM/chatglm3-6b',
                run_cfg=dict(
                    num_gpus=1,
                    num_procs=1),
                tokenizer_kwargs=dict(
                    padding_side='left',
                    truncation_side='left',
                    trust_remote_code=True),
                tokenizer_path='THUDM/chatglm3-6b',
                type='opencompass.models.HuggingFaceChatGLM3'),
            ],
        mode='m2n',
        type='opencompass.partitioners.sub_naive.SubjectiveNaivePartitioner'),
    runner=dict(
        max_num_workers=32,
        partition='llm_dev2',
        quotatype='auto',
        task=dict(
            type='opencompass.tasks.subjective_eval.SubjectiveEvalTask'),
        type='opencompass.runners.SlurmSequentialRunner'))
gpt4=dict(
    abbr='gpt4-turbo',
    batch_size=4,
    key='',
    max_out_len=2048,
    max_seq_len=4096,
    meta_template=dict(
        reserved_roles=[
            dict(api_role='SYSTEM',
                role='SYSTEM'),
            ],
        round=[
            dict(api_role='HUMAN',
                role='HUMAN'),
            dict(api_role='BOT',
                generate=True,
                role='BOT'),
            ]),
    path='gpt-4-1106-preview',
    query_per_second=1,
    retry=20,
    temperature=1,
    type='opencompass.models.OpenAI')
infer=dict(
    partitioner=dict(
        max_task_size=10000,
        strategy='split',
        type='opencompass.partitioners.SizePartitioner'),
    runner=dict(
        max_num_workers=256,
        partition='llm_dev2',
        quotatype='auto',
        task=dict(
            type='opencompass.tasks.OpenICLInferTask'),
        type='opencompass.runners.SlurmSequentialRunner'))
judge_models=[
    dict(abbr='chatglm3-6b-32k-hf',
        batch_size=1,
        generation_kwargs=dict(
            do_sample=True),
        max_out_len=2048,
        max_seq_len=4096,
        meta_template=dict(
            reserved_roles=[
                dict(api_role='SYSTEM',
                    role='SYSTEM'),
                ],
            round=[
                dict(api_role='HUMAN',
                    role='HUMAN'),
                dict(api_role='BOT',
                    generate=True,
                    role='BOT'),
                ]),
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True),
        path='THUDM/chatglm3-6b',
        run_cfg=dict(
            num_gpus=1,
            num_procs=1),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True),
        tokenizer_path='THUDM/chatglm3-6b',
        type='opencompass.models.HuggingFaceChatGLM3'),
    dict(abbr='baichuan',
        batch_size=1,
        generation_kwargs=dict(
            do_sample=True),
        max_out_len=2048,
        max_seq_len=4096,
        meta_template=dict(
            reserved_roles=[
                dict(api_role='SYSTEM',
                    role='SYSTEM'),
                ],
            round=[
                dict(api_role='HUMAN',
                    role='HUMAN'),
                dict(api_role='BOT',
                    generate=True,
                    role='BOT'),
                ]),
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True),
        path='THUDM/chatglm3-6b',
        run_cfg=dict(
            num_gpus=1,
            num_procs=1),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True),
        tokenizer_path='THUDM/chatglm3-6b',
        type='opencompass.models.HuggingFaceChatGLM3'),
    ]
models=[
    dict(abbr='chatglm3-6b-32k-hf',
        batch_size=1,
        generation_kwargs=dict(
            do_sample=True),
        max_out_len=2048,
        max_seq_len=4096,
        meta_template=dict(
            reserved_roles=[
                dict(api_role='SYSTEM',
                    role='SYSTEM'),
                ],
            round=[
                dict(api_role='HUMAN',
                    role='HUMAN'),
                dict(api_role='BOT',
                    generate=True,
                    role='BOT'),
                ]),
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True),
        path='THUDM/chatglm3-6b',
        run_cfg=dict(
            num_gpus=1,
            num_procs=1),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True),
        tokenizer_path='THUDM/chatglm3-6b',
        type='opencompass.models.HuggingFaceChatGLM3'),
    dict(abbr='qwen',
        batch_size=1,
        generation_kwargs=dict(
            do_sample=True),
        max_out_len=2048,
        max_seq_len=4096,
        meta_template=dict(
            reserved_roles=[
                dict(api_role='SYSTEM',
                    role='SYSTEM'),
                ],
            round=[
                dict(api_role='HUMAN',
                    role='HUMAN'),
                dict(api_role='BOT',
                    generate=True,
                    role='BOT'),
                ]),
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True),
        path='THUDM/chatglm3-6b',
        run_cfg=dict(
            num_gpus=1,
            num_procs=1),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True),
        tokenizer_path='THUDM/chatglm3-6b',
        type='opencompass.models.HuggingFaceChatGLM3'),
    dict(abbr='baichuan',
        batch_size=1,
        generation_kwargs=dict(
            do_sample=True),
        max_out_len=2048,
        max_seq_len=4096,
        meta_template=dict(
            reserved_roles=[
                dict(api_role='SYSTEM',
                    role='SYSTEM'),
                ],
            round=[
                dict(api_role='HUMAN',
                    role='HUMAN'),
                dict(api_role='BOT',
                    generate=True,
                    role='BOT'),
                ]),
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True),
        path='THUDM/chatglm3-6b',
        run_cfg=dict(
            num_gpus=1,
            num_procs=1),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True),
        tokenizer_path='THUDM/chatglm3-6b',
        type='opencompass.models.HuggingFaceChatGLM3'),
    ]
qwen=dict(
    abbr='qwen',
    batch_size=1,
    generation_kwargs=dict(
        do_sample=True),
    max_out_len=2048,
    max_seq_len=4096,
    meta_template=dict(
        reserved_roles=[
            dict(api_role='SYSTEM',
                role='SYSTEM'),
            ],
        round=[
            dict(api_role='HUMAN',
                role='HUMAN'),
            dict(api_role='BOT',
                generate=True,
                role='BOT'),
            ]),
    model_kwargs=dict(
        device_map='auto',
        trust_remote_code=True),
    path='THUDM/chatglm3-6b',
    run_cfg=dict(
        num_gpus=1,
        num_procs=1),
    tokenizer_kwargs=dict(
        padding_side='left',
        truncation_side='left',
        trust_remote_code=True),
    tokenizer_path='THUDM/chatglm3-6b',
    type='opencompass.models.HuggingFaceChatGLM3')
subjective_datasets=[
    dict(abbr='creationv2_zh_test',
        eval_cfg=dict(
            evaluator=dict(
                infer_order='double',
                prompt_template=dict(
                    template=dict(
                        round=[
                            dict(prompt='\n请根据提供的 评分要求，用户问题 以及 相应的两个回答（回答1，回答2），判断两个回答中哪一个更好。\n评分要求（重要性依次递减）:\n1. 好的回答必须首先符合用户问题里的各种需求，不能跑题 \n2. 好的回答必须具有逻辑连贯性，围绕一个中心进行回答\n3. 好的回答必须具有创造性的词语和表达丰富度\n\n[用户问题]\n{question}\n\n\n[回答1开始]\n{prediction}\n[回答1结束]\n\n[回答2开始]\n{prediction2}\n[回答2结束]\n\n根据评分要求，在以下 3 个选项中做出选择:\nA. 回答1更好\nB. 回答2更好\nC. 回答1、2平局\n并提供你的解释原因。\n\n如果你认为回答1更好，你的输出应形如：\n选择：A\n原因：blahblah blahblah\n\n\n如果你认为回答2更好，你的输出应形如：\n选择：B\n原因：blahblah blahblah\n\n\n如果你认为回答1、2打成平手，你的输出应形如：\n选择：C\n原因：blahblah blahblah\n\n',
                                role='HUMAN'),
                            ]),
                    type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
                type='opencompass.openicl.icl_evaluator.LMEvaluator'),
            pred_role='BOT'),
        infer_cfg=dict(
            inferencer=dict(
                max_out_len=2048,
                max_seq_len=4096,
                type='opencompass.openicl.icl_inferencer.GenInferencer'),
            prompt_template=dict(
                template=dict(
                    round=[
                        dict(prompt='{question}',
                            role='HUMAN'),
                        ]),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                type='opencompass.openicl.icl_retriever.ZeroRetriever')),
        name='creationv2_zh_test',
        path='data/subjective/compass_arena',
        reader_cfg=dict(
            input_columns=[
                'question',
                'ref',
                ],
            output_column='judge'),
        type='opencompass.datasets.CompassArenaDataset'),
    ]
summarizer=dict(
    summary_type='half_add',
    type='opencompass.summarizers.CompassArenaSummarizer')
work_dir='outputs/compass_arena/20240308_135229'