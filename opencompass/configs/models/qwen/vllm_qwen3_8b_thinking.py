from opencompass.models import VLLM

models = [
    dict(
        type=VLLM,
        abbr='qwen3-8b-thinking-vllm',
        path='Qwen/Qwen3-8B',
        model_kwargs=dict(
            tensor_parallel_size=8,
            # Match ZeroEval's working configuration exactly
            max_model_len=32768,  # Keep ZeroEval's context length
            gpu_memory_utilization=0.8,  # Keep ZeroEval's memory setting
        ),
        max_out_len=32768,  # Keep ZeroEval's --max_tokens 32768
        max_seq_len=32768,  # Keep ZeroEval's sequence length
        batch_size=8,       # Keep ZeroEval's --batch_size 8
        generation_kwargs=dict(
            temperature=0.6,  # Match ZeroEval's --temperature 0.6
            top_p=0.95,       # Match ZeroEval's --top_p 0.95
            top_k=20,         # Match ZeroEval's --top_k 20
        ),
        run_cfg=dict(num_gpus=8, num_procs=1),
        stop_words=['<|im_end|>', '<|im_start|>'],  # Match ZeroEval's --use_imend_stop
    )
] 