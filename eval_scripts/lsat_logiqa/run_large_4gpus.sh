#!/bin/bash
cd ~/opencompass # 要进入到自己的目录，否则会在lizrunv2对应机器上的/workspace

export CUDA_VISIBLE_DEVICES=$1
export HF_ENDPOINT=https://hf-mirror.com




datasets=(
    "lsta_logiqa_gen_zero"
    "lsta_logiqa_gen_few_shot"
)


models=(
    "Qwen2_5_7B_instruct_vllm_4gpus"
    Qwen2_5_32B_instruct_vllm_4gpus
    "Qwen2_5_72B_instruct_vllm_4gpus"
)


datasets_str=$(IFS=" " ; echo "${datasets[*]}")
models_str=$(IFS=" " ; echo "${models[*]}")

echo "Start inferenve use vllm"

VLLM_WORKER_MULTIPROC_METHOD=spawn \
python run.py \
    --datasets $datasets_str \
    --models $models_str \
    --max-num-workers 1 \
    --max-workers-per-gpu 1 \
    --generation-kwargs seed=42 \
    --work-dir ~/opencompass/lsat_logiqa \
    # --reuse 20250630_011117 \
    # --mode eval \

echo "Done inference"


