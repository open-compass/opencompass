#!/bin/bash
cd ~/opencompass # 要进入到自己的目录，否则会在lizrunv2对应机器上的/workspace

export CUDA_VISIBLE_DEVICES=$1
export HF_ENDPOINT=https://hf-mirror.com




datasets=(
    "commonsenseqa_gen_7shot_cot"
    "commonsenseqa_gen_zero_shot_cot"
)


models=(
    # "Qwen2_5_7B_instruct_vllm_4gpus_low"
    "Qwen2_5_32B_instruct_vllm_4gpus_low"
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
    --work-dir ~/opencompass/commonsense \
    # --reuse 20250630_011117 \
    # --mode eval \

echo "Done inference"


