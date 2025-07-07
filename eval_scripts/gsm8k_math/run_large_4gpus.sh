#!/bin/bash
cd ~/opencompass # 要进入到自己的目录

export CUDA_VISIBLE_DEVICES=$1
export HF_ENDPOINT=https://hf-mirror.com


# datasets=(
#     "gsm8k_gen_original_v2"
#     "gsm8k_gen_hard_v2" 
#     "gsm8k_topk"
#     "gsm8k_votek"
#     "gsm8k_random"
#     "gsm8k_dpp"
#     "gsm8k_mmr"
#     "gsm8k_gen_explora"
# )

# datasets=(
    # "gsm8k_gen_original_xxx_v1"
    # "gsm8k_gen_original_xxx_v2"
    # "gsm8k_gen_original_xxx_v3"
    # "gsm8k_gen_original_v2"
    # "gsm8k_gen_original_6shot_v2"
    # "gsm8k_gen_original_4shot_v2"
    # "gsm8k_gen_original_2shot_v2"
    # "gsm8k_gen_hard_v2" 
    # "gsm8k_gen_easy_v2"
    # "gsm8k_gen_zo_v2_original"
    # "gsm8k_gen_zo_v2"
    # "gsm8k_gen_error_8shot"
    # "gsm8k_gen_original_enhance_v2"
    # "gsm8k_gen_original_enhance_6shot_v2"
    # "gsm8k_gen_original_enhance_4shot_v2"
    # "gsm8k_gen_original_enhance_2shot_v2"
# )

# datasets=(
#     "gsm8k_gen_original_enhance_v2"
#     "gsm8k_gen_original_enhance_6shot_v2"
#     "gsm8k_gen_original_enhance_4shot_v2"
#     "gsm8k_gen_original_enhance_2shot_v2"
#     "gsm8k_gen_slow_think_v1"
#     "gsm8k_gen_slow_think_2shot_v1"
#     "gsm8k_gen_slow_think_3shot_v1"
#     "gsm8k_gen_slow_think_v1_turn0"
#     "gsm8k_gen_slow_think_v1_xxx_25"
#     "gsm8k_gen_slow_think_v1_xxx_50"
#     "gsm8k_gen_slow_think_v1_xxx_80"
#     "gsm8k_gen_slow_think_v1_shuffle_100"
#     "gsm8k_gen_slow_think_v1_xxx_100"
# )

# datasets=(
#     "gsm8k_gen_slow_think_2shot_noise_25"
#     "gsm8k_gen_slow_think_2shot_noise_50"
#     "gsm8k_gen_slow_think_2shot_noise_80"
#     "gsm8k_gen_slow_think_2shot_noise_100"
#     "gsm8k_gen_slow_think_2shot_shuffle"
#     "gsm8k_gen_slow_think_3shot_noise_25"
#     "gsm8k_gen_slow_think_3shot_noise_50"
#     "gsm8k_gen_slow_think_3shot_noise_80"
#     "gsm8k_gen_slow_think_3shot_noise_100"
#     "gsm8k_gen_slow_think_3shot_shuffle"
# )

# datasets=(
#     "math_gen_level1_v2"
#     "math_gen_level5_v2"
#     "math_gen_slow_think_from_math"
#     "math_gen_slow_think_xxx"
#     "math_gen_slow_think_2shot"
#     "math_gen_slow_think_3shot"
#     "math_gen_zo_v2"
# )

datasets=(
    "gsm8k_qwen_enhance"
    # "gsm8k_gen_original_v2"
    # "gsm8k_gen_zo_v2_original"
    # "gsm8k_gen_zo_v2"
    # "math_qwen_enhance"
    # "math_gen_zo_v2"
)

# models=(
#     "Qwen2_5_7B_instruct_vllm_4gpus"
#     "Qwen2_5_72B_instruct_vllm_4gpus"
# )
models=(
    # "Qwen2_5_7B_base_vllm_4gpus"
    "Qwen2_5_72B_base_vllm_4gpus"
)
# models=(
#     "Qwen2_5_32B_instruct_vllm_2gpus"
#     "Qwen2_5_72B_instruct_vllm_2gpus"
#     "llama3_3_70B_instruct_vllm_2gpus"
# )

# models=(
#     "ministral_8B_instruct_vllm_4gpus"
#     "llama3_1_8B_instruct_vllm_4gpus"
#     "gemma2_9B_instruct_vllm_4gpus"
#     "Qwen2_5_7B_instruct_vllm_4gpus"
#     "Qwen2_5_14B_instruct_vllm_4gpus"
#     "llama3_2_1B_instruct_vllm_4gpus"
#     "llama3_2_3B_instruct_vllm_4gpus"
#     "gemma2_2B_instruct_vllm_4gpus"
#     "Qwen2_5_0_5_B_instruct_vllm_2gpus"
#     "Qwen2_5_1_5_B_instruct_vllm_4gpus"
#     "Qwen2_5_3_B_instruct_vllm_4gpus"
#     # "llama2_7B_chat_vllm_4gpus.py"
#     # "Qwen_7B_chat_vllm_4gpus"
#     "llama3_8B_instruct_vllm_4gpus"
# )
# models=(
#     "llama3_1_8B_distill_vllm_4gpus"
#     "Qwen2_5_7B_distill_vllm_4gpus"
#     "Qwen2_5_14B_distill_vllm_4gpus"
# )
# models=(
#     "llama3_2_1B_instruct_vllm_4gpus"
#     "llama3_2_3B_instruct_vllm_4gpus"
#     "gemma2_2B_instruct_vllm_4gpus"
#     "Qwen2_5_0_5_B_instruct_vllm_2gpus"
#     "Qwen2_5_1_5_B_instruct_vllm_4gpus"
#     "Qwen2_5_3_B_instruct_vllm_4gpus"
#     "llama2_7B_chat_vllm_4gpus.py"
#     "Qwen_7B_chat_vllm_4gpus"
#     "llama3_8B_instruct_vllm_4gpus"
# )

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
    --work-dir ~/opencompass/math_gsm \
    # --reuse 20250630_011117 \
    # --mode eval \

echo "Done inference"


