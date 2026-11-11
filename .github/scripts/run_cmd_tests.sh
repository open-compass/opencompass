#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 || -z "$1" ]]; then
  echo "Usage: bash .github/scripts/run_cmd_tests.sh <output-directory>" >&2
  exit 1
fi

# The workflow activates the environment before invoking this script.
output_dir="$1"

opencompass \
  --models hf_qwen_2_5_7b \
  --datasets demo_gsm8k_chat_gen \
  --work-dir "$output_dir/regression_result1" \
  --dump-res-length

opencompass \
  --models hf_qwen_2_5_7b \
  --datasets demo_gsm8k_chat_gen \
  --work-dir "$output_dir/regression_result2" \
  --max-num-workers 2 \
  --dump-res-length

opencompass \
  --models hf_qwen_2_5_7b \
  --datasets demo_gsm8k_chat_gen \
  -a lmdeploy \
  --work-dir "$output_dir/regression_result3" \
  --max-num-workers 2 \
  --dump-res-length
