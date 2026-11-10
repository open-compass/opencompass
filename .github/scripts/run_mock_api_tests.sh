#!/usr/bin/env bash

set -u -o pipefail

REPORT_ROOT="${REPORT_ROOT:-${REPORT_DIR:?REPORT_ROOT or REPORT_DIR is required}}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
WORKSPACE="${GITHUB_WORKSPACE:-$(pwd)}"
MOCK_API="${MOCK_API:-/mnt/shared-storage-user/opencompass-shared/qa-llm-cicd/mock_chat_api.py}"
PORT="${MOCK_API_PORT:-26333}"
OPENCOMPASS_CMD="${OPENCOMPASS_CMD:-opencompass}"
read -r -a OPENCOMPASS_CMD_ARGS <<<"$OPENCOMPASS_CMD"

export COMPASS_DATA_CACHE="${COMPASS_DATA_CACHE:-/mnt/shared-storage-user/auto-eval-pipeline/opencompass/llmeval/compass_data_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/mnt/shared-storage-user/auto-eval-pipeline/opencompass/llmeval/hf_cache}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export LMUData="${LMUData:-/mnt/shared-storage-user/auto-eval-pipeline/vlmeval/LMUData}"

RUN_ROOT="$REPORT_ROOT/$RUN_ID"
LOG_ROOT="$RUN_ROOT/mock_api_logs"
mock_start_seconds=$SECONDS
mkdir -p "$LOG_ROOT"
export REPORT_DIR="$RUN_ROOT"
if [[ ! -d "$WORKSPACE" ]]; then
  echo "GitHub workspace is not visible in the rjob container: $WORKSPACE" >&2
  exit 1
fi
cd "$WORKSPACE"
export PYTHONPATH="$WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"
for suite in chat_obj_fullbench chat_sub_fullbench chat_longtext_fullbench; do
  if [[ ! -f "autotest/all/${suite}.py" ]]; then
    echo "Missing suite config in rjob workspace: autotest/all/${suite}.py" >&2
    exit 1
  fi
done
if [[ ! -f "$MOCK_API" ]]; then
  echo "Mock API server script not found: $MOCK_API" >&2
  exit 1
fi

HF_DATASETS_CACHE_SOURCE="$HF_DATASETS_CACHE"
hf_cache_root="$(mktemp -d /tmp/opencompass-mock-hf.XXXXXX)"
if ! python .github/scripts/prepare_hf_cache.py \
  "$HF_DATASETS_CACHE_SOURCE" "$hf_cache_root/datasets"; then
  echo 'Failed to prepare writable HF dataset cache.' >&2
  exit 1
fi
export HF_DATASETS_CACHE="$hf_cache_root/datasets"
echo "HF dataset cache source: $HF_DATASETS_CACHE_SOURCE"
echo "HF dataset cache writes: $HF_DATASETS_CACHE"

server_pid=''
data_link_created=0
declare -a cache_links_created=()
declare -a suite_pids=()
declare -a suite_names=(
  chat_obj_fullbench
  chat_sub_fullbench
  chat_longtext_fullbench
)

cleanup() {
  local pid
  for pid in "${suite_pids[@]:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${suite_pids[@]:-}"; do
    if [[ -n "$pid" ]]; then
      wait "$pid" 2>/dev/null || true
    fi
  done
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  if [[ "$data_link_created" -eq 1 && -L "$WORKSPACE/data" ]]; then
    if [[ "$(readlink "$WORKSPACE/data")" == "$COMPASS_DATA_CACHE/data" ]]; then
      rm -f "$WORKSPACE/data"
    fi
  fi
  if [[ -d "$WORKSPACE/data" && ! -L "$WORKSPACE/data" ]]; then
    local link name expected_target
    for link in "${cache_links_created[@]:-}"; do
      name="${link##*/}"
      expected_target="$COMPASS_DATA_CACHE/data/$name"
      if [[ -L "$link" && "$(readlink "$link")" == "$expected_target" ]]; then
        rm -f "$link"
      fi
    done
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ ! -d "$COMPASS_DATA_CACHE/data" ]]; then
  echo "Dataset cache data directory not found: $COMPASS_DATA_CACHE/data" >&2
  exit 1
fi

if [[ ! -e "$WORKSPACE/data" && ! -L "$WORKSPACE/data" ]]; then
  ln -s "$COMPASS_DATA_CACHE/data" "$WORKSPACE/data"
  data_link_created=1
elif [[ -L "$WORKSPACE/data" ]]; then
  workspace_data_target="$(readlink -f "$WORKSPACE/data")"
  cache_data_target="$(readlink -f "$COMPASS_DATA_CACHE/data")"
  if [[ "$workspace_data_target" != "$cache_data_target" ]]; then
    echo "Refusing to overlay dataset cache onto existing data symlink: $WORKSPACE/data -> $(readlink "$WORKSPACE/data")" >&2
    exit 1
  fi
elif [[ -d "$WORKSPACE/data" ]]; then
  while IFS= read -r -d '' cache_entry; do
    name="${cache_entry##*/}"
    link="$WORKSPACE/data/$name"
    if [[ ! -e "$link" && ! -L "$link" ]]; then
      if ! ln -s "$cache_entry" "$link"; then
        echo "Failed to link dataset cache entry: $link -> $cache_entry" >&2
        exit 1
      fi
      cache_links_created+=("$link")
    elif [[ ! -L "$link" || "$(readlink "$link")" != "$cache_entry" ]]; then
      echo "Preserving pre-existing workspace data entry (not overlaying cache): $link" >&2
    fi
  done < <(find "$COMPASS_DATA_CACHE/data" -mindepth 1 -maxdepth 1 -print0)
else
  echo "Workspace data path exists but is not a directory or symlink: $WORKSPACE/data" >&2
  exit 1
fi

echo "Starting shared mock API on port $PORT"
python "$MOCK_API" --type winrate --host 127.0.0.1 --port "$PORT" >"$LOG_ROOT/mock_chat_api.log" 2>&1 &
server_pid=$!

for _ in {1..30}; do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo 'Mock API exited before becoming healthy.'
    cat "$LOG_ROOT/mock_chat_api.log"
    exit 1
  fi
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
if ! curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo 'Mock API health check timed out.'
  cat "$LOG_ROOT/mock_chat_api.log"
  exit 1
fi

run_suite() {
  local suite="$1"
  local log="$LOG_ROOT/${suite}.log"
  local out="$RUN_ROOT/$suite"
  local start_seconds=$SECONDS

  echo "Starting $suite (output: $out)"
  CHAT_TYPE="$suite" \
    "${OPENCOMPASS_CMD_ARGS[@]}" "autotest/all/${suite}.py" \
    --work-dir "$out" --reuse \
    >"$log" 2>&1
  local status=$?
  if [[ "$status" -eq 0 ]] && grep -Eq \
    'failed with code [1-9][0-9]*| - ERROR - .*task .* fail|Skip eval for .* due to incomplete infer tasks\.' "$log"; then
    echo "Detected failed or incomplete OpenCompass child task in $suite log" >>"$log"
    status=1
  fi
  if [[ "$status" -eq 0 && -d "$out" ]] && grep -REn --include='*.out' \
    'Skip eval for .* due to incomplete infer tasks\.' "$out" >>"$log"; then
    echo "Detected incomplete Watch evaluation in $suite task logs" >>"$log"
    status=1
  fi
  chmod -R a+rwX "$out" 2>/dev/null || true
  echo "$status" >"$LOG_ROOT/${suite}.status"
  echo "$((SECONDS - start_seconds))" >"$LOG_ROOT/${suite}.seconds"
  echo "$suite finished in $((SECONDS - start_seconds))s (exit $status)"
  return "$status"
}

for suite in "${suite_names[@]}"; do
  run_suite "$suite" &
  suite_pids+=("$!")
done

overall_status=0
for i in "${!suite_names[@]}"; do
  suite="${suite_names[$i]}"
  if wait "${suite_pids[$i]}"; then
    echo "$suite completed successfully"
  else
    status=$?
    overall_status=1
    echo "$suite failed with status $status"
  fi
done

for suite in "${suite_names[@]}"; do
  status_file="$LOG_ROOT/${suite}.status"
  status='unknown'
  [[ -f "$status_file" ]] && status="$(<"$status_file")"
  echo "===== $suite (exit $status) log tail ====="
  if [[ "$status" == 0 ]]; then
    tail -n 25 "$LOG_ROOT/${suite}.log" || true
  else
    tail -n 100 "$LOG_ROOT/${suite}.log" || true
  fi
done

if [[ "$overall_status" -ne 0 ]]; then
  echo 'At least one mock API suite failed.'
fi
echo "$((SECONDS - mock_start_seconds))" >"$LOG_ROOT/total.seconds"
echo "Total mock API test time: $((SECONDS - mock_start_seconds))s"
exit "$overall_status"
