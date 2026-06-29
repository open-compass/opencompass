# ReasonZoo Evaluation Framework

[![arXiv](https://img.shields.io/badge/arXiv-2508.15754-b31b1b.svg)](https://arxiv.org/abs/2508.15754)
[![Hugging Face Datasets](https://img.shields.io/badge/Hugging%20Face-Dataset-blue.svg)](https://huggingface.co/datasets/opencompass/ReasonZoo)

This repository contains the evaluation code for assessing language models on the ReasonZoo benchmark.  
The dataset are also provided on [Hugging Face Datasets](https://huggingface.co/datasets/opencompass/ReasonZoo).

---

## Repository Structure

```
.
├── infer/                 # Inference module
│   ├── models/           # Model implementations and configurations
│   ├── infer.py         # Main inference script
│   └── data_loader.py   # Data loading utilities
├── eval/                 # Evaluation module
│   ├── eval.py          # Main evaluation script
│   └── eval_utils.py    # Evaluation utilities and metrics
├── config/              # Configuration files
│   └── config.yaml      # Main configuration
└── data/                # Dataset directory
    ├── dailylogic/           # dailylogic puzzles
    ├── puzzle_and_code/ # Puzzle and coding tasks
    ├── physics/         # Physics problems
    ├── number_calculation/ # Numerical calculations
    ├── boolean_logic/  # Logic calculations
    ├── gradeschoolmath/   # Grade school math
    ├── formal_language/   # Formal language tasks
    ├── communication_code/   # Cipher and coding tasks
    └── operation_research/ # Operations research problems
```

## Usage

### Build a local sandbox
If you use sandbox/agent mode, build a sandbox server using [SandboxFusion](https://github.com/bytedance/SandboxFusion).
According to the instructions provided in https://github.com/bytedance/SandboxFusion, install SandboxFusion and launch it. 

1. Install SandboxFusion following the instructions at https://github.com/bytedance/SandboxFusion
2. Set up the sandbox environment:
   ```bash
   # Create a dedicated conda environment to avoid dependency conflicts
   # The sandbox environment must be named "sandbox-runtime"
   conda create -n sandbox-runtime python==3.11
   pip install -r runtime/python/requirement.txt
   
   # Install and run SandboxFusion
   pip install poetry
   poetry install
   mkdir -p docs/build
   make run-online
   ```
3. Update the sandbox URL in your configuration. We recommend http://localhost:8080 for simplicity.

### Running Inference

Use the following command to run inference on your models:

```bash
python infer/infer.py \
  --model_name $MODEL_NAME \
  --model $MODEL_NAME \
  --split $SPLIT \
  --mode $MODE \
  --code_mode $CODE_MODE \
  --output_dir $output_dir \
  --num_workers 128
```

**Parameters:**
- `MODEL_NAME`: Name of the model to evaluate (e.g., "Qwen3-8B")
- `SPLIT`: Dataset split to evaluate on (e.g., "dailylogic", "physics", "boolean_logic")
- `MODE`: Evaluation mode
- `CODE_MODE`: Code evaluation mode ("noncode" or "pot" or "sandbox" or "agent")
- `output_dir`: Directory to save inference results
- `num_workers`: Number of parallel workers for inference

### Running Evaluation

After inference, evaluate the results using:

```bash
python eval/eval.py \
  "$SOURCE_FOLDER" \
  "$TARGET_FOLDER" \
  "$CSV_FILE" \
  --use_llm_judge \
  --max_workers $MAX_WORKERS
```

> **Note:** If you're using the LLM judge feature, remember to configure your LLM service URL and API key in the `process_llm_evaluation()` function.

**Parameters:**
- `SOURCE_FOLDER`: Path to folder containing inference results
- `TARGET_FOLDER`: Path to save evaluation outputs
- `CSV_FILE`: Path to save evaluation summary CSV
- `--use_llm_judge`: Enable LLM-based evaluation for complex tasks
- `--max_workers`: Maximum number of parallel workers for evaluation

## Dataset Categories

The ReasonZoo evaluation covers multiple reasoning domains:

- **Logic & Puzzles**: dailylogic puzzles, logic calculations
- **Mathematics**: Grade school math, number calculations
- **Science**: Physics problems, operations research
- **Programming**: Cipher and code tasks, puzzle and code combinations
- **Formal Systems**: Formal language processing

## Configuration

The evaluation framework is highly configurable through `config/config.yaml`:

```yaml
# Response and data keys
response_key: 'response'
error_key: 'error'
prompt_key: 'prompt'

# Evaluation parameters
max_tokens: 32768
max_rounds: 10
save_prompt: True
```

## Key Features

- **Scalable Architecture**: Parallel processing with configurable worker counts
- **Multi-Model Support**: Easy integration of new language models
- **Comprehensive Evaluation**: Multiple reasoning task categories
- **Flexible Configuration**: Customizable evaluation parameters
- **LLM-based Judging**: Advanced evaluation for complex reasoning tasks

## Acknowledgements

This work builds on the core evaluation strategies pioneered by [KOR-Bench](https://github.com/KOR-Bench/KOR-Bench), in particular its task taxonomy and split-management framework, which we integrate into our end-to-end workflow. Our work further enriches these foundations with a high-throughput, parallel inference engine, an LLM-based adjudication layer, and both “program-of-thought” and function-calling agent modes within a sandboxed environment, etc. We acknowledge with gratitude the  [vLLM](https://github.com/vllm-project/vllm) and SandboxFusion [SandboxFusion](https://github.com/bytedance/SandboxFusion) projects for furnishing the high-performance inference framework and sandboxed execution environment, respectively, which were indispensable to this work. Together, these components enable fast, reproducible benchmarking across a wide variety of reasoning tasks.