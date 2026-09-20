# Code Evaluation

This page uses `humaneval` and `mbpp` to explain pass@1 / pass@k configuration. When code must actually execute, as with multilingual `humaneval-x` or `DS1000`, use an independent Docker code-evaluation service to avoid running model-generated code in the development environment.

## pass@1

To generate one reply and evaluate pass@1, use [opencompass/configs/datasets/humaneval/humaneval_gen_8e312c.py](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/humaneval/humaneval_gen_8e312c.py) and [opencompass/configs/datasets/mbpp/deprecated_mbpp_gen_1e1056.py](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/mbpp/deprecated_mbpp_gen_1e1056.py), following the general [Quick Start](../get_started/quick_start.md).

For multilingual evaluation, see [Code Execution Service](#code-execution-service) below.

## pass@k

To generate multiple replies per example for pass@k, use one of the following approaches. Both examples generate 10 replies.

### General Case

Most models support the Hugging Face generation argument `num_return_sequences`, which can directly produce multiple replies:

```python
from opencompass.datasets import MBPPDatasetV2, MBPPPassKEvaluator

with read_base():
    from .datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets
    from .datasets.mbpp.deprecated_mbpp_gen_1e1056 import mbpp_datasets

mbpp_datasets[0]['type'] = MBPPDatasetV2
mbpp_datasets[0]['eval_cfg']['evaluator']['type'] = MBPPPassKEvaluator
mbpp_datasets[0]['reader_cfg']['output_column'] = 'test_column'

datasets = []
datasets += humaneval_datasets
datasets += mbpp_datasets

models = [
    dict(
        type=HuggingFaceCausalLM,
        ...,
        generation_kwargs=dict(
            num_return_sequences=10,
            do_sample=True,
            top_p=0.95,
            temperature=0.8,
        ),
        ...,
    )
]
```

MBPP needs corresponding changes to `type`, `eval_cfg.evaluator.type`, and `reader_cfg.output_column` for the new behavior.

Replies must be stochastic, so set `generation_kwargs`, especially `num_return_sequences`. It must be at least k because pass@k is a probability estimate.

See [examples/eval_code_passk.py](https://github.com/open-compass/opencompass/blob/main/examples/eval_code_passk.py) for a concrete configuration.

### Models Without Multiple-Reply Support

For an API or incomplete Hugging Face model that cannot return multiple replies, repeat the dataset instead:

```python
from opencompass.datasets import MBPPDatasetV2, MBPPPassKEvaluator

with read_base():
    from .datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets
    from .datasets.mbpp.deprecated_mbpp_gen_1e1056 import mbpp_datasets

humaneval_datasets[0]['abbr'] = 'openai_humaneval_pass10'
humaneval_datasets[0]['num_repeats'] = 10
mbpp_datasets[0]['abbr'] = 'mbpp_pass10'
mbpp_datasets[0]['num_repeats'] = 10
mbpp_datasets[0]['type'] = MBPPDatasetV2
mbpp_datasets[0]['eval_cfg']['evaluator']['type'] = MBPPPassKEvaluator
mbpp_datasets[0]['reader_cfg']['output_column'] = 'test_column'

datasets = []
datasets += humaneval_datasets
datasets += mbpp_datasets

models = [
    dict(
        type=HuggingFaceCausalLM,
        ...,
        generation_kwargs=dict(
            do_sample=True,
            top_p=0.95,
            temperature=0.8,
        ),
        ...,
    )
]
```

Because the dataset prompt does not change, modify these fields to repeat data:

- `num_repeats`: number of dataset repetitions.
- `abbr`: update the dataset abbreviation with the repeat count because dataset size changes; this prevents mismatches with `.cache/dataset_size.json`.

For MBPP, likewise change `type`, `eval_cfg.evaluator.type`, and `reader_cfg.output_column`. Set stochastic `generation_kwargs` on the model as well.

See [examples/eval_code_passk_repeat_dataset.py](https://github.com/open-compass/opencompass/blob/main/examples/eval_code_passk_repeat_dataset.py).

## Code Execution Service

Datasets requiring real code execution are evaluated in an independent Docker service. OpenCompass uses a service built from [code-evaluator](https://github.com/open-compass/code-evaluator).

### Supported Datasets

1. humaneval-x

   The multilingual [humaneval-x](https://huggingface.co/datasets/THUDM/humaneval-x) dataset. Download the required language file (`xx.jsonl.gz`) from its [download location](https://github.com/THUDM/CodeGeeX2/tree/main/benchmark/humanevalx) and place it under `./data/humanevalx`.

   Supported languages are `python`, `cpp`, `go`, `java`, and `js`.

2. DS1000

   The multi-library Python dataset [DS1000](https://github.com/xlang-ai/DS-1000). Download it from [ds1000_data.zip](https://github.com/xlang-ai/DS-1000/blob/main/ds1000_data.zip).

   Supported libraries are `Pandas`, `Numpy`, `Tensorflow`, `Scipy`, `Sklearn`, `Pytorch`, and `Matplotlib`.

### Launching the Code Evaluation Service

1. Ensure you have installed Docker, please refer to [Docker installation document](https://docs.docker.com/engine/install/).
2. Pull the source code of the code evaluation service project and build the Docker image.

Choose the dockerfile corresponding to the dataset you need, and replace `humanevalx` or `ds1000` in the command below.

```shell
git clone https://github.com/open-compass/code-evaluator.git
docker build -t code-eval-{your-dataset}:latest -f docker/{your-dataset}/Dockerfile .
```

3. Create a container with the following commands:

```shell
# Log output format
docker run -it -p 5000:5000 code-eval-{your-dataset}:latest python server.py

# Run the program in the background
# docker run -itd -p 5000:5000 code-eval-{your-dataset}:latest python server.py

# Using different ports
# docker run -itd -p 5001:5001 code-eval-{your-dataset}:latest python server.py --port 5001
```

**Note:**

- If you encounter a timeout during the evaluation of Go, please use the following command when creating the container.

```shell
docker run -it -p 5000:5000 -e GO111MODULE=on -e GOPROXY=https://goproxy.io code-eval-{your-dataset}:latest python server.py
```

4. To ensure you have access to the service, use the following command to check the inference environment and evaluation service connection status. (If both inferences and code evaluations run on the same host, skip this step.)

```shell
ping your_service_ip_address
telnet your_service_ip_address your_service_port
```

## Local Code Evaluation

When the model inference and code evaluation services are running on the same host or within the same local area network, direct code reasoning and evaluation can be performed. **Note: DS1000 is currently not supported, please proceed with remote evaluation.**

### Configuration File

We provide [the configuration file](https://github.com/open-compass/opencompass/blob/main/examples/eval_codegeex2.py) of using `humanevalx` for evaluation on `codegeex2` as reference.

The dataset and related post-processing configuration files are available at this [link](https://github.com/open-compass/opencompass/tree/main/opencompass/configs/datasets/humanevalx). Note the `evaluator` field in `humanevalx_eval_cfg_dict`.

```python
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import HumanevalXDataset, HumanevalXEvaluator

humanevalx_reader_cfg = dict(
    input_columns=['prompt'], output_column='task_id', train_split='test')

humanevalx_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='{prompt}'),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=1024))

humanevalx_eval_cfg_dict = {
    lang : dict(
            evaluator=dict(
                type=HumanevalXEvaluator,
                language=lang,
                ip_address="localhost",    # replace to your code_eval_server ip_address, port
                port=5000),               # refer to https://github.com/open-compass/code-evaluator to launch a server
            pred_role='BOT')
    for lang in ['python', 'cpp', 'go', 'java', 'js']   # do not support rust now
}

humanevalx_datasets = [
    dict(
        type=HumanevalXDataset,
        abbr=f'humanevalx-{lang}',
        language=lang,
        path='./data/humanevalx',
        reader_cfg=humanevalx_reader_cfg,
        infer_cfg=humanevalx_infer_cfg,
        eval_cfg=humanevalx_eval_cfg_dict[lang])
    for lang in ['python', 'cpp', 'go', 'java', 'js']
]
```

### Task Launch

Refer to the [Five-Minute Quick Start](../get_started/quick_start.md).

## Remote Code Evaluation

Model inference and code evaluation services located in different machines which cannot be accessed directly require prior model inference before collecting the code evaluation results. The configuration file and inference process can be reused from the previous tutorial.

### Collect Inference Results(Only for Humanevalx)

OpenCompass provides `tools/collect_code_preds.py` to post-process and collect inference results. Supply the configuration used to launch the task and the working directory to reuse. Its reuse argument follows the same semantics as `opencompass --reuse`; see [Task Recovery, Artifact Reuse, and Evaluation-only Runs](../execution/tasks_and_runners.md#task-recovery-artifact-reuse-and-evaluation-only-runs).
It is the same with `-r` option in `run.py`. More details can be referred through the [documentation](https://opencompass.readthedocs.io/en/latest/get_started/quick_start.html#launching-evaluation).

```shell
python tools/collect_code_preds.py [config] [-r latest]
```

The collected results will be organized as following under the `-r` folder:

```
workdir/humanevalx
├── codegeex2-6b
│   ├── humanevalx_cpp.json
│   ├── humanevalx_go.json
│   ├── humanevalx_java.json
│   ├── humanevalx_js.json
│   └── humanevalx_python.json
├── CodeLlama-13b
│   ├── ...
├── CodeLlama-13b-Instruct
│   ├── ...
├── CodeLlama-13b-Python
│   ├── ...
├── ...
```

For DS1000, you just need to obtain the corresponding prediction file generated by `opencompass`.

### Code Evaluation

Make sure your code evaluation service is started, and use `curl` to request:

#### The following only supports Humanevalx

```shell
curl -X POST -F 'file=@{result_absolute_path}' -F 'dataset={dataset/language}' {your_service_ip_address}:{your_service_port}/evaluate
```

For example:

```shell
curl -X POST -F 'file=@./examples/humanevalx/python.json' -F 'dataset=humanevalx/python' localhost:5000/evaluate
```

The we have:

```
"{\"pass@1\": 37.19512195121951%}"
```

Additionally, we offer an extra option named `with_prompt`(Defaults to `True`), since some models(like `WizardCoder`) generate complete codes without requiring the form of concatenating prompt and prediction. You may refer to the following commands for evaluation.

```shell
curl -X POST -F 'file=@./examples/humanevalx/python.json' -F 'dataset=humanevalx/python' -H 'with-prompt: False' localhost:5000/evaluate
```

#### The following only supports DS1000

Make sure the code evaluation service is started, then use `curl` to submit a request:

```shell
curl -X POST -F 'file=@./internlm-chat-7b-hf-v11/ds1000_Numpy.json' localhost:5000/evaluate
```

DS1000 supports additional debug parameters. Be aware that a large amount of log will be generated when it is turned on:

- `full`: Additional print out of the original prediction for each error sample, post-processing prediction, running program, and final error.
- `half`: Additional print out of the running program and final error for each error sample.
- `error`: Additional print out of the final error for each error sample.

```shell
curl -X POST -F 'file=@./internlm-chat-7b-hf-v11/ds1000_Numpy.json' -F 'debug=error' localhost:5000/evaluate
```

You can also modify the `num_workers` in the same way to control the degree of parallelism.

## Advanced Tutorial

Besides evaluating the supported HUMANEVAList data set, users might also need:

### Support New Dataset

See [Adding a Dataset](../extension/new_dataset.md).

### Modify Post-Processing

1. For local evaluation, follow the post-processing section in the tutorial on supporting new datasets to modify the post-processing method.
2. For remote evaluation, please modify the post-processing part in the tool's `collect_code_preds.py`.
3. Some parts of post-processing could also be modified in the code evaluation service, more information will be available in the next section.

### Debugging Code Evaluation Service

When supporting new datasets or modifying post-processors, it is possible that modifications need to be made to the original code evaluation service. Please make changes based on the following steps:

1. Remove the installation of the `code-evaluator` in `Dockerfile`, mount the `code-evaluator` when starting the container instead:

```shell
docker run -it -p 5000:5000 -v /local/path/of/code-evaluator:/workspace/code-evaluator code-eval:latest bash
```

2. Install and start the code evaluation service locally. At this point, any necessary modifications can be made to the local copy of the `code-evaluator`.

```shell
cd code-evaluator && pip install -r requirements.txt
python server.py
```
