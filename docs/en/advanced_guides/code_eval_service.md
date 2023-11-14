# Code Evaluation Tutorial

To complete LLM code capability evaluation, we need to set up an independent evaluation environment to avoid executing erroneous codes on development environments which would cause unavoidable losses. The current Code Evaluation Service used in OpenCompass refers to the project [code-evaluator](https://github.com/open-compass/code-evaluator.git), which has already supported evaluating datasets for multiple programming languages [humaneval-x](https://huggingface.co/datasets/THUDM/humaneval-x). The following tutorials will introduce how to conduct code review services under different requirements.

Dataset [download address](https://github.com/THUDM/CodeGeeX2/tree/main/benchmark/humanevalx). Please download the needed files (xx.jsonl.gz) into `./data/humanevalx` folder.

Supported languages are `python`, `cpp`, `go`, `java`, `js`.

## Launching the Code Evaluation Service

1. Ensure you have installed Docker, please refer to [Docker installation document](https://docs.docker.com/engine/install/).
2. Pull the source code of the code evaluation service project and build the Docker image.

```shell
git clone https://github.com/open-compass/code-evaluator.git
cd code-evaluator/docker
sudo docker build -t code-eval:latest .
```

3. Create a container with the following commands:

```shell
# Log output format
sudo docker run -it -p 5000:5000 code-eval:latest python server.py

# Run the program in the background
# sudo docker run -itd -p 5000:5000 code-eval:latest python server.py

# Using different ports
# sudo docker run -itd -p 5001:5001 code-eval:latest python server.py --port 5001
```

4. To ensure you have access to the service, use the following command to check the inference environment and evaluation service connection status. (If both inferences and code evaluations run on the same host, skip this step.)

```shell
ping your_service_ip_address
telnet your_service_ip_address your_service_port
```

## Local Code Evaluation

When the model inference and code evaluation services are running on the same host or within the same local area network, direct code reasoning and evaluation can be performed.

### Configuration File

We provide [the configuration file](https://github.com/open-compass/opencompass/blob/main/configs/eval_codegeex2.py) of using `humanevalx` for evaluation on `codegeex2` as reference.

The dataset and related post-processing configurations files can be found at this [link](https://github.com/open-compass/opencompass/tree/main/configs/datasets/humanevalx) with attention paid to the `evaluator` field in the humanevalx_eval_cfg_dict.

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

Refer to the [Quick Start](../get_started.html)

## Remote Code Evaluation

Model inference and code evaluation services located in different machines which cannot be accessed directly require prior model inference before collecting the code evaluation results. The configuration file and inference process can be reused from the previous tutorial.

### Collect Inference Results

In OpenCompass's tools folder, there is a script called `collect_code_preds.py` provided to process and collect the inference results after providing the task launch configuration file during startup along with specifying the working directory used corresponding to the task.
It is the same with `-r` option in `run.py`. More details can be referred through the [documentation](https://opencompass.readthedocs.io/en/latest/get_started.html#launch-evaluation).

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

### Code Evaluation

Make sure your code evaluation service is started, and use `curl` to request:

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

## Advanced Tutorial

Besides evaluating the supported HUMANEVAList data set, users might also need:

### Support New Dataset

Please refer to the [tutorial on supporting new datasets](./new_dataset.md).

### Modify Post-Processing

1. For local evaluation, follow the post-processing section in the tutorial on supporting new datasets to modify the post-processing method.
2. For remote evaluation, please modify the post-processing part in the tool's `collect_code_preds.py`.
3. Some parts of post-processing could also be modified in the code evaluation service, more information will be available in the next section.

### Debugging Code Evaluation Service

When supporting new datasets or modifying post-processors, it is possible that modifications need to be made to the original code evaluation service. Please make changes based on the following steps:

1. Remove the installation of the `code-evaluator` in `Dockerfile`, mount the `code-evaluator` when starting the container instead:

```shell
sudo docker run -it -p 5000:5000 -v /local/path/of/code-evaluator:/workspace/code-evaluator code-eval:latest bash
```

2. Install and start the code evaluation service locally. At this point, any necessary modifications can be made to the local copy of the `code-evaluator`.

```shell
cd code-evaluator && pip install -r requirements.txt
python server.py
```
