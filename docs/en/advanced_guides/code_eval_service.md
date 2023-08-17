# Code Evaluation Service

We support evaluating datasets of multiple programming languages, similar to [humaneval-x](https://huggingface.co/datasets/THUDM/humaneval-x). Before starting, make sure that you have started the code evaluation service. You can refer to the [code-evaluator](https://github.com/Ezra-Yu/code-evaluator) project for the code evaluation service.

## Launching the Code Evaluation Service

Make sure you have installed Docker, then build an image and run a container service.

Build the Docker image:

```shell
git clone https://github.com/Ezra-Yu/code-evaluator.git
cd code-evaluator/docker
sudo docker build -t code-eval:latest .
```

After obtaining the image, create a container with the following commands:

```shell
# Log output format
sudo docker run -it -p 5000:5000 code-eval:latest python server.py

# Run the program in the background
# sudo docker run -itd -p 5000:5000 code-eval:latest python server.py

# Using different ports
# sudo docker run -itd -p 5001:5001 code-eval:latest python server.py --port 5001
```

Ensure that you can access the service and check the following commands (skip this step if you are running the service on a local host):

```shell
ping your_service_ip_address
telnet your_service_ip_address your_service_port
```

```note
If computing nodes cannot connect to the evaluation service, you can directly run `python run.py xxx...`. The resulting code will be saved in the 'outputs' folder. After migration, use [code-evaluator](https://github.com/Ezra-Yu/code-evaluator) directly to get the results (no need to consider the eval_cfg configuration later).
```

## Configuration File

We have provided the [configuration file](https://github.com/InternLM/opencompass/blob/main/configs/eval_codegeex2.py) for evaluating huamaneval-x on codegeex2 .

The dataset and related post-processing configuration files can be found at this [link](https://github.com/InternLM/opencompass/tree/main/configs/datasets/humanevalx). Note the `evaluator` field in `humanevalx_eval_cfg_dict`.

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
                port=5000),               # refer to https://github.com/Ezra-Yu/code-evaluator to launch a server
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
