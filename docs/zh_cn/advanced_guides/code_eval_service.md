# 代码评测服务

我们支持评测多编程语言的数据集，类似 [humaneval-x](https://huggingface.co/datasets/THUDM/humaneval-x). 在启动之前需要确保你已经启动了代码评测服务，代码评测服务可参考[code-evaluator](https://github.com/Ezra-Yu/code-evaluator)项目。

## 启动代码评测服务

确保您已经安装了 docker，然后构建一个镜像并运行一个容器服务。

构建 Docker 镜像：

```shell
git clone https://github.com/Ezra-Yu/code-evaluator.git
cd code-evaluator/docker
sudo docker build -t code-eval:latest .
```

获取镜像后，使用以下命令创建容器：

```shell
# 输出日志格式
sudo docker run -it -p 5000:5000 code-eval:latest python server.py

# 在后台运行程序
# sudo docker run -itd -p 5000:5000 code-eval:latest python server.py

# 使用不同的端口
# sudo docker run -itd -p 5001:5001 code-eval:latest python server.py --port 5001
```

确保您能够访问服务，检查以下命令(如果在本地主机中运行服务，就跳过这个操作)：

```shell
ping your_service_ip_address
telnet your_service_ip_address your_service_port
```

```note
如果运算节点不能连接到评估服务，也可直接运行 `python run.py xxx...`，代码生成结果会保存在 'outputs' 文件夹下，迁移后直接使用 [code-evaluator](https://github.com/Ezra-Yu/code-evaluator) 评测得到结果（不需要考虑后面 eval_cfg 的配置）。
```

## 配置文件

我么已经给了 huamaneval-x 在 codegeex2 上评估的[配置文件](https://github.com/InternLM/opencompass/blob/main/configs/eval_codegeex2.py)。

其中数据集以及相关后处理的配置文件为这个[链接](https://github.com/InternLM/opencompass/tree/main/configs/datasets/humanevalx)， 需要注意 `humanevalx_eval_cfg_dict` 中的
`evaluator` 字段。

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
