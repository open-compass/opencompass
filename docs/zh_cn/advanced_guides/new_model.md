# 支持新模型

目前我们已经支持的模型有 HF 模型、部分模型 API 、部分第三方模型。

## 新增API模型

新增基于API的模型，需要在 `opencompass/models` 下新建 `mymodel_api.py` 文件，继承 `BaseAPIModel`，并实现 `generate` 方法来进行推理，以及 `get_token_len` 方法来计算 token 的长度。在定义好之后修改对应配置文件名称即可。

```python
from ..base_api import BaseAPIModel

class MyModelAPI(BaseAPIModel):

    is_api: bool = True

    def __init__(self,
                 path: str,
                 max_seq_len: int = 2048,
                 query_per_second: int = 1,
                 retry: int = 2,
                 **kwargs):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         meta_template=meta_template,
                         query_per_second=query_per_second,
                         retry=retry)
        ...

    def generate(
        self,
        inputs,
        max_out_len: int = 512,
        temperature: float = 0.7,
    ) -> List[str]:
        """Generate results given a list of inputs."""
        pass

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized string."""
        pass
```

## 新增第三方模型

新增基于第三方的模型，需要在 `opencompass/models` 下新建 `mymodel.py` 文件，继承 `BaseModel`，并实现  `generate` 方法来进行生成式推理， `get_ppl` 方法来进行判别式推理，以及 `get_token_len` 方法来计算 token 的长度。在定义好之后修改对应配置文件名称即可。

```python
from ..base import BaseModel

class MyModel(BaseModel):

    def __init__(self,
                 pkg_root: str,
                 ckpt_path: str,
                 tokenizer_only: bool = False,
                 meta_template: Optional[Dict] = None,
                 **kwargs):
        ...

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized strings."""
        pass

    def generate(self, inputs: List[str], max_out_len: int) -> List[str]:
        """Generate results given a list of inputs. """
        pass

    def get_ppl(self,
                inputs: List[str],
                mask_length: Optional[List[int]] = None) -> List[float]:
        """Get perplexity scores given a list of inputs."""
        pass
```
