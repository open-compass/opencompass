# 支持新模型

目前我们已经支持的模型有 HF 模型、部分模型 API 、部分第三方模型。

## 新增API模型

新增基于 API 的模型，需要在 `opencompass/models` 下新建 `mymodel_api.py` 文件，继承 `BaseAPIModel`，并实现 `generate` 方法来进行推理，以及 `get_token_len` 方法来计算 token 的长度。

```python
from typing import Dict, List, Optional

from opencompass.registry import MODELS

from .base_api import BaseAPIModel


@MODELS.register_module()
class MyModelAPI(BaseAPIModel):

    is_api: bool = True

    def __init__(self,
                 path: str,
                 max_seq_len: int = 2048,
                 query_per_second: int = 1,
                 meta_template: Optional[Dict] = None,
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

定义模型后，需要让配置文件能访问到这个类。常见做法有两种：

- 在配置文件中直接从具体模块导入，例如 `from opencompass.models.mymodel_api import MyModelAPI`，然后写 `type=MyModelAPI`；
- 若希望写 `from opencompass.models import MyModelAPI`，需要在 `opencompass/models/__init__.py` 中导出该类；若希望在配置中使用字符串形式 `type='MyModelAPI'`，则需要像上例一样通过 `@MODELS.register_module()` 注册。

## 新增第三方模型

新增基于第三方的模型，需要在 `opencompass/models` 下新建 `mymodel.py` 文件，继承 `BaseModel`，并实现 `generate` 方法来进行生成式推理，`get_ppl` 方法来进行判别式推理，以及 `get_token_len` 方法来计算 token 的长度。

```python
from typing import Dict, List, Optional

from opencompass.registry import MODELS

from .base import BaseModel


@MODELS.register_module()
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

同样地，定义模型后需要在配置文件中导入模型类，或将类导出到 `opencompass/models/__init__.py` 并通过 `MODELS` 注册后再使用字符串形式的 `type`。
