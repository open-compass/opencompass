# Add a Model

Currently, we support HF models, some model APIs, and some third-party models.

## Adding API Models

To add a new API-based model, create `mymodel_api.py` under `opencompass/models`. In this file, inherit from `BaseAPIModel` and implement the `generate` method for inference and the `get_token_len` method to calculate token length.

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

After defining the model, make sure the config can access the class. There are two common options:

- Import the class from its concrete module in the config, for example `from opencompass.models.mymodel_api import MyModelAPI`, then use `type=MyModelAPI`;
- If you want to write `from opencompass.models import MyModelAPI`, export the class in `opencompass/models/__init__.py`. If you want to use a string such as `type='MyModelAPI'`, register the class with `@MODELS.register_module()` as shown above.

## Adding Third-Party Models

To add a new third-party model, create `mymodel.py` under `opencompass/models`. In this file, inherit from `BaseModel` and implement the `generate` method for generative inference, the `get_ppl` method for discriminative inference, and the `get_token_len` method to calculate token length.

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

Likewise, after defining the model, either import the model class directly in the config, or export it from `opencompass/models/__init__.py` and register it with `MODELS` before using a string `type`.
