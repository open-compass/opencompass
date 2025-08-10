# Add a Model

Currently, we support HF models, some model APIs, and some third-party models.

## Adding API Models

To add a new API-based model, you need to create a new file named `mymodel_api.py` under `opencompass/models` directory. In this file, you should inherit from `BaseAPIModel` and implement the `generate` method for inference and the `get_token_len` method to calculate the length of tokens. Once you have defined the model, you can modify the corresponding configuration file.

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

## Adding Third-Party Models

To add a new third-party model, you need to create a new file named `mymodel.py` under `opencompass/models` directory. In this file, you should inherit from `BaseModel` and implement the `generate` method for generative inference, the `get_ppl` method for discriminative inference, and the `get_token_len` method to calculate the length of tokens. Once you have defined the model, you can modify the corresponding configuration file.

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
