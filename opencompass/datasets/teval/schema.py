from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass
class ResponseDataSample:
    """
    Args:
        template(str): Format string with keyword-only arguments. For
            example '{who} like {what}'
        pred(Any): Parsed data from LLM generating response.
        gt(Any): Ground truth data
        meta_data(dict, optional): Meta information will be used to evaluate
             LLM's response
    """
    template: str
    pred: Any
    gt: Any
    meta_data: dict = None
