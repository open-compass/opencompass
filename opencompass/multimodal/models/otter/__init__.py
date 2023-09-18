from typing import TYPE_CHECKING

from transformers.utils import (OptionalDependencyNotAvailable,
                                is_torch_available)

if TYPE_CHECKING:
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass

from .otter import Otter
from .post_processor import OTTERMMBenchPostProcessor
from .prompt_constructor import OTTERMMBenchPromptConstructor

__all__ = [
    'Otter', 'OTTERMMBenchPromptConstructor', 'OTTERMMBenchPostProcessor'
]
