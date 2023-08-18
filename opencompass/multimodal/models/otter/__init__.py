from typing import TYPE_CHECKING

from transformers.utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

from .otter import Otter

__all__ = ["Otter"]

_import_structure = {
    "configuration_otter": [
        "OtterConfig",
    ],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_otter"] = [
        "OtterModel",
        "OtterPreTrainedModel",
        "OtterForConditionalGeneration",
    ]

if TYPE_CHECKING:
    from .configuration_otter import OtterConfig

    # from .processing_otter import OtterProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_otter import OtterForConditionalGeneration, OtterModel, OtterPreTrainedModel

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
