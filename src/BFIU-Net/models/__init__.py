from __future__ import annotations

from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any


_MODEL_IDS = (
    "BFIU_Net",
    "NIU-Net",
    "LHU-Net",
    "HLU-Net",
    "U_Net",
    "W_NET",
    "MS_UNET",
)
_MODEL_ALIASES = {name.replace("-", "_").upper(): name for name in _MODEL_IDS}
_MODEL_ALIASES.update({
    "R4_NO_EXCHANGE": "NIU-Net",
    "R5_LOW_TO_HIGH": "LHU-Net",
    "R6_HIGH_TO_LOW": "HLU-Net",
    "BASIC_UNET": "U_Net",
})
_COMPARISON_ID_ATTRIBUTES = {
    "W_NET": "wnet_model_id",
    "MS_UNET": "ms_unet_model_id",
}


def list_models() -> tuple[str, ...]:

    return _MODEL_IDS


def normalize_model_name(name: str) -> str:

    if isinstance(name, str):
        canonical = name.strip().replace("-", "_").upper()
        if canonical in _MODEL_ALIASES:
            return _MODEL_ALIASES[canonical]
    raise ValueError("Unknown model {!r}; choose one of {}".format(name, ", ".join(_MODEL_IDS)))


def get_model_module(name: str) -> ModuleType:

    name = normalize_model_name(name)
    return import_module("{}.{}".format(__name__, name))


def build_model(name: str, input_shape=(128, 128, 128, 1)) -> Any:

    return get_model_module(name).build_unet3d(input_shape=input_shape)


def _checked_module(name: str, model: Any) -> ModuleType:
    name = normalize_model_name(name)
    module = get_model_module(name)
    if getattr(model, "name", None) != module.MODEL_ID:
        raise ValueError(
            "Registry model ID mismatch: {!r} requires {!r}, got {!r}".format(
                name, module.MODEL_ID, getattr(model, "name", None)
            )
        )
    return module


def save_weights_checked(name: str, model: Any, path) -> Path:

    module = _checked_module(name, model)
    destination = Path(path).expanduser()
    module.save_weights_checked(model, destination)
    return destination


def load_weights_checked(name: str, model: Any, path) -> Any:

    name = normalize_model_name(name)
    module = _checked_module(name, model)
    source = Path(path).expanduser()
    attribute = _COMPARISON_ID_ATTRIBUTES.get(name)
    if attribute is not None:
        import h5py

        with h5py.File(str(source), "r") as handle:
            stored_id = handle.attrs.get(attribute)
        if isinstance(stored_id, bytes):
            stored_id = stored_id.decode("utf-8")
        if stored_id != module.MODEL_ID:
            raise ValueError("Checkpoint model ID mismatch for {!r}".format(name))
    module.load_weights_checked(model, source)
    return model


__all__ = [
    "list_models", "normalize_model_name", "get_model_module", "build_model",
    "save_weights_checked", "load_weights_checked",
]
