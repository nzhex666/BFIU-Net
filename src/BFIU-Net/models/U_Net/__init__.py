import hashlib as _hashlib
import importlib.util as _util
from pathlib import Path as _Path
import sys as _sys

_directory = _Path(__file__).resolve().parent
_key = "_c0_plain_unet3d_entry_" + _hashlib.sha256(str(_directory).encode("utf-8")).hexdigest()[:20]
if _key not in _sys.modules:
    _spec = _util.spec_from_file_location(_key, str(_directory / "models.py"))
    _module = _util.module_from_spec(_spec)
    _sys.modules[_key] = _module
    try:
        _spec.loader.exec_module(_module)
    except BaseException:
        _sys.modules.pop(_key, None)
        raise
else:
    _module = _sys.modules[_key]

__all__ = list(_module.__all__)
for _name in __all__:
    globals()[_name] = getattr(_module, _name)
