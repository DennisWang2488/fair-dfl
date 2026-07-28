"""Unified training package for all DFL methods."""

from importlib import import_module

__all__ = [
    "eval_split",
    "eval_split_medical",
    "evaluate_model",
    "run_method_seed",
    "run_methods",
    "train_single_stage",
    "MethodSpec",
    "resolve_method_spec",
    "resolve_method_backend",
]

_EXPORT_MAP = {
    "eval_split": ".eval",
    "eval_split_medical": ".eval",
    "evaluate_model": ".eval",
    "run_method_seed": ".loop",
    "run_methods": ".loop",
    "train_single_stage": ".loop",
    "MethodSpec": ".method_spec",
    "resolve_method_spec": ".method_spec",
    "resolve_method_backend": ".method_spec",
}


def __getattr__(name: str):
    module_name = _EXPORT_MAP.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    return getattr(module, name)


def __dir__():
    return sorted(list(globals().keys()) + __all__)
