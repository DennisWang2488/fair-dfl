"""Task layer for data-driven optimization experiments."""

from importlib import import_module

__all__ = [
    "MultiDimKnapsackTask",
    "PortfolioQPSimplexTask",
    "PortfolioQPMultiConstraintTask",
    "MedicalResourceAllocationTask",
    "SingleResourceAlphaFairTask",
]

_EXPORT_MAP = {
    "MultiDimKnapsackTask": ".md_knapsack",
    "PortfolioQPSimplexTask": ".portfolio_qp_simplex",
    "PortfolioQPMultiConstraintTask": ".portfolio_qp_multi_constraint",
    "MedicalResourceAllocationTask": ".medical_resource_allocation",
    "SingleResourceAlphaFairTask": ".single_resource_alpha_fair",
}
_OPTIONAL_EXPORTS: set = set()


def __getattr__(name: str):
    module_name = _EXPORT_MAP.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = import_module(module_name, __name__)
    except ImportError:
        if name in _OPTIONAL_EXPORTS:
            return None
        raise
    return getattr(module, name)


def __dir__():
    return sorted(list(globals().keys()) + __all__)
