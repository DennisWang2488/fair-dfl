"""Fair multi-objective decision-focused learning (fair_dfl)."""

__all__ = ["run_experiment_unified"]


def __getattr__(name: str):
    if name == "run_experiment_unified":
        from .runner import run_experiment_unified

        return run_experiment_unified
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
