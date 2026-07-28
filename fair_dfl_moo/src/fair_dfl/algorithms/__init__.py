"""Algorithm components used by training/loop.py.

  - mo_handler.py        — MOO gradient handlers (PCGrad, MGDA, NashMTL, ...)
  - torch_utils.py       — gradient manipulation utilities
  - finite_diff_grad.py  — finite-difference decision-gradient helper

The unified training pipeline in training/loop.py is the single trainer; the former
core_methods.py legacy trainer has been removed (see docs/REFACTOR_PLAN.md).
"""
