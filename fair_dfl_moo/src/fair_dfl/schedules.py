"""Learning-rate and alpha schedule helpers used across active methods."""

import math
from typing import Dict


def alpha_value(t: int, schedule_cfg: Dict[str, float]) -> float:
    sch_type = schedule_cfg.get("type", "sigmoid_decay")
    if sch_type == "constant":
        return float(schedule_cfg.get("value", 0.0))
    if sch_type == "sigmoid_decay":
        alpha_max = float(schedule_cfg.get("alpha_max", 1.0))
        alpha_min = float(schedule_cfg.get("alpha_min", 0.0))
        midpoint = float(schedule_cfg.get("midpoint", 100.0))
        temperature = float(schedule_cfg.get("temperature", 20.0))
        scaled = (float(t) - midpoint) / max(temperature, 1e-8)
        return alpha_min + (alpha_max - alpha_min) / (1.0 + math.exp(scaled))
    if sch_type == "paper_decay":
        # PLG guidance-weight schedule, Jeon et al. 2025 (arXiv:2509.08359), eq. (4):
        #   alpha_t = (1 + exp((t - c)/temperature))^(-kappa)
        # The paper compares two variants, kappa in {0, 1}:
        #   kappa = 0 => alpha == 1 for all t (the update always bisects the
        #               prediction- and decision-loss gradients);
        #   kappa = 1 => alpha decays sigmoidally from ~1 to ~0, so the update leans
        #               toward the decision gradient late in training.
        # c (inflection epoch) and temperature (steepness) are FIXED, not tuned; kappa
        # is the only tunable knob. (Matches experiments/hp_testing/PARAMETER_GUIDE.md.)
        kappa = float(schedule_cfg.get("kappa", 1.0))
        c = float(schedule_cfg.get("c", 100.0))
        temperature = float(schedule_cfg.get("temperature", 20.0))
        z = (float(t) - c) / max(temperature, 1e-8)
        z = min(z, 60.0)   # guard math.exp overflow; alpha is already ~0 well before this
        return float((1.0 + math.exp(z)) ** (-kappa))
    if sch_type == "poly_decay":
        alpha_max = float(schedule_cfg.get("alpha_max", 1.0))
        alpha_min = float(schedule_cfg.get("alpha_min", 0.0))
        power = float(schedule_cfg.get("power", 1.0))
        horizon = float(schedule_cfg.get("horizon", 200.0))
        ratio = max(0.0, 1.0 - min(float(t) / max(horizon, 1.0), 1.0))
        return alpha_min + (alpha_max - alpha_min) * (ratio**power)
    if sch_type == "inv_sqrt":
        alpha0 = float(schedule_cfg.get("alpha0", 1.0))
        alpha_min = float(schedule_cfg.get("alpha_min", 0.0))
        return max(alpha_min, alpha0 / math.sqrt(float(t) + 1.0))
    raise ValueError(f"Unsupported alpha schedule type: {sch_type}")


def lr_value(t: int, lr: float, lr_decay: float) -> float:
    return lr / (1.0 + lr_decay * float(t))
