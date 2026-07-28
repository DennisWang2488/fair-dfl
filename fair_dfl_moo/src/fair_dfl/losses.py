"""Prediction/fairness loss utilities shared by task implementations."""

import numpy as np


def softplus_with_grad(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positive = np.maximum(z, 0.0)
    exp_term = np.exp(-np.abs(z))
    value = positive + np.log1p(exp_term)
    sigmoid = np.empty_like(value)
    pos_mask = z >= 0.0
    sigmoid[pos_mask] = 1.0 / (1.0 + np.exp(-z[pos_mask]))
    exp_z = np.exp(z[~pos_mask])
    sigmoid[~pos_mask] = exp_z / (1.0 + exp_z)
    return value, sigmoid


def mse_loss_and_grad(pred: np.ndarray, true: np.ndarray) -> tuple[float, np.ndarray]:
    diff = pred - true
    loss = float(np.mean(diff * diff))
    grad = (2.0 / pred.size) * diff
    return loss, grad


def group_mse_mad_loss_and_grad(
    pred: np.ndarray,
    true: np.ndarray,
    groups: np.ndarray,
    smoothing: float = 1e-6,
) -> tuple[float, np.ndarray]:
    unique_groups = np.unique(groups)
    errors = (pred - true) ** 2
    group_mse = []
    for g in unique_groups:
        mask = groups == g
        group_mse.append(errors[:, mask].mean())
    group_mse_arr = np.asarray(group_mse, dtype=float)
    mean_mse = float(group_mse_arr.mean())
    gap = group_mse_arr - mean_mse
    smooth_abs = np.sqrt(gap * gap + smoothing)
    loss = float(smooth_abs.mean())

    # d(loss)/d(mse_g) = (1/G) * (phi'(gap_g) - mean_h phi'(gap_h))
    dphi = gap / smooth_abs
    dloss_dmse = (dphi - dphi.mean()) / max(len(unique_groups), 1)

    grad = np.zeros_like(pred)
    for idx, g in enumerate(unique_groups):
        mask = groups == g
        denom = pred.shape[0] * int(mask.sum())
        if denom == 0:
            continue
        grad[:, mask] = dloss_dmse[idx] * (2.0 * (pred[:, mask] - true[:, mask]) / float(denom))
    return loss, grad


def group_mse_generalized_entropy_loss_and_grad(
    pred: np.ndarray,
    true: np.ndarray,
    groups: np.ndarray,
    alpha: float = 2.0,
    eps: float = 1e-8,
) -> tuple[float, np.ndarray]:
    unique_groups = np.unique(groups)
    errors = (pred - true) ** 2
    group_mse = []
    for g in unique_groups:
        mask = groups == g
        group_mse.append(errors[:, mask].mean())
    group_mse_arr = np.asarray(group_mse, dtype=float)

    mu = float(np.mean(group_mse_arr))
    mu = max(mu, eps)
    ratio = np.clip(group_mse_arr / mu, eps, None)
    n_groups = max(len(unique_groups), 1)

    if abs(alpha - 1.0) < 1e-12:
        loss = float(np.mean(ratio * np.log(ratio)))
        a = np.log(ratio) + 1.0
        da = float(np.mean(ratio * a))
        dloss_dmse = (a - da) / (n_groups * mu)
    elif abs(alpha) < 1e-12:
        loss = float(-np.mean(np.log(ratio)))
        dloss_dmse = (1.0 - 1.0 / ratio) / (n_groups * mu)
    else:
        moment = float(np.mean(ratio**alpha))
        loss = float((moment - 1.0) / (alpha * (alpha - 1.0)))
        dloss_dmse = (ratio ** (alpha - 1.0) - moment) / ((alpha - 1.0) * n_groups * mu)

    grad = np.zeros_like(pred)
    for idx, g in enumerate(unique_groups):
        mask = groups == g
        denom = pred.shape[0] * int(mask.sum())
        if denom == 0:
            continue
        grad[:, mask] = dloss_dmse[idx] * (2.0 * (pred[:, mask] - true[:, mask]) / float(denom))
    return loss, grad


def group_mse_gap_loss_and_grad(
    pred: np.ndarray,
    true: np.ndarray,
    groups: np.ndarray,
    smoothing: float = 1e-6,
) -> tuple[float, np.ndarray]:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return 0.0, np.zeros_like(pred)

    errors = (pred - true) ** 2
    if len(unique_groups) == 2:
        g0, g1 = unique_groups[0], unique_groups[1]
        m0, m1 = groups == g0, groups == g1
        n0, n1 = int(m0.sum()), int(m1.sum())
        if n0 == 0 or n1 == 0:
            return 0.0, np.zeros_like(pred)
        mse0 = float(errors[:, m0].mean())
        mse1 = float(errors[:, m1].mean())
        gap = mse0 - mse1
        loss = float(np.sqrt(gap * gap + smoothing))
        coeff = gap / max(loss, 1e-12)
        grad = np.zeros_like(pred)
        grad[:, m0] = coeff * 2.0 * (pred[:, m0] - true[:, m0]) / float(pred.shape[0] * n0)
        grad[:, m1] = -coeff * 2.0 * (pred[:, m1] - true[:, m1]) / float(pred.shape[0] * n1)
        return loss, grad

    group_mse = []
    for g in unique_groups:
        mask = groups == g
        group_mse.append(errors[:, mask].mean())
    group_mse_arr = np.asarray(group_mse, dtype=float)
    mean_mse = float(group_mse_arr.mean())
    gap = group_mse_arr - mean_mse
    smooth_abs = np.sqrt(gap * gap + smoothing)
    loss = float(smooth_abs.mean())

    dphi = gap / smooth_abs
    dloss_dmse = (dphi - dphi.mean()) / max(len(unique_groups), 1)

    grad = np.zeros_like(pred)
    for idx, g in enumerate(unique_groups):
        mask = groups == g
        denom = pred.shape[0] * int(mask.sum())
        if denom == 0:
            continue
        grad[:, mask] = dloss_dmse[idx] * (2.0 * (pred[:, mask] - true[:, mask]) / float(denom))
    return loss, grad


def group_pred_mean_dp_loss_and_grad(
    pred: np.ndarray,
    true: np.ndarray,  # noqa: ARG001 - kept for API symmetry with other fairness losses
    groups: np.ndarray,
    smoothing: float = 1e-6,
) -> tuple[float, np.ndarray]:
    """Demographic parity on predictions: MAD of per-group mean predictions.

    Loss
    ----
    L = mean_g sqrt( (mu_g - mu_bar)^2 + smoothing )
        where mu_g    = mean(pred | group=g)        (per-group mean prediction)
              mu_bar  = mean_g(mu_g)                (mean of per-group means)

    Unlike ``group_mse_*`` losses (which equalise per-group MSE i.e. accuracy
    parity), this targets demographic parity: it penalises differences in the
    average predicted benefit across groups regardless of label values.
    """
    unique_groups = np.unique(groups)
    K = len(unique_groups)
    if K < 2:
        return 0.0, np.zeros_like(pred)

    group_means = np.zeros(K, dtype=np.float64)
    group_sizes = np.zeros(K, dtype=np.float64)
    for idx, g in enumerate(unique_groups):
        mask = groups == g
        n_g = float(mask.sum())
        group_sizes[idx] = n_g
        group_means[idx] = float(pred[:, mask].mean()) if n_g > 0 else 0.0

    mean_of_means = float(group_means.mean())
    dev = group_means - mean_of_means                          # (K,)
    smooth_abs = np.sqrt(dev * dev + smoothing)                # (K,)
    loss = float(smooth_abs.mean())

    # d(loss)/d(mu_g) via chain rule (identical to MAD form):
    #   d(loss)/d(mu_g) = (1/K) * (dev_g / smooth_abs_g - mean_h(dev_h / smooth_abs_h))
    dphi = dev / smooth_abs                                    # (K,)
    dloss_dmu = (dphi - dphi.mean()) / float(K)                # (K,)

    # d(mu_g)/d(pred[b, i]) = 1 / (B * n_g)  if i in g else 0
    grad = np.zeros_like(pred)
    B = float(pred.shape[0])
    for idx, g in enumerate(unique_groups):
        mask = groups == g
        n_g = group_sizes[idx]
        if n_g == 0:
            continue
        grad[:, mask] = dloss_dmu[idx] / (B * n_g)
    return loss, grad


def group_residual_mean_bias_parity_loss_and_grad(
    pred: np.ndarray,
    true: np.ndarray,
    groups: np.ndarray,
    smoothing: float = 1e-6,
) -> tuple[float, np.ndarray]:
    """Bias parity (calibration first moment): MAD of per-group mean residuals.

    Loss
    ----
    L = mean_g sqrt( (b_g - b_bar)^2 + smoothing )
        where b_g    = mean(pred - true | group=g)    (per-group mean signed residual)
              b_bar  = mean_g(b_g)                    (mean of per-group biases)

    Sits in the **calibration / sufficiency** family of fairness measures:

    * Different from ``group_mse_*`` (separation / equalised errors): MSE-based
      losses equalise the *magnitude* of error per group, but a group can have
      low MSE while still being systematically over- or under-predicted (e.g.
      MSE = variance + bias^2, and the variance can dominate).
    * Different from ``group_pred_mean_dp`` (independence / demographic parity):
      DP equalises raw prediction means without reference to labels, so it
      penalises differences in the predicted distribution even when those
      differences correctly reflect different ground truths. Bias parity only
      penalises the *signed* prediction error per group, so a group with a
      genuinely higher mean label is fine as long as the predictor isn't
      systematically biased on it.

    Both forms aggregate per-group statistics with the same MAD aggregator as
    ``mad`` and ``dp``, so the only thing that changes is the per-group
    statistic.
    """
    unique_groups = np.unique(groups)
    K = len(unique_groups)
    if K < 2:
        return 0.0, np.zeros_like(pred)

    residual = pred - true                                     # (B, N)
    group_bias = np.zeros(K, dtype=np.float64)
    group_sizes = np.zeros(K, dtype=np.float64)
    for idx, g in enumerate(unique_groups):
        mask = groups == g
        n_g = float(mask.sum())
        group_sizes[idx] = n_g
        group_bias[idx] = float(residual[:, mask].mean()) if n_g > 0 else 0.0

    mean_of_bias = float(group_bias.mean())
    dev = group_bias - mean_of_bias                            # (K,)
    smooth_abs = np.sqrt(dev * dev + smoothing)                # (K,)
    loss = float(smooth_abs.mean())

    # d(loss)/d(b_g) via the same chain as MAD/DP.
    dphi = dev / smooth_abs                                    # (K,)
    dloss_db = (dphi - dphi.mean()) / float(K)                 # (K,)

    # d(b_g)/d(pred[b, i]) = 1 / (B * n_g) for i in g, since residual = pred - true
    # and b_g is linear in pred. Identical denominator to dp.
    grad = np.zeros_like(pred)
    B = float(pred.shape[0])
    for idx, g in enumerate(unique_groups):
        mask = groups == g
        n_g = group_sizes[idx]
        if n_g == 0:
            continue
        grad[:, mask] = dloss_db[idx] / (B * n_g)
    return loss, grad


def group_mse_atkinson_loss_and_grad(
    pred: np.ndarray,
    true: np.ndarray,
    groups: np.ndarray,
    smoothing: float = 1e-6,
    epsilon: float = 0.5,
) -> tuple[float, np.ndarray]:
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    if n_groups < 2:
        return 0.0, np.zeros_like(pred)

    errors = (pred - true) ** 2
    group_mse = np.zeros(n_groups, dtype=np.float64)
    group_sizes = np.zeros(n_groups, dtype=np.float64)
    group_masks = []
    for idx, g in enumerate(unique_groups):
        mask = groups == g
        group_masks.append(mask)
        n_g = float(mask.sum())
        group_sizes[idx] = n_g
        group_mse[idx] = max(float(errors[:, mask].mean()), smoothing) if n_g > 0 else smoothing

    grand_mean = float(np.mean(group_mse))
    grand_mean_safe = max(grand_mean, 1e-12)

    if abs(epsilon - 1.0) < 1e-12:
        log_mse = np.log(group_mse)
        geomean = float(np.exp(np.mean(log_mse)))
        loss = max(1.0 - geomean / grand_mean_safe, 0.0)

        dloss_dmse = np.zeros(n_groups, dtype=np.float64)
        for idx in range(n_groups):
            d_geomean = geomean / (float(n_groups) * group_mse[idx])
            d_grand = 1.0 / float(n_groups)
            dloss_dmse[idx] = -(d_geomean * grand_mean_safe - geomean * d_grand) / (grand_mean_safe ** 2)
    else:
        one_minus_eps = 1.0 - epsilon
        powered = np.power(group_mse, one_minus_eps)
        mean_powered = float(np.mean(powered))
        mean_powered_safe = max(mean_powered, 1e-12)
        ede = mean_powered_safe ** (1.0 / one_minus_eps)
        loss = max(1.0 - ede / grand_mean_safe, 0.0)

        dloss_dmse = np.zeros(n_groups, dtype=np.float64)
        for idx in range(n_groups):
            d_ede = (1.0 / float(n_groups)) * ede / mean_powered_safe * (group_mse[idx] ** (-epsilon))
            d_grand = 1.0 / float(n_groups)
            dloss_dmse[idx] = -(d_ede * grand_mean_safe - ede * d_grand) / (grand_mean_safe ** 2)

    grad = np.zeros_like(pred)
    for idx, mask in enumerate(group_masks):
        denom = pred.shape[0] * group_sizes[idx]
        if denom == 0:
            continue
        grad[:, mask] = dloss_dmse[idx] * 2.0 * (pred[:, mask] - true[:, mask]) / denom
    return loss, grad


def group_mse_gini_loss_and_grad(
    pred: np.ndarray,
    true: np.ndarray,
    groups: np.ndarray,
    smoothing: float = 1e-6,
) -> tuple[float, np.ndarray]:
    """Gini coefficient over per-group MSE (accuracy-parity flavor).

    Loss
    ----
    L = (1/(2 K^2 mu_bar)) * sum_{i,j} sqrt((m_i - m_j)^2 + s)
        where m_g    = MSE(pred, true | group=g)   (per-group MSE)
              mu_bar = mean_g(m_g)                  (mean of per-group MSE)
              s      = smoothing

    Range: 0 (all groups equal MSE) to ~1 (one group dominates). Symmetric
    aggregator with mad/atkinson; differs only in the inequality functional.
    """
    unique_groups = np.unique(groups)
    K = len(unique_groups)
    if K < 2:
        return 0.0, np.zeros_like(pred)

    errors = (pred - true) ** 2
    group_mse = np.zeros(K, dtype=np.float64)
    group_sizes = np.zeros(K, dtype=np.float64)
    group_masks = []
    for idx, g in enumerate(unique_groups):
        mask = groups == g
        group_masks.append(mask)
        n_g = float(mask.sum())
        group_sizes[idx] = n_g
        group_mse[idx] = float(errors[:, mask].mean()) if n_g > 0 else 0.0

    mu_bar = float(np.mean(group_mse))
    mu_safe = max(mu_bar, 1e-12)

    diff = group_mse[:, None] - group_mse[None, :]               # (K, K)
    smooth_abs = np.sqrt(diff * diff + smoothing)                # (K, K)
    N = float(smooth_abs.sum()) / (2.0 * K * K)
    loss = N / mu_safe

    # dN/dm_g = (1/K^2) * sum_j (m_g - m_j) / smooth_abs(m_g, m_j)
    dN_dm = (diff / smooth_abs).sum(axis=1) / float(K * K)       # (K,)
    dD_dm = 1.0 / float(K)                                        # constant
    dloss_dm = (dN_dm - loss * dD_dm) / mu_safe                   # (K,)

    grad = np.zeros_like(pred)
    B = float(pred.shape[0])
    for idx, mask in enumerate(group_masks):
        n_g = group_sizes[idx]
        if n_g == 0:
            continue
        grad[:, mask] = dloss_dm[idx] * 2.0 * (pred[:, mask] - true[:, mask]) / (B * n_g)
    return loss, grad


def _interp_quantile_with_weights(
    sorted_vec: np.ndarray, src_tau: np.ndarray, target_tau: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Linear interpolation of ``sorted_vec`` at quantile probabilities ``target_tau``.

    Returns
    -------
    values    : (M,) interpolated values
    idx_lo    : (M,) lower bracket indices into ``sorted_vec``
    idx_hi    : (M,) upper bracket indices
    w_lo      : (M,) weight on the lower endpoint
    w_hi      : (M,) weight on the upper endpoint

    Such that values[k] = w_lo[k] * sorted_vec[idx_lo[k]] + w_hi[k] * sorted_vec[idx_hi[k]].
    For target probabilities outside [src_tau[0], src_tau[-1]], the value is
    clamped to the first / last entry (constant-extrapolation, derivative 0
    in the clamped region).
    """
    n = len(sorted_vec)
    # Bracketing indices: pos[k] in [1, n-1], so pos-1 in [0, n-2].
    pos = np.searchsorted(src_tau, target_tau)
    pos = np.clip(pos, 1, n - 1)
    idx_lo = pos - 1
    idx_hi = pos
    tau_lo = src_tau[idx_lo]
    tau_hi = src_tau[idx_hi]
    denom = np.maximum(tau_hi - tau_lo, 1e-15)
    w_hi = (target_tau - tau_lo) / denom
    w_lo = 1.0 - w_hi
    # Out-of-range: snap (constant extrapolation).
    below = target_tau < src_tau[0]
    if below.any():
        w_lo = np.where(below, 1.0, w_lo)
        w_hi = np.where(below, 0.0, w_hi)
        idx_lo = np.where(below, 0, idx_lo)
        idx_hi = np.where(below, 0, idx_hi)
    above = target_tau > src_tau[-1]
    if above.any():
        w_lo = np.where(above, 1.0, w_lo)
        w_hi = np.where(above, 0.0, w_hi)
        idx_lo = np.where(above, n - 1, idx_lo)
        idx_hi = np.where(above, n - 1, idx_hi)
    values = w_lo * sorted_vec[idx_lo] + w_hi * sorted_vec[idx_hi]
    return values, idx_lo, idx_hi, w_lo, w_hi


def group_pred_wasserstein2_dp_loss_and_grad(
    pred: np.ndarray,
    true: np.ndarray,  # noqa: ARG001 - kept for API symmetry
    groups: np.ndarray,
    smoothing: float = 1e-6,  # noqa: ARG001 - kept for API symmetry
    n_quantiles: int = 256,
) -> tuple[float, np.ndarray]:
    """Wasserstein-2 demographic parity between the two group-conditional
    predicted-utility distributions.

    Loss
    ----
    L = W_2^2(p_pred | A=0, p_pred | A=1)
      = mean_{k=1..m} (q_0[k] - q_1[k])^2

    where q_g[k] is the empirical predicted-utility quantile of group g at
    rank probability tau_k = (k - 0.5) / m, and m = min(n_quantiles, n_0, n_1).
    Resampling to a common quantile grid handles unequal group sizes.

    This is the (squared) 2-Wasserstein distance with the empirical-quantile
    estimator — see Chzhen, Denis, Hebiri, Oneto, Pontil (NeurIPS 2020), "Fair
    regression with Wasserstein barycenters". Unlike the mean-based DP
    (``group_pred_mean_dp``), this captures distribution shape — two groups
    with the same mean but different variance have non-zero W2 here.

    Implemented for the binary protected-attribute case (K = 2). For K != 2
    the loss is 0 with zero gradient.

    Differentiability
    -----------------
    The argsort permutation is integer-valued and not differentiable, but
    composing ``sort`` with any smooth downstream loss reduces to indexing the
    inputs by the permutation, which is differentiable in the gather sense.
    The quantile-interpolation step is piecewise-linear; gradient is exact
    away from the discrete kinks where the active bracket changes.
    """
    unique_groups = np.unique(groups)
    if len(unique_groups) != 2:
        return 0.0, np.zeros_like(pred)

    g0_mask = groups == unique_groups[0]
    g1_mask = groups == unique_groups[1]
    n0 = int(g0_mask.sum())
    n1 = int(g1_mask.sum())
    if n0 == 0 or n1 == 0:
        return 0.0, np.zeros_like(pred)

    B = pred.shape[0]
    m = int(min(n_quantiles, n0, n1))
    if m < 2:
        return 0.0, np.zeros_like(pred)

    target_tau = (np.arange(m, dtype=np.float64) + 0.5) / float(m)
    src_tau_0 = (np.arange(n0, dtype=np.float64) + 0.5) / float(n0)
    src_tau_1 = (np.arange(n1, dtype=np.float64) + 0.5) / float(n1)

    loss_total = 0.0
    grad = np.zeros_like(pred, dtype=np.float64)

    for b in range(B):
        p0 = pred[b, g0_mask].astype(np.float64, copy=False)
        p1 = pred[b, g1_mask].astype(np.float64, copy=False)
        argsort_0 = np.argsort(p0, kind="stable")
        argsort_1 = np.argsort(p1, kind="stable")
        sp0 = p0[argsort_0]
        sp1 = p1[argsort_1]

        q0, idx0_lo, idx0_hi, w0_lo, w0_hi = _interp_quantile_with_weights(
            sp0, src_tau_0, target_tau
        )
        q1, idx1_lo, idx1_hi, w1_lo, w1_hi = _interp_quantile_with_weights(
            sp1, src_tau_1, target_tau
        )

        diff = q0 - q1
        loss_b = float(np.mean(diff * diff))
        loss_total += loss_b

        # d(loss_b)/d(q0[k]) = 2 * diff[k] / m;   d/d(q1[k]) = -2 * diff[k] / m
        dq0 = 2.0 * diff / float(m)
        dq1 = -dq0

        # Map quantile gradients back to sorted positions then to original positions.
        grad_sp0 = np.zeros(n0, dtype=np.float64)
        np.add.at(grad_sp0, idx0_lo, w0_lo * dq0)
        np.add.at(grad_sp0, idx0_hi, w0_hi * dq0)
        grad_sp1 = np.zeros(n1, dtype=np.float64)
        np.add.at(grad_sp1, idx1_lo, w1_lo * dq1)
        np.add.at(grad_sp1, idx1_hi, w1_hi * dq1)

        grad_p0 = np.zeros(n0, dtype=np.float64)
        grad_p0[argsort_0] = grad_sp0
        grad_p1 = np.zeros(n1, dtype=np.float64)
        grad_p1[argsort_1] = grad_sp1

        grad[b, g0_mask] = grad_p0 / float(B)
        grad[b, g1_mask] = grad_p1 / float(B)

    loss = loss_total / float(B)
    return loss, grad


def pred_atkinson_individual_loss_and_grad(
    pred: np.ndarray,
    true: np.ndarray,  # noqa: ARG001 - kept for API symmetry
    groups: np.ndarray,  # noqa: ARG001 - kept for API symmetry
    smoothing: float = 1e-6,
    epsilon: float = 0.5,
) -> tuple[float, np.ndarray]:
    """Atkinson inequality over per-individual predicted utilities.

    Unlike the group-MSE Atkinson loss, this operates directly on the
    per-patient prediction vector in each batch row and does not aggregate by
    protected group:

        A_eps(x) = 1 - EDE_eps(x) / mean(x)

    where ``EDE_eps(x) = (mean_i x_i**(1-eps))**(1/(1-eps))`` for
    ``eps != 1`` and ``EDE_1(x) = exp(mean_i log x_i)``. Inputs are clipped to
    ``smoothing`` because the index is defined for positive utilities.
    """
    del true, groups
    eps = max(float(smoothing), 1e-12)
    x_raw = np.asarray(pred, dtype=np.float64)
    x = np.clip(x_raw, eps, None)
    if x.ndim == 1:
        x = x.reshape(1, -1)
        clipped_mask = (x_raw.reshape(1, -1) <= eps)
    elif x.ndim == 2:
        clipped_mask = x_raw <= eps
    else:
        raise ValueError("pred must be a 1D or 2D array.")

    B, n = x.shape
    if n < 2:
        return 0.0, np.zeros_like(pred, dtype=np.float64)

    grad = np.zeros_like(x, dtype=np.float64)
    loss_total = 0.0
    epsilon = float(epsilon)
    for b in range(B):
        xb = x[b]
        mu = float(np.mean(xb))
        mu_safe = max(mu, 1e-12)
        if abs(epsilon - 1.0) < 1e-12:
            ede = float(np.exp(np.mean(np.log(xb))))
            d_ede = ede / (float(n) * xb)
        else:
            one_minus_eps = 1.0 - epsilon
            mean_powered = float(np.mean(np.power(xb, one_minus_eps)))
            mean_powered_safe = max(mean_powered, 1e-12)
            ede = mean_powered_safe ** (1.0 / one_minus_eps)
            d_ede = (
                ede
                / (float(n) * mean_powered_safe)
                * np.power(xb, -epsilon)
            )

        loss_total += 1.0 - ede / mu_safe
        d_mu = 1.0 / float(n)
        grad[b] = -(d_ede * mu_safe - ede * d_mu) / (mu_safe ** 2)

    grad[clipped_mask] = 0.0
    if np.asarray(pred).ndim == 1:
        return float(loss_total), grad.reshape(np.asarray(pred).shape)
    return float(loss_total / float(B)), grad / float(B)


def group_fairness_loss_and_grad(
    pred: np.ndarray,
    true: np.ndarray,
    groups: np.ndarray,
    fairness_type: str = "mad",
    smoothing: float = 1e-6,
    ge_alpha: float = 2.0,
) -> tuple[float, np.ndarray]:
    mode = str(fairness_type).strip().lower()
    if mode == "gap":
        return group_mse_gap_loss_and_grad(
            pred=pred,
            true=true,
            groups=groups,
            smoothing=smoothing,
        )
    if mode == "mad":
        return group_mse_mad_loss_and_grad(
            pred=pred,
            true=true,
            groups=groups,
            smoothing=smoothing,
        )
    if mode == "atkinson":
        return group_mse_atkinson_loss_and_grad(
            pred=pred,
            true=true,
            groups=groups,
            smoothing=smoothing,
        )
    if mode == "gini":
        return group_mse_gini_loss_and_grad(
            pred=pred,
            true=true,
            groups=groups,
            smoothing=smoothing,
        )
    if mode in {"demographic_parity", "dp"}:
        return group_pred_mean_dp_loss_and_grad(
            pred=pred,
            true=true,
            groups=groups,
            smoothing=smoothing,
        )
    if mode in {"wasserstein2_dp", "w2_dp", "w2dp"}:
        return group_pred_wasserstein2_dp_loss_and_grad(
            pred=pred,
            true=true,
            groups=groups,
            smoothing=smoothing,
        )
    if mode in {"atkinson_individual", "atkinson_ind", "atki"}:
        return pred_atkinson_individual_loss_and_grad(
            pred=pred,
            true=true,
            groups=groups,
            smoothing=smoothing,
        )
    if mode in {"bias_parity", "bp"}:
        return group_residual_mean_bias_parity_loss_and_grad(
            pred=pred,
            true=true,
            groups=groups,
            smoothing=smoothing,
        )
    if mode in {"generalized_entropy", "ge"}:
        return group_mse_generalized_entropy_loss_and_grad(
            pred=pred,
            true=true,
            groups=groups,
            alpha=ge_alpha,
            eps=max(float(smoothing), 1e-12),
        )
    raise ValueError(f"Unknown fairness_type: {fairness_type}")
