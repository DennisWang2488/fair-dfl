"""Task layer for data-driven optimization experiments used by the 8 active methods."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from ..losses import pred_atkinson_individual_loss_and_grad
from .base import BaseTask, SplitData, TaskData


@dataclass
class MedicalSplit:
    x: np.ndarray
    y: np.ndarray
    cost: np.ndarray
    race: np.ndarray


@dataclass
class MedicalResourceAllocationTask(BaseTask):
    data_csv: str
    n_sample: int
    data_seed: int
    split_seed: int
    test_fraction: float
    val_fraction: float
    alpha_fair: float
    budget: float
    decision_mode: str = "group"
    fairness_type: str = "mad"
    budget_rho: float = 0.35

    def __post_init__(self) -> None:
        self.name = "medical_resource_allocation"
        self.n_outputs = 1
        mode = str(self.decision_mode).strip().lower()
        if mode not in {"group", "individual"}:
            raise ValueError("decision_mode must be 'group' or 'individual'.")
        self.decision_mode = mode
        ft = str(self.fairness_type).strip().lower()
        if ft == "demographic_parity":
            ft = "dp"
        if ft == "bias_parity":
            ft = "bp"
        if ft in {"w2_dp", "w2dp"}:
            ft = "wasserstein2_dp"
        if ft in {"atkinson_ind", "atki"}:
            ft = "atkinson_individual"
        if ft not in {"gap", "mad", "atkinson", "dp", "bp", "wasserstein2_dp", "atkinson_individual"}:
            raise ValueError(
                "fairness_type must be 'gap', 'mad', 'atkinson', 'dp' "
                "(alias 'demographic_parity'), 'bp' (alias 'bias_parity'), "
                "'wasserstein2_dp' (alias 'w2_dp' / 'w2dp'), or "
                "'atkinson_individual' (alias 'atkinson_ind' / 'atki')."
            )
        self.fairness_type = ft

    def _resolve_data_csv(self) -> Path:
        p = Path(self.data_csv)
        if p.exists():
            return p
        root = Path(__file__).resolve().parents[3]
        p2 = (root / self.data_csv).resolve()
        if p2.exists():
            return p2
        raise FileNotFoundError(f"Medical CSV not found: {self.data_csv}")

    @staticmethod
    def _get_dem_features(df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns if c.startswith("dem_") and "race" not in c]

    @staticmethod
    def _get_comorbidity_features(df: pd.DataFrame) -> List[str]:
        cols = []
        for col in df.columns:
            if col == "gagne_sum_tm1" or col.endswith("_elixhauser_tm1") or col.endswith("_romano_tm1"):
                cols.append(col)
        return cols

    @staticmethod
    def _get_cost_features(df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns if c.startswith("cost_") and c not in {"cost_t", "cost_avoidable_t"}]

    @staticmethod
    def _get_lab_features(df: pd.DataFrame) -> List[str]:
        cols = []
        for col in df.columns:
            if col.endswith("_tests_tm1") or col.endswith("-low_tm1") or col.endswith("-high_tm1") or col.endswith("-normal_tm1"):
                cols.append(col)
        return cols

    @staticmethod
    def _get_med_features(df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns if c.startswith("lasix_")]

    @classmethod
    def _get_all_features(cls, df: pd.DataFrame) -> List[str]:
        return cls._get_dem_features(df) + cls._get_comorbidity_features(df) + cls._get_cost_features(df) + cls._get_lab_features(df) + cls._get_med_features(df)

    @staticmethod
    def _split_indices(n: int, test_fraction: float, val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not (0.0 < test_fraction < 1.0):
            raise ValueError("test_fraction must be in (0,1)")
        if not (0.0 <= val_fraction < 1.0):
            raise ValueError("val_fraction must be in [0,1)")
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        n_test = int(round(test_fraction * n))
        n_test = min(max(n_test, 1), n - 2)
        n_remaining = n - n_test
        n_val = int(round(val_fraction * n_remaining))
        if val_fraction > 0.0:
            n_val = min(max(n_val, 1), n_remaining - 1)
        else:
            n_val = 0
        test_idx = idx[:n_test]
        val_idx = idx[n_test : n_test + n_val]
        train_idx = idx[n_test + n_val :]
        return train_idx, val_idx, test_idx

    def generate_data(self, seed: int) -> TaskData:
        # Use the explicit seed parameter to override data_seed so callers can
        # control randomness through the BaseTask interface.  split_seed is
        # intentionally left unchanged so that callers can vary data generation
        # independently of the train/val/test split.
        if seed != self.data_seed:
            self.data_seed = seed
        path = self._resolve_data_csv()
        df = pd.read_csv(path)
        if self.n_sample > 0 and self.n_sample < len(df):
            df = df.sample(n=self.n_sample, random_state=self.data_seed).reset_index(drop=True)

        feature_cols = self._get_all_features(df)
        if not feature_cols:
            raise ValueError("No medical features selected.")

        x_all = df[feature_cols].to_numpy(dtype=float)
        true_benefit = np.maximum(df["benefit"].to_numpy(dtype=float) * 100.0, 1.0) + 1.0
        cost = np.maximum(df["cost_t_capped"].to_numpy(dtype=float) * 10.0, 1.0)
        race = df["race"].to_numpy(dtype=int)

        # Dynamic budget from budget_rho when budget is not explicitly set (<=0)
        if self.budget <= 0:
            self.budget = float(self.budget_rho * np.sum(cost))

        train_idx, val_idx, test_idx = self._split_indices(
            n=x_all.shape[0],
            test_fraction=float(self.test_fraction),
            val_fraction=float(self.val_fraction),
            seed=int(self.split_seed),
        )
        mean = x_all[train_idx].mean(axis=0, keepdims=True)
        std = x_all[train_idx].std(axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        x_scaled = (x_all - mean) / std

        def make(indices: np.ndarray) -> MedicalSplit:
            return MedicalSplit(
                x=x_scaled[indices].astype(np.float64, copy=True),
                y=true_benefit[indices].astype(np.float64, copy=True),
                cost=cost[indices].astype(np.float64, copy=True),
                race=race[indices].astype(np.int64, copy=True),
            )

        self._splits = {
            "train": make(train_idx),
            "val": make(val_idx),
            "test": make(test_idx),
        }

        return TaskData(
            train=SplitData(x=self._splits["train"].x, y=self._splits["train"].y.reshape(-1, 1)),
            val=SplitData(x=self._splits["val"].x, y=self._splits["val"].y.reshape(-1, 1)),
            test=SplitData(x=self._splits["test"].x, y=self._splits["test"].y.reshape(-1, 1)),
            groups=np.array([0], dtype=int),
            meta={
                "feature_dim": np.asarray([x_scaled.shape[1]], dtype=np.int64),
                "n_total": np.asarray([x_scaled.shape[0]], dtype=np.int64),
                "n_train": np.asarray([train_idx.shape[0]], dtype=np.int64),
                "n_val": np.asarray([val_idx.shape[0]], dtype=np.int64),
                "n_test": np.asarray([test_idx.shape[0]], dtype=np.int64),
                "feature_cols": np.asarray(feature_cols, dtype=object),
            },
        )

    @staticmethod
    def _alpha_obj(utility: np.ndarray, alpha: float) -> float:
        u = np.clip(utility, 1e-12, None)
        if abs(alpha - 1.0) < 1e-12:
            return float(np.sum(np.log(u)))
        return float(np.sum(u ** (1.0 - alpha) / (1.0 - alpha)))

    @staticmethod
    def _normalized_regret(regret: float, denominator_obj: float, eps: float = 1e-12) -> float:
        denom = max(abs(float(denominator_obj)), eps)
        return float(regret / denom)

    @staticmethod
    def _solve_group(pred_r: np.ndarray, cost: np.ndarray, group_idx: np.ndarray, budget: float, alpha: float) -> np.ndarray:
        # Vectorized group-coupled alpha-fair closed form matching Organized-FDFL.
        b = np.asarray(pred_r, dtype=np.float64).reshape(-1)
        c = np.asarray(cost, dtype=np.float64).reshape(-1)
        g = np.asarray(group_idx).reshape(-1)
        n = b.shape[0]
        if n == 0:
            return np.zeros_like(b)
        if alpha <= 0:
            raise ValueError("Group closed form is defined for alpha > 0.")

        if abs(alpha - 1.0) < 1e-9:
            unique_groups, group_inv, group_counts = np.unique(g, return_inverse=True, return_counts=True)
            k = len(unique_groups)
            gk = group_counts[group_inv].astype(np.float64)
            return budget / (float(k) * gk * np.clip(c, 1e-12, None))

        unique_groups, group_inv = np.unique(g, return_inverse=True)
        sort_order = np.argsort(g)
        sorted_groups = g[sort_order]
        _, group_start_indices = np.unique(sorted_groups, return_index=True)

        term_s_all = (np.power(np.clip(c, 1e-12, None), -1.0 / alpha) * np.power(np.clip(b, 1e-12, None), 1.0 / alpha)) ** (1.0 - alpha)
        term_h_all = np.power(np.clip(c, 1e-12, None), (alpha - 1.0) / alpha) * np.power(np.clip(b, 1e-12, None), (1.0 - alpha) / alpha)
        s_k = np.add.reduceat(term_s_all[sort_order], group_start_indices)
        h_k = np.add.reduceat(term_h_all[sort_order], group_start_indices)

        if 0.0 < alpha < 1.0:
            exponent = 1.0 / (alpha - 2.0)
            psi_k = np.power(np.clip(s_k / (1.0 - alpha), 1e-12, None), exponent)
        else:
            # beta = (alpha - 2) / (alpha^2 - 2 alpha + 2); KKT-derived, matches Prop. 1.
            exponent = (alpha - 2.0) / (alpha * alpha - 2.0 * alpha + 2.0)
            psi_k = np.power(np.clip(s_k / (alpha - 1.0), 1e-12, None), exponent)

        xi = float(np.sum(h_k * psi_k))
        if abs(xi) < 1e-12:
            return np.zeros_like(b)

        psi_mapped = psi_k[group_inv]
        phi_all = np.power(np.clip(c, 1e-12, None), -1.0 / alpha) * np.power(np.clip(b, 1e-12, None), (1.0 - alpha) / alpha)
        d_star = (budget / xi) * psi_mapped * phi_all
        return np.clip(d_star, 0.0, None)

    @staticmethod
    def _solve_group_grad_jacobian(pred_r: np.ndarray, cost: np.ndarray, group_idx: np.ndarray, budget: float, alpha: float) -> np.ndarray:
        # Jacobian d(d*)/d(pred_r) for group-coupled objective.
        b = np.asarray(pred_r, dtype=np.float64).reshape(-1)
        c = np.asarray(cost, dtype=np.float64).reshape(-1)
        g = np.asarray(group_idx).reshape(-1)
        n = b.shape[0]
        if n == 0:
            return np.zeros((0, 0), dtype=np.float64)
        if abs(alpha - 1.0) < 1e-9:
            return np.zeros((n, n), dtype=np.float64)

        d_star = MedicalResourceAllocationTask._solve_group(b, c, g, budget, alpha)
        unique_groups, group_inv = np.unique(g, return_inverse=True)
        sort_order = np.argsort(g)
        sorted_groups = g[sort_order]
        _, group_start_indices = np.unique(sorted_groups, return_index=True)

        term_s_all = (np.power(np.clip(c, 1e-12, None), -1.0 / alpha) * np.power(np.clip(b, 1e-12, None), 1.0 / alpha)) ** (1.0 - alpha)
        term_h_all = np.power(np.clip(c, 1e-12, None), (alpha - 1.0) / alpha) * np.power(np.clip(b, 1e-12, None), (1.0 - alpha) / alpha)
        s_k = np.add.reduceat(term_s_all[sort_order], group_start_indices)
        h_k = np.add.reduceat(term_h_all[sort_order], group_start_indices)

        if 0.0 < alpha < 1.0:
            exponent = 1.0 / (alpha - 2.0)
            psi_k = np.power(np.clip(s_k / (1.0 - alpha), 1e-12, None), exponent)
        else:
            # beta = (alpha - 2) / (alpha^2 - 2 alpha + 2); KKT-derived, matches Prop. 1.
            exponent = (alpha - 2.0) / (alpha * alpha - 2.0 * alpha + 2.0)
            psi_k = np.power(np.clip(s_k / (alpha - 1.0), 1e-12, None), exponent)

        xi = float(np.sum(h_k * psi_k))
        if abs(xi) < 1e-12:
            return np.zeros((n, n), dtype=np.float64)

        phi_all = np.power(np.clip(c, 1e-12, None), -1.0 / alpha) * np.power(np.clip(b, 1e-12, None), (1.0 - alpha) / alpha)
        d_s_db_diag = ((1.0 - alpha) / alpha) * np.power(np.clip(c, 1e-12, None), -(1.0 - alpha) / alpha) * np.power(np.clip(b, 1e-12, None), (1.0 - 2.0 * alpha) / alpha)
        d_h_db_diag = ((1.0 - alpha) / alpha) * np.power(np.clip(c, 1e-12, None), (alpha - 1.0) / alpha) * np.power(np.clip(b, 1e-12, None), (1.0 - 2.0 * alpha) / alpha)

        psi_mapped = psi_k[group_inv]
        s_mapped = s_k[group_inv]
        h_mapped = h_k[group_inv]
        d_psi_db_diag = exponent * (psi_mapped / np.clip(s_mapped, 1e-12, None)) * d_s_db_diag
        d_xi_db = d_h_db_diag * psi_mapped + h_mapped * d_psi_db_diag
        d_phi_db_diag = ((1.0 - alpha) / alpha) * (phi_all / np.clip(b, 1e-12, None))

        same_group_mask = (g[:, None] == g[None, :]).astype(np.float64)
        term1_d_n = np.outer(phi_all, d_psi_db_diag) * same_group_mask
        term2_d_n = np.diag(psi_mapped * d_phi_db_diag)
        grad_n = budget * (term1_d_n + term2_d_n)
        grad_xi_term = np.outer(d_star, d_xi_db)
        jac = (grad_n - grad_xi_term) / xi
        return jac.astype(np.float64, copy=False)

    @staticmethod
    def _solve_group_vjp(
        v: np.ndarray,
        pred_r: np.ndarray,
        cost: np.ndarray,
        group_idx: np.ndarray,
        budget: float,
        alpha: float,
    ) -> np.ndarray:
        """Compute v @ (d(d*)/d(pred_r)) without forming the full Jacobian.

        This is an O(n) time and memory implementation of the vector-Jacobian
        product that replaces the O(n^2) explicit Jacobian construction in
        ``_solve_group_grad_jacobian``.

        Parameters
        ----------
        v : (n,) vector to left-multiply with the Jacobian
        pred_r, cost, group_idx : same as ``_solve_group``
        budget, alpha : same as ``_solve_group``

        Returns
        -------
        result : (n,) = v @ Jacobian
        """
        b = np.asarray(pred_r, dtype=np.float64).reshape(-1)
        c = np.asarray(cost, dtype=np.float64).reshape(-1)
        g = np.asarray(group_idx).reshape(-1)
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        n = b.shape[0]
        if n == 0:
            return np.zeros_like(b)
        if abs(alpha - 1.0) < 1e-9:
            return np.zeros_like(b)

        d_star = MedicalResourceAllocationTask._solve_group(b, c, g, budget, alpha)
        unique_groups, group_inv = np.unique(g, return_inverse=True)
        K = len(unique_groups)
        sort_order = np.argsort(g)
        sorted_groups = g[sort_order]
        _, group_start_indices = np.unique(sorted_groups, return_index=True)

        # Per-element intermediate terms (same as in _solve_group_grad_jacobian)
        term_s_all = (np.power(np.clip(c, 1e-12, None), -1.0 / alpha) * np.power(np.clip(b, 1e-12, None), 1.0 / alpha)) ** (1.0 - alpha)
        term_h_all = np.power(np.clip(c, 1e-12, None), (alpha - 1.0) / alpha) * np.power(np.clip(b, 1e-12, None), (1.0 - alpha) / alpha)
        s_k = np.add.reduceat(term_s_all[sort_order], group_start_indices)
        h_k = np.add.reduceat(term_h_all[sort_order], group_start_indices)

        if 0.0 < alpha < 1.0:
            exponent = 1.0 / (alpha - 2.0)
            psi_k = np.power(np.clip(s_k / (1.0 - alpha), 1e-12, None), exponent)
        else:
            # beta = (alpha - 2) / (alpha^2 - 2 alpha + 2); KKT-derived, matches Prop. 1.
            exponent = (alpha - 2.0) / (alpha * alpha - 2.0 * alpha + 2.0)
            psi_k = np.power(np.clip(s_k / (alpha - 1.0), 1e-12, None), exponent)

        xi = float(np.sum(h_k * psi_k))
        if abs(xi) < 1e-12:
            return np.zeros_like(b)

        phi_all = np.power(np.clip(c, 1e-12, None), -1.0 / alpha) * np.power(np.clip(b, 1e-12, None), (1.0 - alpha) / alpha)

        # Per-element derivatives of intermediates w.r.t. b
        d_s_db_diag = ((1.0 - alpha) / alpha) * np.power(np.clip(c, 1e-12, None), -(1.0 - alpha) / alpha) * np.power(np.clip(b, 1e-12, None), (1.0 - 2.0 * alpha) / alpha)
        d_h_db_diag = ((1.0 - alpha) / alpha) * np.power(np.clip(c, 1e-12, None), (alpha - 1.0) / alpha) * np.power(np.clip(b, 1e-12, None), (1.0 - 2.0 * alpha) / alpha)

        psi_mapped = psi_k[group_inv]
        s_mapped = s_k[group_inv]
        h_mapped = h_k[group_inv]
        d_psi_db_diag = exponent * (psi_mapped / np.clip(s_mapped, 1e-12, None)) * d_s_db_diag
        d_xi_db = d_h_db_diag * psi_mapped + h_mapped * d_psi_db_diag
        d_phi_db_diag = ((1.0 - alpha) / alpha) * (phi_all / np.clip(b, 1e-12, None))

        # --- VJP for each Jacobian term ---
        # Full Jacobian is:  Jac = (1/xi) * [ budget*(outer(phi, d_psi_db) * same_group + diag(psi*d_phi_db)) - outer(d*, d_xi_db) ]
        #
        # Term 1: v @ (outer(phi, d_psi_db) * same_group_mask)
        #   = for each element j: d_psi_db[j] * sum_{i in same_group(j)} v[i] * phi[i]
        #   This requires per-group dot products of v and phi.
        v_dot_phi_per_group = np.zeros(K, dtype=np.float64)
        np.add.at(v_dot_phi_per_group, group_inv, v * phi_all)
        vjp_term1 = v_dot_phi_per_group[group_inv] * d_psi_db_diag  # (n,)

        # Term 2: v @ diag(psi_mapped * d_phi_db_diag) = v * psi_mapped * d_phi_db_diag
        vjp_term2 = v * psi_mapped * d_phi_db_diag  # (n,)

        # Term 3: v @ outer(d_star, d_xi_db) = (v . d_star) * d_xi_db
        v_dot_dstar = float(np.dot(v, d_star))
        vjp_term3 = v_dot_dstar * d_xi_db  # (n,)

        result = (budget * (vjp_term1 + vjp_term2) - vjp_term3) / xi
        return result

    @staticmethod
    def _group_objective(decision: np.ndarray, benefit: np.ndarray, group_idx: np.ndarray, alpha: float) -> float:
        # Group-coupled alpha-fair objective matching Organized-FDFL.
        d = np.asarray(decision, dtype=np.float64).reshape(-1)
        b = np.asarray(benefit, dtype=np.float64).reshape(-1)
        g = np.asarray(group_idx).reshape(-1)
        eps = 1e-12
        y = np.clip(b * d, eps, None)
        unique_groups = np.unique(g)
        gk_vals: List[float] = []
        for grp in unique_groups:
            mask = g == grp
            yk = y[mask]
            if len(yk) == 0:
                continue
            if 0.0 < alpha < 1.0:
                gk = float(np.sum(np.power(yk, 1.0 - alpha)) / (1.0 - alpha))
            elif alpha > 1.0:
                gk = float((alpha - 1.0) / np.sum(np.power(yk, 1.0 - alpha)))
            elif abs(alpha - 1.0) < 1e-9:
                gk = float(np.sum(np.log(yk)))
            else:
                gk = float(np.sum(yk))
            gk_vals.append(gk)
        gk_arr = np.clip(np.asarray(gk_vals, dtype=np.float64), eps, None)
        if abs(alpha - 1.0) < 1e-9:
            return float(np.sum(np.log(gk_arr)))
        if abs(alpha) < 1e-12:
            return float(np.sum(gk_arr))
        return float(np.sum(np.power(gk_arr, 1.0 - alpha) / (1.0 - alpha)))

    @staticmethod
    def _group_grad_wrt_decision(decision: np.ndarray, benefit: np.ndarray, group_idx: np.ndarray, alpha: float) -> np.ndarray:
        d = np.asarray(decision, dtype=np.float64).reshape(-1)
        b = np.asarray(benefit, dtype=np.float64).reshape(-1)
        g = np.asarray(group_idx).reshape(-1).astype(np.int64)
        n = d.shape[0]
        if n == 0:
            return np.zeros_like(d)

        u = np.clip(b * d, 1e-12, None)
        unique_groups, group_inv = np.unique(g, return_inverse=True)
        num_groups = len(unique_groups)
        group_counts = np.bincount(group_inv, minlength=num_groups).astype(np.float64)
        beta = alpha

        mu_k = np.zeros(num_groups, dtype=np.float64)
        if beta > 1.0:
            denom_components = np.power(u, 1.0 - beta)
            sum_denom = np.zeros(num_groups, dtype=np.float64)
            np.add.at(sum_denom, group_inv, denom_components)
            mu_k = (beta - 1.0) / np.clip(sum_denom, 1e-12, None)
        else:
            if abs(beta - 1.0) < 1e-12:
                g_beta = np.log(u)
            else:
                g_beta = np.power(u, 1.0 - beta) / (1.0 - beta)
            np.add.at(mu_k, group_inv, g_beta)

        d_j_d_mu = np.zeros_like(mu_k)
        if abs(alpha - 1.0) < 1e-12:
            d_j_d_mu = 1.0 / np.clip(mu_k, 1e-12, None)
        elif abs(alpha) < 1e-12:
            d_j_d_mu = np.ones_like(mu_k)
        else:
            d_j_d_mu = np.power(np.clip(mu_k, 1e-12, None), -alpha)
        d_j_d_mu_map = d_j_d_mu[group_inv]

        if beta > 1.0:
            mu_map = mu_k[group_inv]
            d_mu_d_u = np.power(mu_map, 2.0) * np.power(u, -beta)
        else:
            n_map = np.clip(group_counts[group_inv], 1.0, None)
            d_mu_d_u = (1.0 / n_map) * np.power(u, -beta)

        return d_j_d_mu_map * d_mu_d_u * b

    @staticmethod
    def _solve_alpha_fair(pred_r: np.ndarray, cost: np.ndarray, alpha: float, budget: float) -> np.ndarray:
        r = np.clip(pred_r, 1e-8, None)
        c = np.clip(cost, 1e-8, None)
        n = r.shape[0]
        if n == 0:
            return np.zeros_like(r)
        if abs(alpha - 1.0) < 1e-12:
            return budget / (n * c)
        if abs(alpha) < 1e-12:
            ratios = r / c
            out = np.zeros_like(r)
            out[int(np.argmax(ratios))] = budget / c[int(np.argmax(ratios))]
            return out
        common = np.power(c, -1.0 / alpha) * np.power(r, 1.0 / alpha - 1.0)
        denom = float(np.sum(c * common))
        if denom <= 1e-12:
            return np.zeros_like(r)
        return (budget * common) / denom

    @staticmethod
    def _solve_jacobian(pred_r: np.ndarray, cost: np.ndarray, alpha: float, budget: float, d_star: np.ndarray) -> np.ndarray:
        n = pred_r.shape[0]
        if n == 0 or abs(alpha) < 1e-12 or abs(alpha - 1.0) < 1e-12:
            return np.zeros((n, n), dtype=np.float64)
        r = np.clip(pred_r, 1e-8, None)
        c = np.clip(cost, 1e-8, None)
        term = (1.0 / alpha - 1.0) / r
        jac = -np.outer(d_star, d_star * term * c) / float(max(budget, 1e-12))
        diag = d_star * term * (1.0 - c * d_star / float(max(budget, 1e-12)))
        np.fill_diagonal(jac, diag)
        return jac

    @staticmethod
    def _solve_alpha_fair_vjp(
        v: np.ndarray, pred_r: np.ndarray, cost: np.ndarray, alpha: float, budget: float, d_star: np.ndarray
    ) -> np.ndarray:
        """Return ``v @ d(solve_alpha_fair)/d(pred_r)`` without materialising J.

        The individual alpha-fair closed-form solution has a diagonal plus
        rank-one Jacobian. Materialising it is O(n^2), which is prohibitive for
        the full healthcare cohort; the vector-Jacobian product simplifies to

            (vJ)_j = d_j * t_j * (v_j - c_j / B * sum_i v_i d_i),

        where ``t_j = (1/alpha - 1) / pred_r_j`` and ``B`` is the budget.
        """
        if pred_r.size == 0 or abs(alpha) < 1e-12 or abs(alpha - 1.0) < 1e-12:
            return np.zeros_like(pred_r, dtype=np.float64)
        r = np.clip(pred_r, 1e-8, None)
        c = np.clip(cost, 1e-8, None)
        budget_safe = float(max(budget, 1e-12))
        term = (1.0 / alpha - 1.0) / r
        vd_sum = float(np.dot(v, d_star))
        return d_star * term * (v - (c / budget_safe) * vd_sum)

    def _decision_regret_and_grad(
        self,
        pred_r: np.ndarray,
        true_r: np.ndarray,
        cost: np.ndarray,
        race: np.ndarray,
        need_grad: bool,
    ) -> Tuple[float, float, float, np.ndarray, int, float, np.ndarray, np.ndarray]:
        t0 = perf_counter()
        if self.decision_mode == "group":
            d_true = self._solve_group(true_r, cost, race, budget=self.budget, alpha=self.alpha_fair)
            d_hat = self._solve_group(pred_r, cost, race, budget=self.budget, alpha=self.alpha_fair)
            obj_true = self._group_objective(d_true, true_r, race, alpha=self.alpha_fair)
            obj_hat = self._group_objective(d_hat, true_r, race, alpha=self.alpha_fair)
        else:
            d_true = self._solve_alpha_fair(true_r, cost, alpha=self.alpha_fair, budget=self.budget)
            d_hat = self._solve_alpha_fair(pred_r, cost, alpha=self.alpha_fair, budget=self.budget)
            obj_true = self._alpha_obj(true_r * d_true, alpha=self.alpha_fair)
            obj_hat = self._alpha_obj(true_r * d_hat, alpha=self.alpha_fair)

        regret = obj_true - obj_hat
        loss_dec = float(max(0.0, regret))
        loss_dec_normalized = self._normalized_regret(loss_dec, obj_true, eps=1e-7)
        loss_dec_normalized_pred_obj = self._normalized_regret(loss_dec, obj_hat, eps=1e-7)

        grad_pred = np.zeros_like(pred_r)
        if need_grad and loss_dec > 0.0:
            if self.decision_mode == "group":
                grad_obj_d = self._group_grad_wrt_decision(d_hat, true_r, race, alpha=self.alpha_fair)
                grad_pred = -self._solve_group_vjp(grad_obj_d, pred_r, cost, race, budget=self.budget, alpha=self.alpha_fair)
            else:
                if abs(self.alpha_fair - 1.0) < 1e-12:
                    grad_obj_d = 1.0 / np.clip(d_hat, 1e-12, None)
                elif abs(self.alpha_fair) < 1e-12:
                    grad_obj_d = true_r
                else:
                    grad_obj_d = np.power(np.clip(true_r, 1e-12, None), 1.0 - self.alpha_fair) * np.power(np.clip(d_hat, 1e-12, None), -self.alpha_fair)
                grad_pred = -self._solve_alpha_fair_vjp(
                    grad_obj_d,
                    pred_r,
                    cost,
                    alpha=self.alpha_fair,
                    budget=self.budget,
                    d_star=d_hat,
                )
        decision_ms = (perf_counter() - t0) * 1000.0
        return loss_dec, loss_dec_normalized, loss_dec_normalized_pred_obj, grad_pred, 2, decision_ms, d_true, d_hat

    @staticmethod
    def _pred_loss_and_grad(pred: np.ndarray, true: np.ndarray) -> Tuple[float, np.ndarray]:
        diff = pred - true
        return float(np.mean(diff * diff)), 2.0 * diff / float(max(len(pred), 1))

    @staticmethod
    def _fair_loss_and_grad_gap(pred: np.ndarray, true: np.ndarray, race: np.ndarray, smoothing: float) -> Tuple[float, np.ndarray]:
        """Group accuracy parity: smoothed |MSE_0 - MSE_1| (original metric)."""
        groups = np.unique(race)
        if len(groups) < 2:
            return 0.0, np.zeros_like(pred)
        err = (pred - true) ** 2
        if len(groups) == 2:
            g0, g1 = groups[0], groups[1]
            m0, m1 = race == g0, race == g1
            n0, n1 = int(m0.sum()), int(m1.sum())
            if n0 == 0 or n1 == 0:
                return 0.0, np.zeros_like(pred)
            mse0 = float(np.mean(err[m0]))
            mse1 = float(np.mean(err[m1]))
            gap = mse0 - mse1
            loss = float(np.sqrt(gap * gap + smoothing))
            coeff = gap / max(loss, 1e-12)
            grad = np.zeros_like(pred)
            grad[m0] = coeff * 2.0 * (pred[m0] - true[m0]) / float(n0)
            grad[m1] = -coeff * 2.0 * (pred[m1] - true[m1]) / float(n1)
            return loss, grad

        group_mse = np.asarray([float(np.mean(err[race == g])) for g in groups], dtype=np.float64)
        mean_mse = float(group_mse.mean())
        gap = group_mse - mean_mse
        smooth_abs = np.sqrt(gap * gap + smoothing)
        loss = float(np.mean(smooth_abs))
        dphi = gap / smooth_abs
        dloss_dmse = (dphi - dphi.mean()) / float(len(groups))
        grad = np.zeros_like(pred)
        for i, g in enumerate(groups):
            m = race == g
            ng = int(m.sum())
            if ng == 0:
                continue
            grad[m] = dloss_dmse[i] * 2.0 * (pred[m] - true[m]) / float(ng)
        return loss, grad

    @staticmethod
    def _fair_loss_and_grad_mad(pred: np.ndarray, true: np.ndarray, race: np.ndarray, smoothing: float) -> Tuple[float, np.ndarray]:
        """Mean Absolute Deviation (MAD) of per-group MSE from grand mean MSE.

        MAD = mean_k( sqrt( (mean_err_k - grand_mean_err)^2 + smoothing ) )
        where mean_err_k = mean( (pred - true)^2 ) for individuals in group k.
        """
        groups = np.unique(race)
        K = len(groups)
        if K < 2:
            return 0.0, np.zeros_like(pred)

        err = (pred - true) ** 2
        group_mse = np.zeros(K, dtype=np.float64)
        group_masks: List[np.ndarray] = []
        group_sizes = np.zeros(K, dtype=np.float64)
        for i, g in enumerate(groups):
            m = race == g
            group_masks.append(m)
            ng = float(m.sum())
            group_sizes[i] = ng
            group_mse[i] = float(np.mean(err[m])) if ng > 0 else 0.0

        grand_mean = float(group_mse.mean())
        dev = group_mse - grand_mean                       # (K,)
        smooth_abs = np.sqrt(dev * dev + smoothing)        # (K,)
        loss = float(np.mean(smooth_abs))                  # scalar

        # Gradient: d(loss)/d(pred_i)
        # d(loss)/d(group_mse_k) via chain rule:
        #   d(loss)/d(smooth_abs_k) = 1/K
        #   d(smooth_abs_k)/d(dev_k) = dev_k / smooth_abs_k
        #   d(dev_k)/d(group_mse_j) = delta_{kj} - 1/K
        # So: d(loss)/d(group_mse_k) = (1/K) * sum_j [ (dev_j / smooth_abs_j) * (delta_{jk} - 1/K) ]
        #                             = (1/K) * (dev_k / smooth_abs_k - mean(dev / smooth_abs))
        dphi = dev / smooth_abs                             # (K,)
        dloss_dmse = (dphi - dphi.mean()) / float(K)        # (K,)

        # d(group_mse_k)/d(pred_i) = (2/n_k) * (pred_i - true_i) for i in group k
        grad = np.zeros_like(pred)
        for i in range(K):
            m = group_masks[i]
            ng = group_sizes[i]
            if ng == 0:
                continue
            grad[m] = dloss_dmse[i] * 2.0 * (pred[m] - true[m]) / ng
        return loss, grad

    @staticmethod
    def _fair_loss_and_grad_atkinson(
        pred: np.ndarray, true: np.ndarray, race: np.ndarray, smoothing: float, epsilon: float = 0.5
    ) -> Tuple[float, np.ndarray]:
        """Atkinson index of per-group MSE values.

        For epsilon != 1:
            A = 1 - ( mean_k(mean_err_k^(1-eps)) )^(1/(1-eps)) / grand_mean
        For epsilon == 1:
            A = 1 - ( prod_k(mean_err_k) )^(1/K) / grand_mean

        Uses smoothing as a floor on per-group MSE to avoid log(0)/pow(0).
        """
        groups = np.unique(race)
        K = len(groups)
        if K < 2:
            return 0.0, np.zeros_like(pred)

        err = (pred - true) ** 2
        group_mse = np.zeros(K, dtype=np.float64)
        group_masks: List[np.ndarray] = []
        group_sizes = np.zeros(K, dtype=np.float64)
        for i, g in enumerate(groups):
            m = race == g
            group_masks.append(m)
            ng = float(m.sum())
            group_sizes[i] = ng
            group_mse[i] = max(float(np.mean(err[m])), smoothing) if ng > 0 else smoothing

        grand_mean = float(np.mean(group_mse))
        grand_mean_safe = max(grand_mean, 1e-12)

        if abs(epsilon - 1.0) < 1e-12:
            # Geometric mean case: A = 1 - geomean(group_mse) / grand_mean
            log_mse = np.log(group_mse)
            geomean = float(np.exp(np.mean(log_mse)))
            loss = max(1.0 - geomean / grand_mean_safe, 0.0)

            # Gradient via chain rule:
            # d(A)/d(group_mse_k) = -(1/K) * geomean / (group_mse_k * grand_mean)
            #                       + geomean / (K * grand_mean^2)
            d_A_d_mse = np.zeros(K, dtype=np.float64)
            for k in range(K):
                d_geomean_d_mse_k = geomean / (float(K) * group_mse[k])
                d_grand_d_mse_k = 1.0 / float(K)
                d_A_d_mse[k] = -(d_geomean_d_mse_k * grand_mean_safe - geomean * d_grand_d_mse_k) / (grand_mean_safe ** 2)
        else:
            # General case
            one_minus_eps = 1.0 - epsilon
            powered = np.power(group_mse, one_minus_eps)     # (K,)
            mean_powered = float(np.mean(powered))
            mean_powered_safe = max(mean_powered, 1e-12)
            ede = mean_powered_safe ** (1.0 / one_minus_eps)  # equally distributed equivalent
            loss = max(1.0 - ede / grand_mean_safe, 0.0)

            # d(A)/d(group_mse_k):
            # Let M = mean(mu_k^(1-eps)), EDE = M^(1/(1-eps)), mu_bar = mean(mu_k)
            # d(EDE)/d(mu_k) = (1/K) * EDE / M * mu_k^(-eps)  [= (1/K) * M^(1/(1-eps)-1) * mu_k^(-eps)]
            # d(mu_bar)/d(mu_k) = 1/K
            # d(A)/d(mu_k) = -(d(EDE)/d(mu_k) * mu_bar - EDE * d(mu_bar)/d(mu_k)) / mu_bar^2
            d_A_d_mse = np.zeros(K, dtype=np.float64)
            for k in range(K):
                d_ede_d_mse_k = (1.0 / float(K)) * ede / mean_powered_safe * (group_mse[k] ** (-epsilon))
                d_grand_d_mse_k = 1.0 / float(K)
                d_A_d_mse[k] = -(d_ede_d_mse_k * grand_mean_safe - ede * d_grand_d_mse_k) / (grand_mean_safe ** 2)

        # Chain to individual predictions: d(A)/d(pred_i) = d(A)/d(group_mse_k) * d(group_mse_k)/d(pred_i)
        grad = np.zeros_like(pred)
        for k in range(K):
            m = group_masks[k]
            ng = group_sizes[k]
            if ng == 0:
                continue
            grad[m] = d_A_d_mse[k] * 2.0 * (pred[m] - true[m]) / ng
        return loss, grad

    @staticmethod
    def _fair_loss_and_grad_bias_parity(
        pred: np.ndarray, true: np.ndarray, race: np.ndarray, smoothing: float
    ) -> Tuple[float, np.ndarray]:
        """Bias parity (calibration first moment): MAD of per-group mean residuals.

        Penalises systematic over- or under-prediction across groups while
        ignoring symmetric noise. Sits in the calibration / sufficiency family
        — distinct from ``mad`` (separation, equalised errors) and ``dp``
        (independence, equalised raw predictions).

        L = mean_g sqrt( (b_g - b_bar)^2 + smoothing )
            b_g   = mean(pred - true | race=g)
            b_bar = mean_g(b_g)
        """
        groups = np.unique(race)
        K = len(groups)
        if K < 2:
            return 0.0, np.zeros_like(pred)

        residual = pred - true
        group_bias = np.zeros(K, dtype=np.float64)
        group_masks: List[np.ndarray] = []
        group_sizes = np.zeros(K, dtype=np.float64)
        for i, g in enumerate(groups):
            m = race == g
            group_masks.append(m)
            ng = float(m.sum())
            group_sizes[i] = ng
            group_bias[i] = float(np.mean(residual[m])) if ng > 0 else 0.0

        mean_of_bias = float(group_bias.mean())
        dev = group_bias - mean_of_bias
        smooth_abs = np.sqrt(dev * dev + smoothing)
        loss = float(np.mean(smooth_abs))

        # Same chain as DP/MAD up to d(b_g)/d(pred_i) = 1/n_g for i in g.
        dphi = dev / smooth_abs
        dloss_db = (dphi - dphi.mean()) / float(K)

        grad = np.zeros_like(pred)
        for i in range(K):
            m = group_masks[i]
            ng = group_sizes[i]
            if ng == 0:
                continue
            grad[m] = dloss_db[i] / ng
        return loss, grad

    @staticmethod
    def _fair_loss_and_grad_dp(
        pred: np.ndarray, true: np.ndarray, race: np.ndarray, smoothing: float
    ) -> Tuple[float, np.ndarray]:
        """Demographic parity on predictions: MAD of per-group mean predictions.

        Equalises the per-group mean *predicted* benefit (not per-group MSE),
        so the loss does not depend on ``true`` at all.

        L = mean_g sqrt( (mu_g - mu_bar)^2 + smoothing )
            mu_g   = mean(pred | race=g)
            mu_bar = mean_g(mu_g)
        """
        del true  # unused; demographic parity ignores labels
        groups = np.unique(race)
        K = len(groups)
        if K < 2:
            return 0.0, np.zeros_like(pred)

        group_means = np.zeros(K, dtype=np.float64)
        group_masks: List[np.ndarray] = []
        group_sizes = np.zeros(K, dtype=np.float64)
        for i, g in enumerate(groups):
            m = race == g
            group_masks.append(m)
            ng = float(m.sum())
            group_sizes[i] = ng
            group_means[i] = float(np.mean(pred[m])) if ng > 0 else 0.0

        mean_of_means = float(group_means.mean())
        dev = group_means - mean_of_means                       # (K,)
        smooth_abs = np.sqrt(dev * dev + smoothing)             # (K,)
        loss = float(np.mean(smooth_abs))

        # Gradient: identical chain to MAD up to d(mu_g)/d(pred_i) instead of MSE.
        # d(loss)/d(mu_g) = (1/K) * (dev_g/smooth_abs_g - mean_h(dev_h/smooth_abs_h))
        # d(mu_g)/d(pred_i) = (1/n_g) * 1[i in g]
        dphi = dev / smooth_abs                                 # (K,)
        dloss_dmu = (dphi - dphi.mean()) / float(K)             # (K,)

        grad = np.zeros_like(pred)
        for i in range(K):
            m = group_masks[i]
            ng = group_sizes[i]
            if ng == 0:
                continue
            grad[m] = dloss_dmu[i] / ng
        return loss, grad

    @staticmethod
    def _fair_loss_and_grad_w2_dp(
        pred: np.ndarray, true: np.ndarray, race: np.ndarray, smoothing: float,
        n_quantiles: int = 256,
    ) -> Tuple[float, np.ndarray]:
        """Wasserstein-2 demographic parity: squared 2-Wasserstein distance
        between the two group-conditional predicted-utility distributions.

        L = W_2^2(p_pred | race=0, p_pred | race=1)
          = mean_k (q_0[k] - q_1[k])^2

        where q_g[k] is the empirical predicted-utility quantile of group g
        at rank probability tau_k = (k - 0.5) / m, m = min(n_quantiles, n_0, n_1).

        Resampling to a common quantile grid handles unequal group sizes.
        Reference: Chzhen et al. (NeurIPS 2020) "Fair regression with
        Wasserstein barycenters". Binary protected attribute only.
        """
        del true, smoothing  # unused for W2-DP
        groups = np.unique(race)
        if len(groups) != 2:
            return 0.0, np.zeros_like(pred)

        g0_mask = race == groups[0]
        g1_mask = race == groups[1]
        n0 = int(g0_mask.sum())
        n1 = int(g1_mask.sum())
        if n0 == 0 or n1 == 0:
            return 0.0, np.zeros_like(pred)
        m = int(min(n_quantiles, n0, n1))
        if m < 2:
            return 0.0, np.zeros_like(pred)

        target_tau = (np.arange(m, dtype=np.float64) + 0.5) / float(m)
        src_tau_0 = (np.arange(n0, dtype=np.float64) + 0.5) / float(n0)
        src_tau_1 = (np.arange(n1, dtype=np.float64) + 0.5) / float(n1)

        p0 = pred[g0_mask].astype(np.float64, copy=False)
        p1 = pred[g1_mask].astype(np.float64, copy=False)
        argsort_0 = np.argsort(p0, kind="stable")
        argsort_1 = np.argsort(p1, kind="stable")
        sp0 = p0[argsort_0]
        sp1 = p1[argsort_1]

        # Local copy of the quantile interpolation helper (avoids cross-module
        # dependency on losses._interp_quantile_with_weights).
        def _interp(sorted_vec, src_tau):
            n = len(sorted_vec)
            pos = np.searchsorted(src_tau, target_tau)
            pos = np.clip(pos, 1, n - 1)
            idx_lo = pos - 1
            idx_hi = pos
            tau_lo = src_tau[idx_lo]
            tau_hi = src_tau[idx_hi]
            denom = np.maximum(tau_hi - tau_lo, 1e-15)
            w_hi = (target_tau - tau_lo) / denom
            w_lo = 1.0 - w_hi
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

        q0, idx0_lo, idx0_hi, w0_lo, w0_hi = _interp(sp0, src_tau_0)
        q1, idx1_lo, idx1_hi, w1_lo, w1_hi = _interp(sp1, src_tau_1)

        diff = q0 - q1
        loss = float(np.mean(diff * diff))

        dq0 = 2.0 * diff / float(m)
        dq1 = -dq0

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

        grad = np.zeros_like(pred, dtype=np.float64)
        grad[g0_mask] = grad_p0
        grad[g1_mask] = grad_p1
        return loss, grad

    def _fair_loss_and_grad_atkinson_individual(
        self, pred: np.ndarray, true: np.ndarray, race: np.ndarray, smoothing: float
    ) -> Tuple[float, np.ndarray]:
        """Atkinson inequality over per-patient predicted utilities.

        The EDE parameter matches the downstream alpha-fair utility curvature:
        ``epsilon = alpha_fair``. This is the standard Atkinson/alpha-fair
        coupling because both use the same ``1 - alpha`` power inside the
        equally distributed equivalent.
        """
        return pred_atkinson_individual_loss_and_grad(
            pred=pred,
            true=true,
            groups=race,
            smoothing=smoothing,
            epsilon=float(self.alpha_fair),
        )

    def _compute_fairness(
        self, pred: np.ndarray, true: np.ndarray, race: np.ndarray, smoothing: float
    ) -> Tuple[float, np.ndarray]:
        """Dispatch to the selected fairness metric based on self.fairness_type."""
        if self.fairness_type == "gap":
            return self._fair_loss_and_grad_gap(pred, true, race, smoothing)
        elif self.fairness_type == "mad":
            return self._fair_loss_and_grad_mad(pred, true, race, smoothing)
        elif self.fairness_type == "atkinson":
            return self._fair_loss_and_grad_atkinson(pred, true, race, smoothing)
        elif self.fairness_type in {"dp", "demographic_parity"}:
            return self._fair_loss_and_grad_dp(pred, true, race, smoothing)
        elif self.fairness_type in {"bp", "bias_parity"}:
            return self._fair_loss_and_grad_bias_parity(pred, true, race, smoothing)
        elif self.fairness_type in {"wasserstein2_dp", "w2_dp", "w2dp"}:
            return self._fair_loss_and_grad_w2_dp(pred, true, race, smoothing)
        elif self.fairness_type == "atkinson_individual":
            return self._fair_loss_and_grad_atkinson_individual(pred, true, race, smoothing)
        else:
            raise ValueError(f"Unknown fairness_type: {self.fairness_type}")

    def compute_batch(
        self,
        raw_pred: np.ndarray,
        true: np.ndarray,
        cost: np.ndarray,
        race: np.ndarray,
        need_grads: bool,
        fairness_smoothing: float = 1e-6,
    ) -> Dict[str, np.ndarray | float]:
        pred = np.clip(np.asarray(raw_pred, dtype=float).reshape(-1), 1e-6, None)
        y = np.asarray(true, dtype=float).reshape(-1)
        c = np.asarray(cost, dtype=float).reshape(-1)
        r = np.asarray(race, dtype=int).reshape(-1)

        loss_dec, loss_dec_normalized, loss_dec_normalized_pred_obj, grad_dec, solver_calls, decision_ms, d_true, d_hat = self._decision_regret_and_grad(
            pred_r=pred,
            true_r=y,
            cost=c,
            race=r,
            need_grad=need_grads,
        )
        loss_pred, grad_pred = self._pred_loss_and_grad(pred=pred, true=y)
        loss_fair, grad_fair = self._compute_fairness(pred=pred, true=y, race=r, smoothing=float(fairness_smoothing))
        if not need_grads:
            # Only zero the decision gradient; prediction and fairness
            # gradients are cheap and always needed (e.g. by FPTO).
            grad_dec = np.zeros_like(pred)
        return {
            "loss_dec": float(loss_dec),
            "loss_dec_normalized": float(loss_dec_normalized),
            "loss_dec_normalized_pred_obj": float(loss_dec_normalized_pred_obj),
            "loss_pred": float(loss_pred),
            "loss_fair": float(loss_fair),
            "grad_dec": grad_dec,
            "grad_pred": grad_pred,
            "grad_fair": grad_fair,
            "solver_calls": int(solver_calls),
            "decision_ms": float(decision_ms),
            "decision_true": d_true,
            "decision_pred": d_hat,
        }

    def evaluate_split(self, split: str, pred: np.ndarray, fairness_smoothing: float = 1e-6) -> Dict[str, float]:
        s = self._splits[split]
        out = self.compute_batch(
            raw_pred=np.asarray(pred, dtype=float).reshape(-1),
            true=s.y,
            cost=s.cost,
            race=s.race,
            need_grads=False,
            fairness_smoothing=fairness_smoothing,
        )
        return {
            "regret": float(out["loss_dec"]),
            "regret_normalized": float(out["loss_dec_normalized"]),
            "regret_normalized_pred_obj": float(out["loss_dec_normalized_pred_obj"]),
            "pred_mse": float(out["loss_pred"]),
            "fairness": float(out["loss_fair"]),
            "solver_calls_eval": float(out["solver_calls"]),
            "decision_ms_eval": float(out["decision_ms"]),
        }

    def sample_batch(self, split: str, batch_size: int, rng: np.random.Generator) -> MedicalSplit:
        s = self._splits[split]
        n = s.x.shape[0]
        if batch_size <= 0 or batch_size >= n:
            return s
        idx = rng.choice(n, size=batch_size, replace=False)
        return MedicalSplit(x=s.x[idx], y=s.y[idx], cost=s.cost[idx], race=s.race[idx])

    # ------------------------------------------------------------------
    # Generic decision gradient interface
    # ------------------------------------------------------------------

    def solve_decision(self, pred: np.ndarray, **ctx: Any) -> np.ndarray:
        pred_r = np.clip(np.asarray(pred, dtype=float).reshape(-1), 1e-6, None)
        cost = ctx.get("cost", None)
        race = ctx.get("race", None)
        if cost is None or race is None:
            raise ValueError("cost and race must be provided via ctx.")
        cost = np.asarray(cost, dtype=float).reshape(-1)
        race = np.asarray(race, dtype=int).reshape(-1)
        if self.decision_mode == "group":
            return self._solve_group(pred_r, cost, race, budget=self.budget, alpha=self.alpha_fair)
        else:
            return self._solve_alpha_fair(pred_r, cost, alpha=self.alpha_fair, budget=self.budget)

    def evaluate_objective(self, decision: np.ndarray, true: np.ndarray, **ctx: Any) -> float:
        true_r = np.asarray(true, dtype=float).reshape(-1)
        decision = np.asarray(decision, dtype=float).reshape(-1)
        race = ctx.get("race", None)
        if self.decision_mode == "group":
            if race is None:
                raise ValueError("race must be provided via ctx for group decision_mode.")
            race = np.asarray(race, dtype=int).reshape(-1)
            return self._group_objective(decision, true_r, race, alpha=self.alpha_fair)
        else:
            return self._alpha_obj(true_r * decision, alpha=self.alpha_fair)

    def supported_gradient_strategies(self) -> List[str]:
        return ["analytic", "finite_diff"]

    # Legacy interface (unused for advanced medical run path)
    def compute(self, raw_pred, true, need_grads, fairness_smoothing: float = 1e-6):
        raise NotImplementedError("Use compute_batch for medical_resource_allocation in advanced runner.")
