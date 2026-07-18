import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import cholesky
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ---------- ROBUSTNA POZITIVNO DEFINITNA KOREKCIJA ----------
def nearest_positive_definite(A, epsilon=1e-8, max_shift=100):
    """
    Vraća najbližu pozitivno definitnu matricu.
    Ako eigendecomposition ne uspije, dodaje dijagonalni šift.
    """
    A = np.asarray(A, dtype=float)
    A = (A + A.T) / 2

    # Ako je matrica 1x1, vrati [[1]]
    if A.shape[0] == 1:
        return np.array([[1.0]])

    # Pokušaj sa postupnim povećanjem šifta
    for shift in np.logspace(-8, 2, num=max_shift):
        try:
            A_shift = A + shift * np.eye(A.shape[0])
            eigvals, eigvecs = np.linalg.eigh(A_shift)
            if np.min(eigvals) > 0:
                eigvals = np.maximum(eigvals, epsilon)
                A_corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
                # Dijagonala na 1
                d = np.sqrt(np.diag(A_corr))
                A_corr = A_corr / np.outer(d, d)
                return A_corr
        except np.linalg.LinAlgError:
            continue

    # Krajnji slučaj – veliki šift
    A_shift = A + 1e-3 * np.eye(A.shape[0])
    eigvals, eigvecs = np.linalg.eigh(A_shift)
    eigvals = np.maximum(eigvals, epsilon)
    A_corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(A_corr))
    A_corr = A_corr / np.outer(d, d)
    return A_corr


class HawkesMertonContagion:
    def __init__(self, n_companies, T=1.0, dt=1/252,
                 use_heston=False, use_stochastic_rate=False,
                 barrier_growth_rate=0.02, barrier_target=None,
                 barrier_mean_reversion=0.5,
                 jump_intensity=0.5, jump_mean=-0.15, jump_std=0.10,
                 recovery_base=0.4, recovery_sensitivity=-0.5,
                 regime_switching=False):
        self.N = n_companies
        self.T = T
        self.dt = dt
        self.steps = int(self.T / self.dt) + 1
        self.use_heston = use_heston
        self.use_stochastic_rate = use_stochastic_rate
        self.V0 = np.ones(self.N) * 100.0
        self.drift = np.ones(self.N) * 0.05
        self.vol = np.ones(self.N) * 0.20
        self.kappa = np.ones(self.N) * 2.0
        self.theta = np.ones(self.N) * 0.04
        self.xi = np.ones(self.N) * 0.3
        self.v0 = np.ones(self.N) * 0.04
        self.rho_asset_vol = np.ones(self.N) * -0.7
        self.D0 = np.ones(self.N) * 60.0
        self.barrier_growth_rate = barrier_growth_rate
        self.barrier_target = barrier_target
        self.barrier_mean_reversion = barrier_mean_reversion
        self.r0 = 0.02
        self.r_mean = 0.03
        self.r_speed = 0.5
        self.r_vol = 0.01
        self.rho_asset_rate = -0.2
        self.base_intensity = np.ones(self.N) * 0.01
        self.beta = 2.0
        self.gamma = np.zeros((self.N, self.N))
        self.corr_assets = np.eye(self.N)
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std
        self.recovery_base = recovery_base
        self.recovery_sensitivity = recovery_sensitivity
        self._L_assets = None
        self._L_gbm_rate = None
        self._L_heston = None
        self.exposures = None

        # Regime-Switching
        self.regime_switching = regime_switching
        if regime_switching:
            self.regime_transition = np.array([[0.95, 0.05], [0.10, 0.90]])
            self.regime_jump_intensity = [0.1, 0.8]
            self.regime_recovery_base = [0.55, 0.35]

    def set_contagion_network(self, gamma_matrix):
        if gamma_matrix.shape != (self.N, self.N):
            raise ValueError("Gamma matrica mora biti NxN.")
        np.fill_diagonal(gamma_matrix, 0.0)
        self.gamma = gamma_matrix

    def set_correlation_assets(self, corr_matrix):
        if corr_matrix.shape != (self.N, self.N):
            raise ValueError("Korelaciona matrica mora biti NxN.")
        self.corr_assets = corr_matrix

    def calibrate_kmv(self, equity_values, equity_vols, debt_values,
                      risk_free_rate, time_horizon=1.0, max_iter=100, tol=1e-6):
        N = self.N
        V0_cal = np.zeros(N)
        vol_cal = np.zeros(N)
        for i in range(N):
            E = equity_values[i]
            sigma_E = equity_vols[i]
            D = debt_values[i]
            r = risk_free_rate
            T = time_horizon
            V = E + D
            sigma_V = sigma_E * E / V
            sigma_V = min(sigma_V, 0.8)
            for _ in range(max_iter):
                d1 = (np.log(V / D) + (r + 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
                d2 = d1 - sigma_V * np.sqrt(T)
                E_model = V * stats.norm.cdf(d1) - D * np.exp(-r * T) * stats.norm.cdf(d2)
                sigma_E_model = (V / E_model) * stats.norm.cdf(d1) * sigma_V
                err_E = E_model - E
                err_sigma = sigma_E_model - sigma_E
                if abs(err_E) < tol and abs(err_sigma) < tol:
                    break
                if abs(err_E) > 1e-4:
                    V = V - 0.5 * err_E
                if abs(err_sigma) > 1e-6:
                    sigma_V = sigma_V - 0.3 * err_sigma
                V = max(V, D + 1e-6)
                sigma_V = max(sigma_V, 0.01)
                sigma_V = min(sigma_V, 0.8)
            V0_cal[i] = V
            vol_cal[i] = sigma_V
        self.V0 = V0_cal
        self.vol = vol_cal
        self.D0 = debt_values
        return V0_cal, vol_cal

    def _prepare_cholesky(self):
        """Priprema Cholesky faktorizaciju, sa oporavkom od grešaka."""
        if self.N == 1:
            # Jedna firma – nema korelacije
            self._L_assets = np.array([[1.0]])
            return

        if self.use_heston:
            dim = 2 * self.N + 1
            corr_full = np.eye(dim)
            corr_full[:self.N, :self.N] = self.corr_assets
            for i in range(self.N):
                corr_full[i, self.N + i] = self.rho_asset_vol[i]
                corr_full[self.N + i, i] = self.rho_asset_vol[i]
            corr_full[:self.N, -1] = self.rho_asset_rate
            corr_full[-1, :self.N] = self.rho_asset_rate
            corr_full = nearest_positive_definite(corr_full)
            try:
                self._L_heston = cholesky(corr_full, lower=True)
            except np.linalg.LinAlgError:
                corr_full += 1e-6 * np.eye(dim)
                self._L_heston = cholesky(corr_full, lower=True)
        else:
            if self.use_stochastic_rate:
                dim = self.N + 1
                corr_full = np.eye(dim)
                corr_full[:self.N, :self.N] = self.corr_assets
                corr_full[:self.N, -1] = self.rho_asset_rate
                corr_full[-1, :self.N] = self.rho_asset_rate
                corr_full = nearest_positive_definite(corr_full)
                try:
                    self._L_gbm_rate = cholesky(corr_full, lower=True)
                except np.linalg.LinAlgError:
                    corr_full += 1e-6 * np.eye(dim)
                    self._L_gbm_rate = cholesky(corr_full, lower=True)
            else:
                corr_assets = nearest_positive_definite(self.corr_assets)
                try:
                    self._L_assets = cholesky(corr_assets, lower=True)
                except np.linalg.LinAlgError:
                    corr_assets += 1e-6 * np.eye(self.N)
                    self._L_assets = cholesky(corr_assets, lower=True)

    def simulate_single_path(self, return_paths=False):
        if (self._L_assets is None and self._L_gbm_rate is None and self._L_heston is None):
            self._prepare_cholesky()
        N = self.N
        steps = self.steps
        dt = self.dt
        T = self.T
        use_heston = self.use_heston
        use_rate = self.use_stochastic_rate
        V = np.zeros((steps, N))
        lambda_t = np.zeros((steps, N))
        default_state = np.zeros((steps, N), dtype=bool)
        r_path = np.zeros(steps) if use_rate else None
        v_path = np.zeros((steps, N)) if use_heston else None
        V[0, :] = self.V0
        lambda_t[0, :] = self.base_intensity
        if use_rate:
            r_path[0] = self.r0
        if use_heston:
            v_path[0, :] = self.v0

        if self.regime_switching:
            regime = np.zeros(N, dtype=int)

        if use_heston:
            Z = np.random.standard_normal((steps - 1, self._L_heston.shape[0]))
            W = Z @ self._L_heston.T
            dW_asset = W[:, :N] * np.sqrt(dt)
            dW_vol = W[:, N:2*N] * np.sqrt(dt)
            dW_rate = W[:, -1:] * np.sqrt(dt)
        else:
            if use_rate:
                Z = np.random.standard_normal((steps - 1, self._L_gbm_rate.shape[0]))
                W = Z @ self._L_gbm_rate.T
                dW_asset = W[:, :N] * np.sqrt(dt)
                dW_rate = W[:, -1:] * np.sqrt(dt)
                dW_vol = None
            else:
                Z = np.random.standard_normal((steps - 1, N))
                W = Z @ self._L_assets.T
                dW_asset = W * np.sqrt(dt)
                dW_rate = None
                dW_vol = None
        has_defaulted = np.zeros(N, dtype=bool)
        if use_heston:
            v_curr = self.v0.copy()
        else:
            v_curr = self.vol**2

        for t in range(1, steps):
            if self.regime_switching:
                for i in range(N):
                    if regime[i] == 0:
                        if np.random.rand() < self.regime_transition[0, 1]:
                            regime[i] = 1
                    else:
                        if np.random.rand() < self.regime_transition[1, 0]:
                            regime[i] = 0
                current_jump_intensity = np.array([self.regime_jump_intensity[r] for r in regime])
                current_recovery_base = np.array([self.regime_recovery_base[r] for r in regime])
            else:
                current_jump_intensity = self.jump_intensity
                current_recovery_base = self.recovery_base

            if use_rate:
                r_prev = r_path[t-1]
                dr = self.r_speed * (self.r_mean - r_prev) * dt + self.r_vol * dW_rate[t-1, 0]
                r_curr = r_prev + dr
                r_path[t] = r_curr
            else:
                r_curr = 0.0
            if use_heston:
                v_prev = v_curr
                dv = self.kappa * (self.theta - v_prev) * dt + self.xi * np.sqrt(v_prev) * dW_vol[t-1, :]
                v_curr = v_prev + dv
                v_curr = np.maximum(v_curr, 0.0)
                v_path[t, :] = v_curr
            if use_heston:
                drift_asset = (self.drift - 0.5 * v_curr) * dt
                vol_asset = np.sqrt(v_curr) * dW_asset[t-1, :]
            else:
                drift_asset = (self.drift - 0.5 * self.vol**2) * dt
                vol_asset = self.vol * dW_asset[t-1, :]
            V[t, :] = V[t-1, :] * np.exp(drift_asset + vol_asset)

            if self.regime_switching:
                n_jumps = np.random.poisson(current_jump_intensity * dt, N)
            else:
                if self.jump_intensity > 0:
                    n_jumps = np.random.poisson(self.jump_intensity * dt, N)
                else:
                    n_jumps = 0
            if np.any(n_jumps > 0):
                jump_sizes = np.random.normal(self.jump_mean, self.jump_std, N)
                V[t, :] *= np.exp(jump_sizes * n_jumps)
                V[t, :] = np.maximum(V[t, :], 0.0)

            V[t, has_defaulted] = 0.0
            if self.barrier_target is not None:
                D_curr = self.D0 * np.exp(-self.barrier_mean_reversion * t * dt) + \
                         self.barrier_target * (1 - np.exp(-self.barrier_mean_reversion * t * dt))
            else:
                D_curr = self.D0 * np.exp(self.barrier_growth_rate * t * dt)
            if use_rate:
                D_curr = D_curr * np.exp(-r_curr * (T - t * dt))

            if t > 1:
                prev_new = default_state[t-1, :] & (~default_state[t-2, :])
            else:
                prev_new = np.zeros(N, dtype=bool)
            decay = self.beta * (self.base_intensity - lambda_t[t-1, :]) * dt
            jump_contagion = prev_new.astype(float) @ self.gamma
            lambda_curr = lambda_t[t-1, :] + decay + jump_contagion
            lambda_curr = np.maximum(lambda_curr, 0.0)
            lambda_t[t, :] = lambda_curr

            merton_default = (V[t, :] < D_curr) & (~has_defaulted)
            hazard = 1.0 - np.exp(-lambda_curr * dt)
            poisson_trigger = np.random.uniform(0, 1, N) < hazard
            poisson_default = poisson_trigger & (~has_defaulted) & (~merton_default)
            new_defaults = merton_default | poisson_default
            has_defaulted = has_defaulted | new_defaults
            default_state[t, :] = has_defaulted
            lambda_t[t, has_defaulted] = np.nan

        if return_paths:
            self.V = V
            self.lambda_t = lambda_t
            self.default_state = default_state
            self.r_path = r_path
            self.v_path = v_path
            return V, lambda_t, default_state, r_path, v_path
        else:
            if self.exposures is not None:
                current_lambda = lambda_t[-1, :]
                current_lambda = np.nan_to_num(current_lambda, nan=0.0)
                if self.regime_switching:
                    recovery_rates = current_recovery_base + self.recovery_sensitivity * current_lambda
                else:
                    recovery_rates = self.recovery_base + self.recovery_sensitivity * current_lambda
                recovery_rates = np.clip(recovery_rates, 0.0, 1.0)
                loss = np.sum(default_state[-1, :] * self.exposures * (1 - recovery_rates))
                return loss
            else:
                return 0.0
