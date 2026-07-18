# src/calibration.py
import numpy as np
from scipy.optimize import minimize
from .model import HawkesMertonContagion

def compute_implicit_spread(params, model_base, exposures, n_sims=2000, alpha=0.01):
    np.random.seed(42)
    jump_intensity, gamma_multiplier, recovery_base, recovery_sensitivity = params
    N = model_base.N
    corr = model_base.corr_assets
    model_new = HawkesMertonContagion(
        n_companies=N,
        T=1.0,
        dt=1/252,
        use_heston=False,
        use_stochastic_rate=False,
        barrier_growth_rate=0.02,
        barrier_target=None,
        jump_intensity=jump_intensity,
        jump_mean=-0.15,
        jump_std=0.10,
        recovery_base=recovery_base,
        recovery_sensitivity=recovery_sensitivity
    )
    model_new.V0 = model_base.V0.copy()
    model_new.vol = model_base.vol.copy()
    model_new.D0 = model_base.D0.copy()
    model_new.corr_assets = corr.copy()
    model_new._prepare_cholesky()
    gamma = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                gamma[i, j] = 0.05 * gamma_multiplier * max(0, corr[i, j])
    model_new.set_contagion_network(gamma)
    losses, VaR, CVaR = run_monte_carlo_sequential(
        model_new, n_sims, exposures, alpha=alpha, show_progress=False
    )
    default_prob = np.mean(losses > 0) * 100
    spread_bps = -np.log(1 - default_prob/100) * 10000
    return spread_bps
