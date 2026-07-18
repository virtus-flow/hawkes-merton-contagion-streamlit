# src/utils.py
import numpy as np
from tqdm import tqdm

def run_monte_carlo_sequential(model, n_simulations, exposures, alpha=0.01, show_progress=True):
    model.exposures = exposures
    model._prepare_cholesky()
    losses = np.zeros(n_simulations)
    for i in tqdm(range(n_simulations), desc="Monte Carlo", disable=not show_progress):
        losses[i] = model.simulate_single_path(return_paths=False)
    sorted_losses = np.sort(losses)
    var_idx = int(np.ceil((1 - alpha) * n_simulations)) - 1
    VaR = sorted_losses[var_idx]
    CVaR = np.mean(sorted_losses[var_idx:])
    return losses, VaR, CVaR
