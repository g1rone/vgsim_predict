import itertools
import numpy as np
import pandas as pd


SUMMARY_COLS = ["mig_AB", "mig_BA", "final_A", "final_B", "t_end"]


def distance(sim, obs):
    dist = 0.0

    for col in SUMMARY_COLS:
        dist += ((sim[col] - obs[col]) / (abs(obs[col]) + 1.0)) ** 2

    return float(np.sqrt(dist))


def fit_equal_mu(obs, mu_grid, simulate_func):
    rows = []

    for mu in mu_grid:
        sim = simulate_func(mu, mu, seed=0)
        d = distance(sim, obs)

        rows.append({
            "mu_AB": mu,
            "mu_BA": mu,
            "distance": d,
            **sim,
        })

    return pd.DataFrame(rows).sort_values("distance").reset_index(drop=True)


def fit_unequal_mu(obs, mu_grid, simulate_func):
    rows = []

    for mu_ab, mu_ba in itertools.product(mu_grid, mu_grid):
        sim = simulate_func(mu_ab, mu_ba, seed=0)
        d = distance(sim, obs)

        rows.append({
            "mu_AB": mu_ab,
            "mu_BA": mu_ba,
            "distance": d,
            **sim,
        })

    return pd.DataFrame(rows).sort_values("distance").reset_index(drop=True)