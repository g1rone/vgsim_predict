import itertools
import numpy as np
import pandas as pd


SUMMARY_COLS = ["mig_AB", "mig_BA", "final_A", "final_B", "t_end"]


def distance(sim, obs):
    dist = 0.0

    for col in SUMMARY_COLS:
        dist += ((sim[col] - obs[col]) / (abs(obs[col]) + 1.0)) ** 2

    return float(np.sqrt(dist))


def _unique_sorted(values, decimals=12):
    return np.array(sorted(set(round(float(v), decimals) for v in values)), dtype=float)


def _mean_simulation_result(simulations):
    result = {}

    for col in SUMMARY_COLS:
        result[col] = float(np.mean([sim[col] for sim in simulations]))

    return result


def fit_equal_mu_refined(
    obs,
    mu_grid,
    simulate_func,
    top_k=3,
    refine_steps=4,
    refine_points=15,
    shrink=0.35,
    seeds=(0, 1, 2),
):
    """
    Подбор равного mu через грубую сетку и последовательное уточнение.

    Идея:
    1. Сначала проверяем грубую сетку.
    2. Берём top_k лучших значений.
    3. Вокруг них строим более мелкие сетки.
    4. Повторяем refine_steps раз.
    """

    mu_grid = _unique_sorted(mu_grid)

    if len(mu_grid) < 2:
        raise ValueError("mu_grid must contain at least two values")

    if top_k < 1:
        raise ValueError("top_k must be positive")

    if refine_steps < 0:
        raise ValueError("refine_steps must be non-negative")

    if refine_points < 2:
        raise ValueError("refine_points must be at least 2")

    mu_min = float(mu_grid.min())
    mu_max = float(mu_grid.max())

    initial_step = float(np.median(np.diff(mu_grid)))
    radius = initial_step

    current_grid = mu_grid
    all_rows = []

    for stage in range(refine_steps + 1):
        rows = []

        for mu in current_grid:
            simulations = []
            distances = []

            for seed in seeds:
                sim = simulate_func(mu, mu, seed=seed)
                simulations.append(sim)
                distances.append(distance(sim, obs))

            mean_sim = _mean_simulation_result(simulations)
            mean_distance = float(np.mean(distances))

            rows.append({
                "stage": stage,
                "mu_AB": float(mu),
                "mu_BA": float(mu),
                "distance": mean_distance,
                **mean_sim,
            })

        stage_df = pd.DataFrame(rows).sort_values("distance").reset_index(drop=True)
        all_rows.append(stage_df)

        best_mus = stage_df.head(top_k)["mu_AB"].to_numpy()

        radius *= shrink

        next_grid_values = []

        for best_mu in best_mus:
            left = max(mu_min, best_mu - radius)
            right = min(mu_max, best_mu + radius)

            next_grid_values.extend(np.linspace(left, right, refine_points))

        current_grid = _unique_sorted(next_grid_values)

    result = pd.concat(all_rows, ignore_index=True)
    result = result.sort_values("distance").reset_index(drop=True)

    return result


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