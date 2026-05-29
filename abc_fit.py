import numpy as np
import pandas as pd


DEFAULT_TRAJECTORY_POINTS = 100

AGGREGATE_COLS = [
    "final_A",
    "final_B",
    "peak_A",
    "peak_B",
    "time_peak_A_frac",
    "time_peak_B_frac",
    "first_active_B_frac",
    "area_active_A",
    "area_active_B",
    "mean_active_A",
    "mean_active_B",
    "cum_infections_A",
    "cum_infections_B",
    "cum_recoveries_A",
    "cum_recoveries_B",
]

TIME_COLS = [
    "t_end",
]

TRAJECTORY_SERIES = [
    "active_A",
    "active_B",
    "infect_A",
    "infect_B",
    "recover_A",
    "recover_B",
]

MIGRATION_COLS = [
    "mig_AB",
    "mig_BA",
    "mig_total",
    "mig_balance_abs",
    "first_mig_AB_frac",
    "first_mig_BA_frac",
]

DIAGNOSTIC_COLS = MIGRATION_COLS + [
    "grid_reference_t_end",
]

DISPLAY_FIRST_COLS = [
    "stage",
    "mu",
    "mu_AB_candidate",
    "mu_BA_candidate",
    "distance",
    "trajectory_distance",
    "aggregate_distance",
    "time_distance",
    "time_extra_distance",
    "migration_distance",
    "grid_size",
    "search_radius",
    "n_seeds",
]


def make_trajectory_cols(trajectory_points=DEFAULT_TRAJECTORY_POINTS):
    trajectory_points = int(trajectory_points)

    if trajectory_points < 2:
        raise ValueError("trajectory_points must be at least 2")

    cols = []

    for index in range(trajectory_points):
        for series_name in TRAJECTORY_SERIES:
            cols.append(f"grid_{index:03d}_{series_name}")

    return cols


def make_summary_cols(trajectory_points=DEFAULT_TRAJECTORY_POINTS):
    return (
        AGGREGATE_COLS
        + TIME_COLS
        + MIGRATION_COLS
        + make_trajectory_cols(trajectory_points)
    )


SUMMARY_COLS = make_summary_cols(DEFAULT_TRAJECTORY_POINTS)


def _safe_float(value):
    return float(value)


def _normalized_scalar_error(sim_value, obs_value):
    sim_value = _safe_float(sim_value)
    obs_value = _safe_float(obs_value)
    return (sim_value - obs_value) / (abs(obs_value) + 1.0)


def _time_ratio_error(sim_value, obs_value):
    sim_value = max(_safe_float(sim_value), 0.0)
    obs_value = max(_safe_float(obs_value), 0.0)
    return abs(np.log((sim_value + 1.0) / (obs_value + 1.0)))


def _series_scale(obs_values):
    obs_values = np.asarray(obs_values, dtype=float)

    if obs_values.size == 0:
        return 1.0

    p90 = float(np.percentile(np.abs(obs_values), 90))
    rms = float(np.sqrt(np.mean(obs_values ** 2)))
    mean_abs = float(np.mean(np.abs(obs_values)))

    return max(p90, rms, mean_abs, 1.0)


def _series_distance(sim, obs, cols):
    sim_values = np.array([_safe_float(sim[col]) for col in cols], dtype=float)
    obs_values = np.array([_safe_float(obs[col]) for col in cols], dtype=float)
    scale = _series_scale(obs_values)
    return float(np.sqrt(np.mean(((sim_values - obs_values) / scale) ** 2)))


def _weighted_average_squared(items):
    total_weight = 0.0
    total_value = 0.0

    for value, weight in items:
        weight = float(weight)

        if weight <= 0:
            continue

        total_weight += weight
        total_value += weight * float(value) ** 2

    if total_weight <= 0:
        return 0.0

    return float(np.sqrt(total_value / total_weight))


def trajectory_distance(sim, obs, trajectory_points=DEFAULT_TRAJECTORY_POINTS):
    items = []

    for series_name in TRAJECTORY_SERIES:
        cols = [
            f"grid_{index:03d}_{series_name}"
            for index in range(int(trajectory_points))
        ]

        if series_name.startswith("active"):
            weight = 2.0
        elif series_name.startswith("infect"):
            weight = 1.5
        elif series_name.startswith("recover"):
            weight = 1.0
        else:
            weight = 1.0

        items.append((_series_distance(sim, obs, cols), weight))

    return _weighted_average_squared(items)


def aggregate_distance(sim, obs):
    weights = {
        "final_A": 1.5,
        "final_B": 1.5,
        "peak_A": 2.5,
        "peak_B": 2.5,
        "time_peak_A_frac": 1.0,
        "time_peak_B_frac": 1.0,
        "first_active_B_frac": 2.0,
        "area_active_A": 2.5,
        "area_active_B": 2.5,
        "mean_active_A": 1.5,
        "mean_active_B": 1.5,
        "cum_infections_A": 2.0,
        "cum_infections_B": 2.0,
        "cum_recoveries_A": 1.5,
        "cum_recoveries_B": 1.5,
    }

    items = []

    for col in AGGREGATE_COLS:
        error = _normalized_scalar_error(sim[col], obs[col])
        items.append((error, weights.get(col, 1.0)))

    return _weighted_average_squared(items)


def time_distance(sim, obs):
    return _time_ratio_error(sim["t_end"], obs["t_end"])


def time_extra_distance(sim, obs, time_tolerance=0.10):
    tolerance = max(float(time_tolerance), 0.0)
    base = time_distance(sim, obs)
    allowed = np.log(1.0 + tolerance)
    return float(max(0.0, base - allowed))


def migration_distance(sim, obs):
    weights = {
        "mig_AB": 1.0,
        "mig_BA": 1.0,
        "mig_total": 3.0,
        "mig_balance_abs": 1.0,
        "first_mig_AB_frac": 0.5,
        "first_mig_BA_frac": 0.5,
    }

    items = []

    for col in MIGRATION_COLS:
        error = _normalized_scalar_error(sim[col], obs[col])
        items.append((error, weights.get(col, 1.0)))

    return _weighted_average_squared(items)


def distance_components(
    sim,
    obs,
    trajectory_points=DEFAULT_TRAJECTORY_POINTS,
    trajectory_weight=1.0,
    aggregate_weight=0.75,
    time_weight=1.0,
    time_extra_weight=3.0,
    time_tolerance=0.10,
    migration_weight=0.0,
):
    tr = trajectory_distance(sim, obs, trajectory_points)
    ag = aggregate_distance(sim, obs)
    tm = time_distance(sim, obs)
    te = time_extra_distance(sim, obs, time_tolerance)
    mg = migration_distance(sim, obs)

    items = [
        (tr, trajectory_weight),
        (ag, aggregate_weight),
        (tm, time_weight),
        (te, time_extra_weight),
        (mg, migration_weight),
    ]

    total = _weighted_average_squared(items)

    return {
        "distance": float(total),
        "trajectory_distance": float(tr),
        "aggregate_distance": float(ag),
        "time_distance": float(tm),
        "time_extra_distance": float(te),
        "migration_distance": float(mg),
    }


def distance(
    sim,
    obs,
    trajectory_points=DEFAULT_TRAJECTORY_POINTS,
    trajectory_weight=1.0,
    aggregate_weight=0.75,
    time_weight=1.0,
    time_extra_weight=3.0,
    time_tolerance=0.10,
    migration_weight=0.0,
):
    return distance_components(
        sim=sim,
        obs=obs,
        trajectory_points=trajectory_points,
        trajectory_weight=trajectory_weight,
        aggregate_weight=aggregate_weight,
        time_weight=time_weight,
        time_extra_weight=time_extra_weight,
        time_tolerance=time_tolerance,
        migration_weight=migration_weight,
    )["distance"]


def _unique_sorted(values, decimals=12):
    return np.array(
        sorted(set(round(float(v), decimals) for v in values)),
        dtype=float,
    )


def _mean_simulation_result(simulations):
    if len(simulations) == 0:
        raise ValueError("simulations must contain at least one element")

    result = {}
    all_cols = list(simulations[0].keys())

    for col in all_cols:
        values = [float(sim[col]) for sim in simulations]
        result[col] = float(np.mean(values))

    return result


def _parse_seeds(seeds):
    if isinstance(seeds, str):
        parts = [part.strip() for part in seeds.split(",")]
        return tuple(int(part) for part in parts if part != "")

    return tuple(int(seed) for seed in seeds)


def fit_equal_mu_refined(
    obs,
    mu_grid,
    simulate_func,
    top_k=2,
    refine_steps=2,
    refine_points=7,
    shrink=0.35,
    seeds=(1, 2, 3),
    trajectory_points=DEFAULT_TRAJECTORY_POINTS,
    trajectory_weight=1.0,
    aggregate_weight=0.75,
    time_weight=1.0,
    time_extra_weight=3.0,
    time_tolerance=0.10,
    migration_weight=0.0,
    progress_callback=None,
):
    mu_grid = _unique_sorted(mu_grid)
    seeds = _parse_seeds(seeds)
    trajectory_points = int(trajectory_points)

    if len(mu_grid) < 2:
        raise ValueError("mu_grid must contain at least two values")

    if top_k < 1:
        raise ValueError("top_k must be positive")

    if refine_steps < 0:
        raise ValueError("refine_steps must be non-negative")

    if refine_points < 2:
        raise ValueError("refine_points must be at least 2")

    if not 0 < shrink < 1:
        raise ValueError("shrink must be between 0 and 1")

    if len(seeds) == 0:
        raise ValueError("seeds must contain at least one seed")

    if trajectory_points < 2:
        raise ValueError("trajectory_points must be at least 2")

    mu_min = float(mu_grid.min())
    mu_max = float(mu_grid.max())

    initial_step = float(np.median(np.diff(mu_grid)))
    radius = initial_step

    current_grid = mu_grid
    all_rows = []

    for stage in range(refine_steps + 1):
        rows = []
        grid_size = len(current_grid)

        for index, mu in enumerate(current_grid, start=1):
            if progress_callback is not None:
                progress_callback(stage, refine_steps, index, grid_size, float(mu))

            simulations = []

            for seed in seeds:
                sim = simulate_func(float(mu), float(mu), seed=int(seed))
                simulations.append(sim)

            mean_sim = _mean_simulation_result(simulations)
            dist_info = distance_components(
                sim=mean_sim,
                obs=obs,
                trajectory_points=trajectory_points,
                trajectory_weight=trajectory_weight,
                aggregate_weight=aggregate_weight,
                time_weight=time_weight,
                time_extra_weight=time_extra_weight,
                time_tolerance=time_tolerance,
                migration_weight=migration_weight,
            )

            rows.append({
                "stage": int(stage),
                "mu": float(mu),
                "mu_AB_candidate": float(mu),
                "mu_BA_candidate": float(mu),
                "distance": dist_info["distance"],
                "trajectory_distance": dist_info["trajectory_distance"],
                "aggregate_distance": dist_info["aggregate_distance"],
                "time_distance": dist_info["time_distance"],
                "time_extra_distance": dist_info["time_extra_distance"],
                "migration_distance": dist_info["migration_distance"],
                "grid_size": int(grid_size),
                "search_radius": float(radius),
                "n_seeds": int(len(seeds)),
                **mean_sim,
            })

        stage_df = (
            pd.DataFrame(rows)
            .sort_values("distance")
            .reset_index(drop=True)
        )

        all_rows.append(stage_df)
        best_mus = stage_df.head(top_k)["mu"].to_numpy()

        radius *= shrink
        next_grid_values = []

        for best_mu in best_mus:
            left = max(mu_min, float(best_mu) - radius)
            right = min(mu_max, float(best_mu) + radius)
            next_grid_values.extend(np.linspace(left, right, refine_points))

        current_grid = _unique_sorted(next_grid_values)

    result = pd.concat(all_rows, ignore_index=True)
    result = result.sort_values("distance").reset_index(drop=True)

    return result


def _unique_sorted_pairs(pairs, decimals=12):
    return np.array(
        sorted(
            set(
                (
                    round(float(mu_ab), decimals),
                    round(float(mu_ba), decimals),
                )
                for mu_ab, mu_ba in pairs
            )
        ),
        dtype=float,
    )


def _make_pair_grid(mu_ab_grid, mu_ba_grid):
    pairs = []

    for mu_ab in mu_ab_grid:
        for mu_ba in mu_ba_grid:
            pairs.append((float(mu_ab), float(mu_ba)))

    return _unique_sorted_pairs(pairs)


def fit_unequal_mu_refined(
    obs,
    mu_ab_grid,
    mu_ba_grid,
    simulate_func,
    top_k=3,
    refine_steps=2,
    refine_points=5,
    shrink=0.35,
    seeds=(1, 2, 3),
    trajectory_points=DEFAULT_TRAJECTORY_POINTS,
    trajectory_weight=1.0,
    aggregate_weight=0.75,
    time_weight=1.0,
    time_extra_weight=3.0,
    time_tolerance=0.10,
    migration_weight=0.0,
    progress_callback=None,
):
    mu_ab_grid = _unique_sorted(mu_ab_grid)
    mu_ba_grid = _unique_sorted(mu_ba_grid)
    seeds = _parse_seeds(seeds)
    trajectory_points = int(trajectory_points)

    if len(mu_ab_grid) < 2:
        raise ValueError("mu_ab_grid must contain at least two values")

    if len(mu_ba_grid) < 2:
        raise ValueError("mu_ba_grid must contain at least two values")

    if top_k < 1:
        raise ValueError("top_k must be positive")

    if refine_steps < 0:
        raise ValueError("refine_steps must be non-negative")

    if refine_points < 2:
        raise ValueError("refine_points must be at least 2")

    if not 0 < shrink < 1:
        raise ValueError("shrink must be between 0 and 1")

    if len(seeds) == 0:
        raise ValueError("seeds must contain at least one seed")

    if trajectory_points < 2:
        raise ValueError("trajectory_points must be at least 2")

    mu_ab_min = float(mu_ab_grid.min())
    mu_ab_max = float(mu_ab_grid.max())
    mu_ba_min = float(mu_ba_grid.min())
    mu_ba_max = float(mu_ba_grid.max())

    radius_ab = float(np.median(np.diff(mu_ab_grid)))
    radius_ba = float(np.median(np.diff(mu_ba_grid)))

    current_pairs = _make_pair_grid(mu_ab_grid, mu_ba_grid)
    all_rows = []

    for stage in range(refine_steps + 1):
        rows = []
        grid_size = len(current_pairs)

        for index, pair in enumerate(current_pairs, start=1):
            mu_ab = float(pair[0])
            mu_ba = float(pair[1])

            if progress_callback is not None:
                progress_callback(stage, refine_steps, index, grid_size, mu_ab, mu_ba)

            simulations = []

            for seed in seeds:
                sim = simulate_func(mu_ab, mu_ba, seed=int(seed))
                simulations.append(sim)

            mean_sim = _mean_simulation_result(simulations)
            dist_info = distance_components(
                sim=mean_sim,
                obs=obs,
                trajectory_points=trajectory_points,
                trajectory_weight=trajectory_weight,
                aggregate_weight=aggregate_weight,
                time_weight=time_weight,
                time_extra_weight=time_extra_weight,
                time_tolerance=time_tolerance,
                migration_weight=migration_weight,
            )

            rows.append({
                "stage": int(stage),
                "mu": float((mu_ab + mu_ba) / 2.0),
                "mu_AB_candidate": mu_ab,
                "mu_BA_candidate": mu_ba,
                "distance": dist_info["distance"],
                "trajectory_distance": dist_info["trajectory_distance"],
                "aggregate_distance": dist_info["aggregate_distance"],
                "time_distance": dist_info["time_distance"],
                "time_extra_distance": dist_info["time_extra_distance"],
                "migration_distance": dist_info["migration_distance"],
                "grid_size": int(grid_size),
                "search_radius": float(max(radius_ab, radius_ba)),
                "search_radius_AB": float(radius_ab),
                "search_radius_BA": float(radius_ba),
                "n_seeds": int(len(seeds)),
                **mean_sim,
            })

        stage_df = (
            pd.DataFrame(rows)
            .sort_values("distance")
            .reset_index(drop=True)
        )

        all_rows.append(stage_df)
        best_pairs = stage_df.head(top_k)[["mu_AB_candidate", "mu_BA_candidate"]].to_numpy()

        radius_ab *= shrink
        radius_ba *= shrink
        next_pairs = []

        for best_mu_ab, best_mu_ba in best_pairs:
            left_ab = max(mu_ab_min, float(best_mu_ab) - radius_ab)
            right_ab = min(mu_ab_max, float(best_mu_ab) + radius_ab)
            left_ba = max(mu_ba_min, float(best_mu_ba) - radius_ba)
            right_ba = min(mu_ba_max, float(best_mu_ba) + radius_ba)

            ab_values = np.linspace(left_ab, right_ab, refine_points)
            ba_values = np.linspace(left_ba, right_ba, refine_points)

            for mu_ab in ab_values:
                for mu_ba in ba_values:
                    next_pairs.append((mu_ab, mu_ba))

        current_pairs = _unique_sorted_pairs(next_pairs)

    result = pd.concat(all_rows, ignore_index=True)
    result = result.sort_values("distance").reset_index(drop=True)

    return result