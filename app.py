from pathlib import Path
import shutil
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import VGsim

from abc_fit import (
    AGGREGATE_COLS,
    DIAGNOSTIC_COLS,
    DISPLAY_FIRST_COLS,
    distance_components,
    fit_equal_mu_refined,
    fit_unequal_mu_refined,
    fit_unequal_mu_fast_refined,
)


POP_A = 0
POP_B = 1
POPULATIONS_NUMBER = 2
SUSCEPTIBLE_GROUPS_NUMBER = 1

BIRTH = 0
DEATH = 1
SAMPLING = 2
MUTATION = 3
SUSCCHANGE = 4
MIGRATION = 5

RUNS_DIR = Path("vgsim_runs")
PLOTS_DIR = Path("vgsim_plots")

DEFAULT_VGSIM_PARAMS = {
    "number_of_sites": 1,
    "population_a_size": 200_000,
    "population_b_size": 100_000,
    "transmission_rate": 0.15,
    "recovery_rate": 0.10,
    "sampling_rate": 0.01,
    "iterations": 1_000_000,
    "sample_size": 500,
    "method": "direct",
}

TRAJECTORY_SERIES = [
    "active_A",
    "active_B",
    "infect_A",
    "infect_B",
    "recover_A",
    "recover_B",
]


state = {
    "clean_observed_data": None,
    "clean_observed_timeline": None,
    "observed_data": None,
    "observed_timeline": None,
    "observation_grid_signature": None,
    "observation_vgsim_signature": None,
    "observation_vgsim_params": None,
    "last_observation_df": None,
    "last_fit_df": None,
    "last_posterior_df": None,
    "best_fit_mu": None,
    "best_fit_seed": None,
}


def reset_run_dirs():
    shutil.rmtree(RUNS_DIR, ignore_errors=True)
    RUNS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)


def update_status(text):
    status_label.config(text=text)
    root.update_idletasks()


def fail(title, text):
    messagebox.showerror(title, text)


def warn(title, text):
    messagebox.showwarning(title, text)


def clear_fit_outputs():
    state["last_fit_df"] = None
    state["last_posterior_df"] = None
    state["best_fit_mu"] = None
    state["best_fit_seed"] = None


def _vgsim_signature(params):
    keys = (
        "number_of_sites",
        "population_a_size",
        "population_b_size",
        "transmission_rate",
        "recovery_rate",
        "sampling_rate",
        "iterations",
        "sample_size",
        "method",
    )
    return tuple((key, params[key]) for key in keys)


def _as_int(entry):
    return int(entry.get().strip())


def _as_float(entry):
    return float(entry.get().strip())


def _make_entry(frame, row, col, label, default, width=12):
    ttk.Label(frame, text=label).grid(row=row, column=col, padx=5, pady=4, sticky="w")
    entry = ttk.Entry(frame, width=width)
    entry.insert(0, str(default))
    entry.grid(row=row, column=col + 1, padx=5, pady=4, sticky="w")
    return entry


def _safe_fraction(time_value, t_end, default=1.0):
    if time_value is None or t_end <= 0:
        return float(default)
    return float(time_value / t_end)


def _make_fractions(trajectory_points, grid_start_frac, grid_end_frac):
    trajectory_points = int(trajectory_points)
    grid_start_frac = float(grid_start_frac)
    grid_end_frac = float(grid_end_frac)

    if trajectory_points < 2:
        raise ValueError("trajectory_points must be at least 2")
    if not 0.0 <= grid_start_frac < grid_end_frac <= 1.0:
        raise ValueError("grid_start_frac and grid_end_frac must satisfy 0 <= start < end <= 1")

    return np.linspace(grid_start_frac, grid_end_frac, trajectory_points)


def _value_at_time(times, values, target_time):
    if len(times) == 0:
        return 0.0

    index = int(np.searchsorted(times, float(target_time), side="right") - 1)
    index = max(0, min(index, len(values) - 1))
    return float(values[index])


def _weighted_quantile(values, quantiles):
    values = np.asarray(values, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)

    if values.size == 0:
        return np.full_like(quantiles, np.nan, dtype=float)

    values = np.sort(values)
    positions = np.linspace(0.0, 1.0, len(values))
    return np.interp(quantiles, positions, values)


def _prepare_run_dir(mu_ab, mu_ba, seed):
    RUNS_DIR.mkdir(exist_ok=True)

    run_dir = RUNS_DIR / f"muab_{mu_ab:.8f}_muba_{mu_ba:.8f}_seed_{seed}"

    if run_dir.exists():
        shutil.rmtree(run_dir)

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _create_simulator(mu_ab, mu_ba, seed, params):
    simulator = VGsim.Simulator(
        int(params["number_of_sites"]),
        POPULATIONS_NUMBER,
        SUSCEPTIBLE_GROUPS_NUMBER,
        seed=int(seed),
    )

    simulator.set_population_size(int(params["population_a_size"]), population=POP_A)
    simulator.set_population_size(int(params["population_b_size"]), population=POP_B)

    simulator.set_transmission_rate(float(params["transmission_rate"]))
    simulator.set_recovery_rate(float(params["recovery_rate"]))
    simulator.set_sampling_rate(float(params["sampling_rate"]))

    simulator.set_migration_probability(probability=float(mu_ab), source=POP_A, target=POP_B)
    simulator.set_migration_probability(probability=float(mu_ba), source=POP_B, target=POP_A)

    return simulator


def _run_vgsim(simulator, params):
    try:
        simulator.simulate(
            int(params["iterations"]),
            sample_size=int(params["sample_size"]),
            method=str(params["method"]),
        )
    except TypeError:
        simulator.simulate(int(params["iterations"]), int(params["sample_size"]))


def _load_chain_events(simulator, run_dir):
    base_path = run_dir / "chain_events"
    simulator.export_chain_events(str(base_path))

    npy_path = base_path.with_suffix(".npy")
    if not npy_path.exists():
        raise FileNotFoundError(f"Не найден файл chain events: {npy_path}")

    data = np.load(npy_path, allow_pickle=True)

    times = np.asarray(data[0], dtype=float)
    event_types = np.asarray(data[1], dtype=int)
    haplotypes = np.asarray(data[2], dtype=int)
    populations = np.asarray(data[3], dtype=int)
    new_haplotypes = np.asarray(data[4], dtype=int)
    new_populations = np.asarray(data[5], dtype=int)

    valid_indices = np.where(times > 0)[0]

    if len(valid_indices) == 0:
        empty_float = np.array([], dtype=float)
        empty_int = np.array([], dtype=int)
        return {
            "time": empty_float,
            "type": empty_int,
            "haplotype": empty_int,
            "population": empty_int,
            "new_haplotype": empty_int,
            "new_population": empty_int,
        }

    last_index = valid_indices[-1] + 1
    order = np.argsort(times[:last_index])

    return {
        "time": times[:last_index][order],
        "type": event_types[:last_index][order],
        "haplotype": haplotypes[:last_index][order],
        "population": populations[:last_index][order],
        "new_haplotype": new_haplotypes[:last_index][order],
        "new_population": new_populations[:last_index][order],
    }


def _build_summary_from_chain_events(
    chain_events,
    trajectory_points,
    grid_start_frac,
    grid_end_frac,
    haplotypes_number,
    reference_t_end=None,
    return_timeline=False,
):
    times = chain_events["time"]
    event_types = chain_events["type"]
    haplotypes = chain_events["haplotype"]
    populations = chain_events["population"]
    new_haplotypes = chain_events["new_haplotype"]
    new_populations = chain_events["new_population"]

    active_by_pop_hap = np.zeros((POPULATIONS_NUMBER, int(haplotypes_number)), dtype=int)
    active_by_pop_hap[POP_A, 0] = 1

    cumulative_infections = np.zeros(POPULATIONS_NUMBER, dtype=float)
    cumulative_recoveries = np.zeros(POPULATIONS_NUMBER, dtype=float)

    timeline_time = [0.0]
    timeline_active = [active_by_pop_hap.sum(axis=1).astype(float)]
    timeline_infections = [cumulative_infections.copy()]
    timeline_recoveries = [cumulative_recoveries.copy()]

    peak_active = active_by_pop_hap.sum(axis=1).astype(float)
    time_peak = np.zeros(POPULATIONS_NUMBER, dtype=float)
    area_active = np.zeros(POPULATIONS_NUMBER, dtype=float)

    first_active_B_time = None
    first_mig_AB_time = None
    first_mig_BA_time = None

    mig_ab = 0
    mig_ba = 0
    previous_time = 0.0

    for t, event_type, hap, pop, new_hap, new_pop in zip(
        times,
        event_types,
        haplotypes,
        populations,
        new_haplotypes,
        new_populations,
    ):
        t = float(t)
        area_active += active_by_pop_hap.sum(axis=1).astype(float) * max(0.0, t - previous_time)
        previous_time = t

        if event_type == BIRTH:
            if 0 <= pop < POPULATIONS_NUMBER and 0 <= hap < haplotypes_number:
                active_by_pop_hap[pop, hap] += 1
                cumulative_infections[pop] += 1

        elif event_type in (DEATH, SAMPLING):
            if 0 <= pop < POPULATIONS_NUMBER and 0 <= hap < haplotypes_number:
                active_by_pop_hap[pop, hap] -= 1
                if event_type == DEATH:
                    cumulative_recoveries[pop] += 1

        elif event_type == MUTATION:
            if 0 <= pop < POPULATIONS_NUMBER:
                if 0 <= hap < haplotypes_number:
                    active_by_pop_hap[pop, hap] -= 1
                if 0 <= new_hap < haplotypes_number:
                    active_by_pop_hap[pop, new_hap] += 1

        elif event_type == MIGRATION:
            if (
                0 <= pop < POPULATIONS_NUMBER
                and 0 <= new_pop < POPULATIONS_NUMBER
                and 0 <= hap < haplotypes_number
            ):
                active_by_pop_hap[pop, hap] -= 1
                active_by_pop_hap[new_pop, hap] += 1

                if pop == POP_A and new_pop == POP_B:
                    mig_ab += 1
                    first_mig_AB_time = t if first_mig_AB_time is None else first_mig_AB_time
                elif pop == POP_B and new_pop == POP_A:
                    mig_ba += 1
                    first_mig_BA_time = t if first_mig_BA_time is None else first_mig_BA_time

        elif event_type == SUSCCHANGE:
            pass

        active_by_pop_hap = np.maximum(active_by_pop_hap, 0)
        current_active = active_by_pop_hap.sum(axis=1).astype(float)

        for pop_id in range(POPULATIONS_NUMBER):
            if current_active[pop_id] > peak_active[pop_id]:
                peak_active[pop_id] = current_active[pop_id]
                time_peak[pop_id] = t

        if first_active_B_time is None and current_active[POP_B] > 0:
            first_active_B_time = t

        timeline_time.append(t)
        timeline_active.append(current_active.copy())
        timeline_infections.append(cumulative_infections.copy())
        timeline_recoveries.append(cumulative_recoveries.copy())

    timeline_time = np.asarray(timeline_time, dtype=float)
    timeline_active = np.asarray(timeline_active, dtype=float)
    timeline_infections = np.asarray(timeline_infections, dtype=float)
    timeline_recoveries = np.asarray(timeline_recoveries, dtype=float)

    final_by_pop = timeline_active[-1]
    t_end = float(timeline_time[-1]) if len(timeline_time) else 0.0
    mean_active = area_active / t_end if t_end > 0 else np.zeros(POPULATIONS_NUMBER, dtype=float)
    grid_reference_t_end = t_end if reference_t_end is None else float(reference_t_end)

    result = {
        "mig_AB": float(mig_ab),
        "mig_BA": float(mig_ba),
        "mig_total": float(mig_ab + mig_ba),
        "mig_balance_abs": float(abs(mig_ab - mig_ba)),
        "first_mig_AB_frac": _safe_fraction(first_mig_AB_time, t_end),
        "first_mig_BA_frac": _safe_fraction(first_mig_BA_time, t_end),
        "grid_reference_t_end": float(grid_reference_t_end),
        "final_A": float(final_by_pop[POP_A]),
        "final_B": float(final_by_pop[POP_B]),
        "peak_A": float(peak_active[POP_A]),
        "peak_B": float(peak_active[POP_B]),
        "time_peak_A_frac": _safe_fraction(time_peak[POP_A], t_end, default=0.0),
        "time_peak_B_frac": _safe_fraction(time_peak[POP_B], t_end, default=0.0),
        "first_active_B_frac": _safe_fraction(first_active_B_time, t_end),
        "area_active_A": float(area_active[POP_A]),
        "area_active_B": float(area_active[POP_B]),
        "mean_active_A": float(mean_active[POP_A]),
        "mean_active_B": float(mean_active[POP_B]),
        "cum_infections_A": float(cumulative_infections[POP_A]),
        "cum_infections_B": float(cumulative_infections[POP_B]),
        "cum_recoveries_A": float(cumulative_recoveries[POP_A]),
        "cum_recoveries_B": float(cumulative_recoveries[POP_B]),
        "t_end": float(t_end),
    }

    grid_fractions = _make_fractions(trajectory_points, grid_start_frac, grid_end_frac)

    for index, frac in enumerate(grid_fractions):
        target_time = float(frac) * float(grid_reference_t_end)

        result[f"grid_{index:03d}_active_A"] = _value_at_time(timeline_time, timeline_active[:, POP_A], target_time)
        result[f"grid_{index:03d}_active_B"] = _value_at_time(timeline_time, timeline_active[:, POP_B], target_time)
        result[f"grid_{index:03d}_infect_A"] = _value_at_time(timeline_time, timeline_infections[:, POP_A], target_time)
        result[f"grid_{index:03d}_infect_B"] = _value_at_time(timeline_time, timeline_infections[:, POP_B], target_time)
        result[f"grid_{index:03d}_recover_A"] = _value_at_time(timeline_time, timeline_recoveries[:, POP_A], target_time)
        result[f"grid_{index:03d}_recover_B"] = _value_at_time(timeline_time, timeline_recoveries[:, POP_B], target_time)

    timeline = {
        "time": timeline_time,
        "active": timeline_active,
        "infections": timeline_infections,
        "recoveries": timeline_recoveries,
    }

    if return_timeline:
        return result, timeline

    return result


def simulate(
    mu_ab,
    mu_ba,
    seed=0,
    trajectory_points=100,
    grid_start_frac=0.0,
    grid_end_frac=1.0,
    reference_t_end=None,
    return_timeline=False,
    vgsim_params=None,
):
    params = DEFAULT_VGSIM_PARAMS.copy() if vgsim_params is None else vgsim_params
    haplotypes_number = 4 ** int(params["number_of_sites"])
    run_dir = _prepare_run_dir(mu_ab, mu_ba, seed)

    try:
        simulator = _create_simulator(mu_ab, mu_ba, seed, params)
        _run_vgsim(simulator, params)
        chain_events = _load_chain_events(simulator, run_dir)

        return _build_summary_from_chain_events(
            chain_events=chain_events,
            trajectory_points=trajectory_points,
            grid_start_frac=grid_start_frac,
            grid_end_frac=grid_end_frac,
            haplotypes_number=haplotypes_number,
            reference_t_end=reference_t_end,
            return_timeline=return_timeline,
        )

    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def get_noise_params():
    try:
        reporting_rate = _as_float(reporting_rate_entry)
        rate_jitter = _as_float(rate_jitter_entry)
        seed = _as_int(noise_seed_entry)

        if not 0.0 <= reporting_rate <= 1.0:
            raise ValueError
        if rate_jitter < 0.0:
            raise ValueError

        return reporting_rate, rate_jitter, seed

    except ValueError:
        fail("Ошибка", "Проверь шум: reporting rate должен быть в [0, 1], jitter >= 0, seed целый.")
        return None


def _sample_reporting_rates(rng, size, reporting_rate, rate_jitter):
    if rate_jitter == 0:
        return np.full(size, reporting_rate, dtype=float)

    rates = rng.normal(loc=reporting_rate, scale=rate_jitter, size=size)
    return np.clip(rates, 0.0, 1.0)


def _thin_count(value, rate, rng):
    n = int(max(0, round(float(value))))
    return float(rng.binomial(n, float(rate)))


def apply_underreporting_to_summary(summary, trajectory_points, grid_start_frac, grid_end_frac, seed):
    params = get_noise_params()
    if params is None:
        return None

    reporting_rate, rate_jitter, _ = params
    rng = np.random.default_rng(int(seed))
    result = dict(summary)

    grid_fractions = _make_fractions(trajectory_points, grid_start_frac, grid_end_frac)
    rates = _sample_reporting_rates(rng, len(grid_fractions), reporting_rate, rate_jitter)

    for index, rate in enumerate(rates):
        for series in TRAJECTORY_SERIES:
            col = f"grid_{index:03d}_{series}"
            if col in result:
                result[col] = _thin_count(result[col], rate, rng)

    for series in ("infect_A", "infect_B", "recover_A", "recover_B"):
        cols = [f"grid_{index:03d}_{series}" for index in range(trajectory_points)]
        values = [result[col] for col in cols if col in result]
        values = np.maximum.accumulate(values)

        for col, value in zip(cols, values):
            result[col] = float(value)

    for pop_name, pop_label in (("A", "A"), ("B", "B")):
        active_cols = [f"grid_{index:03d}_active_{pop_name}" for index in range(trajectory_points)]
        active_values = np.array([result[col] for col in active_cols if col in result], dtype=float)

        if active_values.size == 0:
            continue

        peak_index = int(np.argmax(active_values))
        result[f"final_{pop_label}"] = float(active_values[-1])
        result[f"peak_{pop_label}"] = float(active_values[peak_index])
        result[f"time_peak_{pop_label}_frac"] = float(grid_fractions[peak_index])
        result[f"mean_active_{pop_label}"] = float(np.mean(active_values))
        result[f"area_active_{pop_label}"] = float(np.mean(active_values) * float(result.get("t_end", 0.0)))

    b_active = np.array(
        [result.get(f"grid_{index:03d}_active_B", 0.0) for index in range(trajectory_points)],
        dtype=float,
    )
    positive_b = np.where(b_active > 0)[0]
    result["first_active_B_frac"] = float(grid_fractions[positive_b[0]]) if len(positive_b) else 1.0

    for pop in ("A", "B"):
        for prefix, aggregate_col in (
            ("infect", f"cum_infections_{pop}"),
            ("recover", f"cum_recoveries_{pop}"),
        ):
            cols = [f"grid_{index:03d}_{prefix}_{pop}" for index in range(trajectory_points)]
            values = [result[col] for col in cols if col in result]

            if values:
                result[aggregate_col] = float(values[-1])
            elif aggregate_col in result:
                result[aggregate_col] = _thin_count(result[aggregate_col], reporting_rate, rng)

    return result


def apply_underreporting_to_timeline(timeline, seed):
    params = get_noise_params()
    if params is None:
        return None

    reporting_rate, rate_jitter, _ = params
    rng = np.random.default_rng(int(seed))

    time = np.asarray(timeline["time"], dtype=float)
    noisy = {
        "time": time.copy(),
        "active": np.zeros_like(timeline["active"], dtype=float),
        "infections": np.zeros_like(timeline["infections"], dtype=float),
        "recoveries": np.zeros_like(timeline["recoveries"], dtype=float),
    }

    rates = _sample_reporting_rates(rng, len(time), reporting_rate, rate_jitter)

    for i, rate in enumerate(rates):
        for key in ("active", "infections", "recoveries"):
            for pop in range(POPULATIONS_NUMBER):
                noisy[key][i, pop] = _thin_count(timeline[key][i, pop], rate, rng)

    noisy["infections"] = np.maximum.accumulate(noisy["infections"], axis=0)
    noisy["recoveries"] = np.maximum.accumulate(noisy["recoveries"], axis=0)

    return noisy


def noise_enabled():
    return bool(noise_enabled_var.get())



def refresh_observation_target():
    if state["clean_observed_data"] is None:
        warn("Нет наблюдения", "Сначала нажми «Просимулировать».")
        return

    distance_params = get_distance_params()
    if distance_params is None:
        return

    trajectory_points, grid_start_frac, grid_end_frac = distance_params[:3]

    if noise_enabled():
        noise_params = get_noise_params()
        if noise_params is None:
            return

        _, _, noise_seed = noise_params

        state["observed_data"] = apply_underreporting_to_summary(
            state["clean_observed_data"],
            trajectory_points,
            grid_start_frac,
            grid_end_frac,
            seed=noise_seed,
        )
        state["observed_timeline"] = apply_underreporting_to_timeline(
            state["clean_observed_timeline"],
            seed=noise_seed + 1,
        )

        if state["observed_data"] is None or state["observed_timeline"] is None:
            return

        label = (
            f"Шум включён: используется неполная регистрация случаев. "
            f"reporting_rate={reporting_rate_entry.get()}, "
            f"rate_jitter={rate_jitter_entry.get()}."
        )

    else:
        state["observed_data"] = dict(state["clean_observed_data"])
        state["observed_timeline"] = {
            key: np.array(value, copy=True)
            for key, value in state["clean_observed_timeline"].items()
        }
        label = "Шум выключен: используется чистая VGsim-симуляция."

    clear_fit_outputs()
    state["last_observation_df"] = pd.DataFrame([state["observed_data"]])

    result_label.config(text=label)
    show_table(state["last_observation_df"])
    update_status("Текущее наблюдение обновлено.")


def toggle_noise():
    noise_enabled_var.set(not noise_enabled())
    refresh_observation_target()


def get_mu_grid():
    try:
        mu_min = _as_float(mu_min_entry)
        mu_max = _as_float(mu_max_entry)
        steps = _as_int(steps_entry)

        if mu_min < 0 or mu_max < 0 or mu_min > mu_max or steps < 2:
            raise ValueError

        return np.linspace(mu_min, mu_max, steps)

    except ValueError:
        fail("Ошибка", "Проверь min mu, max mu и steps.")
        return None


def get_vgsim_params():
    try:
        params = {
            "number_of_sites": _as_int(number_of_sites_entry),
            "population_a_size": _as_int(pop_a_size_entry),
            "population_b_size": _as_int(pop_b_size_entry),
            "transmission_rate": _as_float(transmission_rate_entry),
            "recovery_rate": _as_float(recovery_rate_entry),
            "sampling_rate": _as_float(sampling_rate_entry),
            "iterations": _as_int(iterations_entry),
            "sample_size": _as_int(sample_size_entry),
            "method": method_entry.get().strip(),
        }

        if not 1 <= params["number_of_sites"] <= 6:
            raise ValueError
        if params["population_a_size"] <= 0 or params["population_b_size"] <= 0:
            raise ValueError
        if params["transmission_rate"] < 0 or params["recovery_rate"] < 0 or params["sampling_rate"] < 0:
            raise ValueError
        if params["iterations"] <= 0 or params["sample_size"] <= 0:
            raise ValueError
        if params["method"] == "":
            raise ValueError

        return params

    except ValueError:
        fail("Ошибка", "Проверь параметры VGsim: sites, sizes, beta, gamma, sampling, iterations, sample size и method.")
        return None


def get_observation_params():
    try:
        obs_mu_ab = _as_float(obs_mu_ab_entry)
        obs_mu_ba = _as_float(obs_mu_ba_entry)
        obs_seed = _as_int(obs_seed_entry)

        if obs_mu_ab < 0 or obs_mu_ba < 0:
            raise ValueError

        return obs_mu_ab, obs_mu_ba, obs_seed

    except ValueError:
        fail("Ошибка", "Проверь параметры создания тестового наблюдения.")
        return None


def get_distance_params():
    try:
        params = (
            _as_int(trajectory_points_entry),
            _as_float(grid_start_entry),
            _as_float(grid_end_entry),
            _as_float(trajectory_weight_entry),
            _as_float(aggregate_weight_entry),
            _as_float(time_weight_entry),
            _as_float(time_extra_weight_entry),
            _as_float(time_tolerance_entry),
            _as_float(migration_weight_entry),
        )

        trajectory_points, grid_start_frac, grid_end_frac = params[:3]
        weights = params[3:]

        if trajectory_points < 2:
            raise ValueError
        if not 0.0 <= grid_start_frac < grid_end_frac <= 1.0:
            raise ValueError
        if any(weight < 0 for weight in weights):
            raise ValueError
        if sum(weights[:3]) + weights[3] + weights[5] <= 0:
            raise ValueError

        return params

    except ValueError:
        fail("Ошибка", "Проверь параметры distance.")
        return None


def get_fit_params():
    try:
        seeds = tuple(
            int(part.strip())
            for part in fit_seeds_entry.get().split(",")
            if part.strip()
        )

        params = (
            _as_int(top_k_entry),
            _as_int(refine_steps_entry),
            _as_int(refine_points_entry),
            _as_float(shrink_entry),
            seeds,
        )

        top_k, refine_steps, refine_points, shrink, seeds = params

        if top_k < 1 or refine_steps < 0 or refine_points < 2:
            raise ValueError
        if not 0 < shrink < 1:
            raise ValueError
        if not seeds:
            raise ValueError

        return params

    except ValueError:
        fail("Ошибка", "Проверь top_k, refine_steps, refine_points, shrink и fit seeds.")
        return None


def get_posterior_params():
    try:
        samples = _as_int(posterior_samples_entry)
        epsilon_quantile = _as_float(posterior_epsilon_quantile_entry)
        seed = _as_int(posterior_seed_entry)
        bins = _as_int(posterior_bins_entry)

        if samples < 5 or not 0.0 < epsilon_quantile <= 1.0 or bins < 2:
            raise ValueError

        return samples, epsilon_quantile, seed, bins

    except ValueError:
        fail("Ошибка", "Проверь posterior: samples >= 5, epsilon quantile в (0, 1], bins >= 2.")
        return None


def check_observation_compatibility(vgsim_params, grid_signature):
    if state["observed_data"] is None:
        warn("Нет наблюдения", "Сначала создай тестовое наблюдение VGsim.")
        return False

    if state["observation_grid_signature"] != grid_signature:
        warn("Параметры сетки изменились", "Пересоздай наблюдение или верни прежние параметры сетки.")
        return False

    if state["observation_vgsim_signature"] != _vgsim_signature(vgsim_params):
        warn(
            "Параметры VGsim изменились",
            "Наблюдение создано с другими beta/gamma/sampling/размерами/iterations. "
            "Пересоздай наблюдение или верни прежние параметры.",
        )
        return False

    return True


def _display_df(df):
    display_cols = []

    for group in (DISPLAY_FIRST_COLS, DIAGNOSTIC_COLS, AGGREGATE_COLS):
        for col in group:
            if col in df.columns and col not in display_cols:
                display_cols.append(col)

    for col in df.columns:
        if not col.startswith("grid_") and col not in display_cols:
            display_cols.append(col)

    return df[display_cols]


def show_table(df):
    df = _display_df(df)

    table.delete(*table.get_children())
    table["columns"] = list(df.columns)
    table["show"] = "headings"

    for col in df.columns:
        table.heading(col, text=col)
        table.column(col, width=120, anchor="center")

    for _, row in df.iterrows():
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(round(float(value), 6))
            else:
                values.append(value)
        table.insert("", "end", values=values)


def _plot_active_timeline(timeline, title, output_path, clean_timeline=None, fit_timeline=None):
    if timeline is None:
        warn("Нет графика", "Сначала создай нужную симуляцию.")
        return

    plt.figure(figsize=(10, 5))

    if clean_timeline is not None:
        plt.step(
            clean_timeline["time"],
            clean_timeline["active"][:, POP_A],
            where="post",
            color="red",
            alpha=0.95,
            linewidth=2.4,
            label="A true",
        )
        plt.step(
            clean_timeline["time"],
            clean_timeline["active"][:, POP_B],
            where="post",
            color="blue",
            alpha=0.95,
            linewidth=2.4,
            label="B true",
        )

    observed_alpha = 1.0 if clean_timeline is None else 0.55
    observed_width = 2.6 if clean_timeline is None else 2.0

    plt.step(
        timeline["time"],
        timeline["active"][:, POP_A],
        where="post",
        color="darkred",
        linewidth=observed_width,
        label="A observed",
        alpha=observed_alpha,
    )
    plt.step(
        timeline["time"],
        timeline["active"][:, POP_B],
        where="post",
        color="navy",
        linewidth=observed_width,
        label="B observed",
        alpha=observed_alpha,
    )

    if fit_timeline is not None:
        plt.step(
            fit_timeline["time"],
            fit_timeline["active"][:, POP_A],
            where="post",
            color="orange",
            linestyle="--",
            linewidth=2.0,
            label="A fit",
            alpha=1,
        )
        plt.step(
            fit_timeline["time"],
            fit_timeline["active"][:, POP_B],
            where="post",
            color="cyan",
            linestyle="--",
            linewidth=2.0,
            label="B fit",
            alpha=1,
        )

    plt.xlabel("Время")
    plt.ylabel("Активные заражённые")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.show()

def show_observation_plot():
    clean = state["clean_observed_timeline"] if noise_enabled() else None
    suffix = "underreported" if noise_enabled() else "clean"
    path = PLOTS_DIR / f"observation_active_{suffix}.png"

    _plot_active_timeline(
        state["observed_timeline"],
        "Текущее наблюдение: active infected A/B",
        path,
        clean_timeline=clean,
    )
    update_status(f"График наблюдения сохранён: {path}")


def show_best_fit_plot():
    if state["best_fit_mu_ab"] is None or state["best_fit_mu_ba"] is None:
        warn("Нет подбора", "Сначала выполни fit.")
        return

    distance_params = get_distance_params()
    if distance_params is None:
        return

    trajectory_points, grid_start_frac, grid_end_frac = distance_params[:3]

    if state["observation_grid_signature"] != (trajectory_points, grid_start_frac, grid_end_frac):
        warn("Параметры сетки изменились", "Верни параметры distance, с которыми создавалось наблюдение.")
        return

    try:
        update_status("Строим график лучшей подобранной симуляции...")

        _, fit_timeline = simulate(
            state["best_fit_mu_ab"],
            state["best_fit_mu_ba"],
            seed=state["best_fit_seed"],
            trajectory_points=trajectory_points,
            grid_start_frac=grid_start_frac,
            grid_end_frac=grid_end_frac,
            return_timeline=True,
            vgsim_params=state["observation_vgsim_params"],
        )

        clean = state["clean_observed_timeline"] if noise_enabled() else None
        path = PLOTS_DIR / "best_fit_active.png"
        _plot_active_timeline(
            state["observed_timeline"],
            (
                f"Лучшая чистая симуляция ({state['best_fit_kind']}): "
                f"mu_AB = {state['best_fit_mu_ab']:.6f}, "
                f"mu_BA = {state['best_fit_mu_ba']:.6f}, "
                f"seed = {state['best_fit_seed']}"
            ),
            path,
            clean_timeline=clean,
            fit_timeline=fit_timeline,
        )
        update_status(f"График сравнения сохранён: {path}")

    except Exception as error:
        fail("Ошибка", f"Ошибка исполнения:\n{error}")
        update_status("Ошибка исполнения.")


def create_observation():
    obs_params = get_observation_params()
    distance_params = get_distance_params()
    vgsim_params = get_vgsim_params()

    if obs_params is None or distance_params is None or vgsim_params is None:
        return

    obs_mu_ab, obs_mu_ba, obs_seed = obs_params
    trajectory_points, grid_start_frac, grid_end_frac = distance_params[:3]

    try:
        update_status("Создаём тестовое наблюдение через VGsim...")

        clean_data, clean_timeline = simulate(
            obs_mu_ab,
            obs_mu_ba,
            seed=obs_seed,
            trajectory_points=trajectory_points,
            grid_start_frac=grid_start_frac,
            grid_end_frac=grid_end_frac,
            return_timeline=True,
            vgsim_params=vgsim_params,
        )

        state["clean_observed_data"] = clean_data
        state["clean_observed_timeline"] = clean_timeline
        state["observation_grid_signature"] = (trajectory_points, grid_start_frac, grid_end_frac)
        state["observation_vgsim_signature"] = _vgsim_signature(vgsim_params)
        state["observation_vgsim_params"] = vgsim_params.copy()

        refresh_observation_target()
        update_status("Наблюдение готово.")

    except Exception as error:
        fail("Ошибка", f"Не удалось создать наблюдение:\n{error}")
        update_status("Ошибка при создании наблюдения.")


def _simulate_candidate_for_distance(mu_ab, mu_ba, seed, distance_params, vgsim_params, reference_t_end):
    trajectory_points, grid_start_frac, grid_end_frac = distance_params[:3]

    return simulate(
        mu_ab,
        mu_ba,
        seed=seed,
        trajectory_points=trajectory_points,
        grid_start_frac=grid_start_frac,
        grid_end_frac=grid_end_frac,
        reference_t_end=reference_t_end,
        vgsim_params=vgsim_params,
    )



def fit_equal_mu():
    mu_grid = get_mu_grid()
    distance_params = get_distance_params()
    fit_params = get_fit_params()
    vgsim_params = get_vgsim_params()

    if mu_grid is None or distance_params is None or fit_params is None or vgsim_params is None:
        return

    trajectory_points, grid_start_frac, grid_end_frac = distance_params[:3]
    grid_signature = (trajectory_points, grid_start_frac, grid_end_frac)

    if not check_observation_compatibility(vgsim_params, grid_signature):
        return

    top_k, refine_steps, refine_points, shrink, seeds = fit_params
    reference_t_end = float(state["observed_data"]["t_end"])

    def simulate_candidate(mu_ab, mu_ba, seed=0):
        return _simulate_candidate_for_distance(
            mu_ab=mu_ab,
            mu_ba=mu_ba,
            seed=seed,
            distance_params=distance_params,
            vgsim_params=vgsim_params,
            reference_t_end=reference_t_end,
        )

    try:
        update_status("Запускаем equal-fit через VGsim...")

        result = fit_equal_mu_refined(
            obs=state["observed_data"],
            mu_grid=mu_grid,
            simulate_func=simulate_candidate,
            top_k=top_k,
            refine_steps=refine_steps,
            refine_points=refine_points,
            shrink=shrink,
            seeds=seeds,
            trajectory_points=trajectory_points,
            trajectory_weight=distance_params[3],
            aggregate_weight=distance_params[4],
            time_weight=distance_params[5],
            time_extra_weight=distance_params[6],
            time_tolerance=distance_params[7],
            migration_weight=distance_params[8],
            progress_callback=lambda stage, max_stage, index, grid_size, mu: update_status(
                f"Equal-fit: stage {stage}/{max_stage}, mu {index}/{grid_size}, current mu = {mu:.6f}"
            ),
        )

        best = result.iloc[0]
        state["last_fit_df"] = result
        state["best_fit_mu"] = float(best["mu"])
        state["best_fit_mu_ab"] = float(best["mu_AB_candidate"])
        state["best_fit_mu_ba"] = float(best["mu_BA_candidate"])
        state["best_fit_kind"] = "equal"
        state["best_fit_seed"] = int(seeds[0])

        target_label = "шумному наблюдению" if noise_enabled() else "чистому наблюдению"

        result_label.config(
            text=(
                f"Лучший equal-fit по {target_label}: mu = {best['mu']:.6f}, "
                f"distance = {best['distance']:.6f}, "
                f"traj/agg/time/textra/mig = "
                f"{best['trajectory_distance']:.4f}/"
                f"{best['aggregate_distance']:.4f}/"
                f"{best['time_distance']:.4f}/"
                f"{best['time_extra_distance']:.4f}/"
                f"{best['migration_distance']:.4f}."
            )
        )

        update_status("Equal-fit готов.")
        show_table(result)

    except Exception as error:
        fail("Ошибка", f"Не удалось выполнить equal-fit:\n{error}")
        update_status("Ошибка при equal-fit.")


def fit_unequal_mu():
    mu_grid = get_mu_grid()
    distance_params = get_distance_params()
    fit_params = get_fit_params()
    vgsim_params = get_vgsim_params()

    if mu_grid is None or distance_params is None or fit_params is None or vgsim_params is None:
        return

    trajectory_points, grid_start_frac, grid_end_frac = distance_params[:3]
    grid_signature = (trajectory_points, grid_start_frac, grid_end_frac)

    if not check_observation_compatibility(vgsim_params, grid_signature):
        return

    top_k, refine_steps, refine_points, shrink, seeds = fit_params
    reference_t_end = float(state["observed_data"]["t_end"])

    def simulate_candidate(mu_ab, mu_ba, seed=0):
        return _simulate_candidate_for_distance(
            mu_ab=mu_ab,
            mu_ba=mu_ba,
            seed=seed,
            distance_params=distance_params,
            vgsim_params=vgsim_params,
            reference_t_end=reference_t_end,
        )

    try:
        update_status("Запускаем unequal-fit через VGsim...")

        result = fit_unequal_mu_refined(
            obs=state["observed_data"],
            mu_ab_grid=mu_grid,
            mu_ba_grid=mu_grid,
            simulate_func=simulate_candidate,
            top_k=top_k,
            refine_steps=refine_steps,
            refine_points=refine_points,
            shrink=shrink,
            seeds=seeds,
            trajectory_points=trajectory_points,
            trajectory_weight=distance_params[3],
            aggregate_weight=distance_params[4],
            time_weight=distance_params[5],
            time_extra_weight=distance_params[6],
            time_tolerance=distance_params[7],
            migration_weight=distance_params[8],
            progress_callback=lambda stage, max_stage, index, grid_size, mu_ab, mu_ba: update_status(
                f"Unequal-fit: stage {stage}/{max_stage}, pair {index}/{grid_size}, "
                f"mu_AB = {mu_ab:.6f}, mu_BA = {mu_ba:.6f}"
            ),
        )

        best = result.iloc[0]
        state["last_fit_df"] = result
        state["best_fit_mu"] = float(best["mu"])
        state["best_fit_mu_ab"] = float(best["mu_AB_candidate"])
        state["best_fit_mu_ba"] = float(best["mu_BA_candidate"])
        state["best_fit_kind"] = "unequal"
        state["best_fit_seed"] = int(seeds[0])

        target_label = "шумному наблюдению" if noise_enabled() else "чистому наблюдению"

        result_label.config(
            text=(
                f"Лучший unequal-fit по {target_label}: "
                f"mu_AB = {best['mu_AB_candidate']:.6f}, "
                f"mu_BA = {best['mu_BA_candidate']:.6f}, "
                f"distance = {best['distance']:.6f}, "
                f"traj/agg/time/textra/mig = "
                f"{best['trajectory_distance']:.4f}/"
                f"{best['aggregate_distance']:.4f}/"
                f"{best['time_distance']:.4f}/"
                f"{best['time_extra_distance']:.4f}/"
                f"{best['migration_distance']:.4f}."
            )
        )

        update_status("Unequal-fit готов.")
        show_table(result)

    except Exception as error:
        fail("Ошибка", f"Не удалось выполнить unequal-fit:\n{error}")
        update_status("Ошибка при unequal-fit.")


def fit_unequal_mu_fast():
    mu_grid = get_mu_grid()
    distance_params = get_distance_params()
    fit_params = get_fit_params()
    vgsim_params = get_vgsim_params()

    if mu_grid is None or distance_params is None or fit_params is None or vgsim_params is None:
        return

    trajectory_points, grid_start_frac, grid_end_frac = distance_params[:3]
    grid_signature = (trajectory_points, grid_start_frac, grid_end_frac)

    if not check_observation_compatibility(vgsim_params, grid_signature):
        return

    top_k, refine_steps, refine_points, shrink, seeds = fit_params
    rerank_top = max(15, int(top_k) * int(refine_points))
    reference_t_end = float(state["observed_data"]["t_end"])

    def simulate_candidate(mu_ab, mu_ba, seed=0):
        return _simulate_candidate_for_distance(
            mu_ab=mu_ab,
            mu_ba=mu_ba,
            seed=seed,
            distance_params=distance_params,
            vgsim_params=vgsim_params,
            reference_t_end=reference_t_end,
        )

    try:
        update_status("Запускаем fast unequal-fit через VGsim...")

        result = fit_unequal_mu_fast_refined(
            obs=state["observed_data"],
            mu_ab_grid=mu_grid,
            mu_ba_grid=mu_grid,
            simulate_func=simulate_candidate,
            top_k=top_k,
            refine_steps=refine_steps,
            refine_points=refine_points,
            shrink=shrink,
            seeds=seeds,
            rerank_top=rerank_top,
            trajectory_points=trajectory_points,
            trajectory_weight=distance_params[3],
            aggregate_weight=distance_params[4],
            time_weight=distance_params[5],
            time_extra_weight=distance_params[6],
            time_tolerance=distance_params[7],
            migration_weight=distance_params[8],
            progress_callback=lambda phase, stage, max_stage, index, grid_size, mu_ab, mu_ba: update_status(
                f"Fast unequal-fit {phase}: stage {stage}/{max_stage}, pair {index}/{grid_size}, "
                f"mu_AB = {mu_ab:.6f}, mu_BA = {mu_ba:.6f}"
            ),
        )

        best = result.iloc[0]
        state["last_fit_df"] = result
        state["best_fit_mu"] = float(best["mu"])
        state["best_fit_mu_ab"] = float(best["mu_AB_candidate"])
        state["best_fit_mu_ba"] = float(best["mu_BA_candidate"])
        state["best_fit_kind"] = "unequal fast"
        state["best_fit_seed"] = int(seeds[0])

        target_label = "шумному наблюдению" if noise_enabled() else "чистому наблюдению"

        result_label.config(
            text=(
                f"Лучший fast unequal-fit по {target_label}: "
                f"mu_AB = {best['mu_AB_candidate']:.6f}, "
                f"mu_BA = {best['mu_BA_candidate']:.6f}, "
                f"distance = {best['distance']:.6f}, "
                f"rerank_top = {rerank_top}, "
                f"traj/agg/time/textra/mig = "
                f"{best['trajectory_distance']:.4f}/"
                f"{best['aggregate_distance']:.4f}/"
                f"{best['time_distance']:.4f}/"
                f"{best['time_extra_distance']:.4f}/"
                f"{best['migration_distance']:.4f}."
            )
        )

        update_status("Fast unequal-fit готов.")
        show_table(result)

    except Exception as error:
        fail("Ошибка", f"Ошибка исполнения:\n{error}")
        update_status("Ошибка исполнения.")

def create_observation_and_fit():
    create_observation()
    if state["observed_data"] is not None:
        fit_equal_mu()


def show_abc_posterior_distribution():
    mu_grid = get_mu_grid()
    distance_params = get_distance_params()
    posterior_params = get_posterior_params()
    vgsim_params = get_vgsim_params()

    if mu_grid is None or distance_params is None or posterior_params is None or vgsim_params is None:
        return

    trajectory_points, grid_start_frac, grid_end_frac = distance_params[:3]
    grid_signature = (trajectory_points, grid_start_frac, grid_end_frac)

    if not check_observation_compatibility(vgsim_params, grid_signature):
        return

    samples, epsilon_quantile, posterior_seed, bins = posterior_params
    mu_min = float(np.min(mu_grid))
    mu_max = float(np.max(mu_grid))
    reference_t_end = float(state["observed_data"]["t_end"])
    rng = np.random.default_rng(posterior_seed)
    rows = []

    try:
        update_status("Строим ABC posterior: запускаем симуляции из prior...")

        for index in range(samples):
            mu = float(rng.uniform(mu_min, mu_max))
            sim_seed = int(rng.integers(1, 2_147_483_647))

            update_status(f"ABC posterior: sample {index + 1}/{samples}, mu = {mu:.6f}")

            sim = _simulate_candidate_for_distance(
                mu_ab=mu,
                mu_ba=mu,
                seed=sim_seed,
                distance_params=distance_params,
                vgsim_params=vgsim_params,
                reference_t_end=reference_t_end,
            )

            dist_info = distance_components(
                sim=sim,
                obs=state["observed_data"],
                trajectory_points=trajectory_points,
                trajectory_weight=distance_params[3],
                aggregate_weight=distance_params[4],
                time_weight=distance_params[5],
                time_extra_weight=distance_params[6],
                time_tolerance=distance_params[7],
                migration_weight=distance_params[8],
            )

            rows.append({
                "mu": mu,
                "mu_AB_candidate": mu,
                "mu_BA_candidate": mu,
                "seed": sim_seed,
                **dist_info,
                **sim,
            })

        posterior_df = pd.DataFrame(rows).sort_values("distance").reset_index(drop=True)
        epsilon = float(np.quantile(posterior_df["distance"], epsilon_quantile))
        posterior_df["accepted"] = posterior_df["distance"] <= epsilon

        accepted_df = posterior_df[posterior_df["accepted"]].copy()

        if accepted_df.empty:
            warn("Нет принятых mu", "ABC posterior не построился: ни один sample не прошёл epsilon.")
            update_status("ABC posterior: нет принятых sample.")
            return

        accepted_mu = accepted_df["mu"].to_numpy(dtype=float)

        map_mu = float(accepted_df.iloc[0]["mu"])
        posterior_mean = float(np.mean(accepted_mu))
        posterior_median = float(np.median(accepted_mu))
        low95, high95 = _weighted_quantile(accepted_mu, [0.025, 0.975])

        state["last_posterior_df"] = posterior_df

        noise_title = "target: noisy registered observation" if noise_enabled() else "target: clean observation"
        path = PLOTS_DIR / "abc_posterior_mu.png"

        plt.figure(figsize=(10, 5))
        plt.hist(
            accepted_mu,
            bins=bins,
            range=(mu_min, mu_max),
            density=True,
            alpha=0.65,
            label="accepted mu",
        )
        plt.axvline(map_mu, linestyle="--", linewidth=1.5, label=f"MAP mu = {map_mu:.6f}")
        plt.axvline(posterior_mean, linestyle=":", linewidth=1.8, label=f"mean mu = {posterior_mean:.6f}")
        plt.axvspan(low95, high95, alpha=0.15, label=f"95% interval [{low95:.6f}, {high95:.6f}]")
        plt.xlabel("mu")
        plt.ylabel("posterior density")
        plt.title(
            "ABC posterior for equal migration mu\n"
            f"{noise_title}, prior: Uniform({mu_min:.6f}, {mu_max:.6f}), "
            f"epsilon: {epsilon_quantile:.2f} quantile = {epsilon:.6f}, "
            f"accepted: {len(accepted_df)}/{len(posterior_df)}"
        )
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.show()

        result_label.config(
            text=(
                f"ABC posterior: MAP mu = {map_mu:.6f}, "
                f"mean = {posterior_mean:.6f}, "
                f"median = {posterior_median:.6f}, "
                f"95% interval = [{low95:.6f}, {high95:.6f}], "
                f"epsilon = {epsilon:.6f}, "
                f"accepted = {len(accepted_df)}/{len(posterior_df)}."
            )
        )

        update_status(f"ABC posterior готов. График сохранён: {path}")
        show_table(posterior_df)

    except Exception as error:
        fail("Ошибка", f"Не удалось построить ABC posterior:\n{error}")
        update_status("Ошибка при построении ABC posterior.")


reset_run_dirs()

root = tk.Tk()
root.title("VGsim Predict - equal/unequal mu + underreporting")
root.geometry("1550x950")
root.minsize(1300, 800)

try:
    root.state("zoomed")
except tk.TclError:
    root.attributes("-zoomed", True)

style = ttk.Style()
style.theme_use("clam")
style.configure("TButton", padding=5)
style.configure("Treeview", rowheight=24)

settings_frame = ttk.LabelFrame(root, text="Сетка поиска mu")
settings_frame.pack(fill="x", padx=10, pady=5)

mu_min_entry = _make_entry(settings_frame, 0, 0, "min mu", "0.001")
mu_max_entry = _make_entry(settings_frame, 0, 2, "max mu", "0.300")
steps_entry = _make_entry(settings_frame, 0, 4, "steps", "10")

virus_frame = ttk.LabelFrame(root, text="Параметры вируса и VGsim")
virus_frame.pack(fill="x", padx=10, pady=5)

number_of_sites_entry = _make_entry(virus_frame, 0, 0, "sites", DEFAULT_VGSIM_PARAMS["number_of_sites"])
pop_a_size_entry = _make_entry(virus_frame, 0, 2, "pop A size", DEFAULT_VGSIM_PARAMS["population_a_size"], width=14)
pop_b_size_entry = _make_entry(virus_frame, 0, 4, "pop B size", DEFAULT_VGSIM_PARAMS["population_b_size"], width=14)
transmission_rate_entry = _make_entry(virus_frame, 1, 0, "beta", DEFAULT_VGSIM_PARAMS["transmission_rate"])
recovery_rate_entry = _make_entry(virus_frame, 1, 2, "gamma", DEFAULT_VGSIM_PARAMS["recovery_rate"])
sampling_rate_entry = _make_entry(virus_frame, 1, 4, "sampling", DEFAULT_VGSIM_PARAMS["sampling_rate"])
iterations_entry = _make_entry(virus_frame, 2, 0, "iterations", DEFAULT_VGSIM_PARAMS["iterations"], width=14)
sample_size_entry = _make_entry(virus_frame, 2, 2, "sample size", DEFAULT_VGSIM_PARAMS["sample_size"], width=14)
method_entry = _make_entry(virus_frame, 2, 4, "method", DEFAULT_VGSIM_PARAMS["method"])

obs_frame = ttk.LabelFrame(root, text="Параметры создания тестового наблюдения VGsim")
obs_frame.pack(fill="x", padx=10, pady=5)

obs_mu_ab_entry = _make_entry(obs_frame, 0, 0, "obs mu_AB", "0.100")
obs_mu_ba_entry = _make_entry(obs_frame, 0, 2, "obs mu_BA", "0.100")
obs_seed_entry = _make_entry(obs_frame, 0, 4, "obs seed", "777")

noise_frame = ttk.LabelFrame(root, text="Модель наблюдения: неполная регистрация")
noise_frame.pack(fill="x", padx=10, pady=5)

noise_enabled_var = tk.BooleanVar(value=False)
ttk.Checkbutton(noise_frame, text="шум", variable=noise_enabled_var).grid(
    row=0,
    column=0,
    columnspan=3,
    padx=5,
    pady=4,
    sticky="w",
)

reporting_rate_entry = _make_entry(noise_frame, 0, 3, "reporting rate p", "0.80")
rate_jitter_entry = _make_entry(noise_frame, 0, 5, "rate jitter", "0.03")
noise_seed_entry = _make_entry(noise_frame, 0, 7, "noise seed", "2027")

ttk.Button(noise_frame, text="Применить шум", command=refresh_observation_target, width=28).grid(
    row=0,
    column=9,
    padx=5,
    pady=4,
)
ttk.Button(noise_frame, text="Вкл/выкл шум", command=toggle_noise, width=18).grid(
    row=0,
    column=10,
    padx=5,
    pady=4,
)

distance_frame = ttk.LabelFrame(root, text="Параметры distance")
distance_frame.pack(fill="x", padx=10, pady=5)

trajectory_points_entry = _make_entry(distance_frame, 0, 0, "trajectory points", "100")
grid_start_entry = _make_entry(distance_frame, 0, 2, "grid start", "0.00")
grid_end_entry = _make_entry(distance_frame, 0, 4, "grid end", "1.00")
trajectory_weight_entry = _make_entry(distance_frame, 1, 0, "trajectory weight", "1.0")
aggregate_weight_entry = _make_entry(distance_frame, 1, 2, "aggregate weight", "0.75")
time_weight_entry = _make_entry(distance_frame, 1, 4, "time weight", "1.0")
time_extra_weight_entry = _make_entry(distance_frame, 1, 6, "time extra weight", "3.0")
time_tolerance_entry = _make_entry(distance_frame, 2, 0, "time tolerance", "0.10")
migration_weight_entry = _make_entry(distance_frame, 2, 2, "migration event weight", "0.0")

fit_frame = ttk.LabelFrame(root, text="Параметры refined fit")
fit_frame.pack(fill="x", padx=10, pady=5)

top_k_entry = _make_entry(fit_frame, 0, 0, "top_k", "3")
refine_steps_entry = _make_entry(fit_frame, 0, 2, "refine_steps", "2")
refine_points_entry = _make_entry(fit_frame, 0, 4, "refine_points", "5")
shrink_entry = _make_entry(fit_frame, 0, 6, "shrink", "0.35")
fit_seeds_entry = _make_entry(fit_frame, 0, 8, "fit seeds", "1,2,3,4,5", width=18)

posterior_frame = ttk.LabelFrame(root, text="Параметры ABC posterior")
posterior_frame.pack(fill="x", padx=10, pady=5)

posterior_samples_entry = _make_entry(posterior_frame, 0, 0, "samples", "40")
posterior_epsilon_quantile_entry = _make_entry(posterior_frame, 0, 2, "epsilon quantile", "0.25")
posterior_seed_entry = _make_entry(posterior_frame, 0, 4, "posterior seed", "2026")
posterior_bins_entry = _make_entry(posterior_frame, 0, 6, "bins", "12")

actions_frame = ttk.LabelFrame(root, text="Действия")
actions_frame.pack(fill="x", padx=10, pady=5)

ttk.Button(actions_frame, text="Просимулировать", command=create_observation, width=34).grid(row=0, column=0, padx=5, pady=4)
ttk.Button(actions_frame, text="Подобрать equal mu", command=fit_equal_mu, width=34).grid(row=0, column=1, padx=5, pady=4)
ttk.Button(actions_frame, text="Подобрать unequal mu", command=fit_unequal_mu, width=34).grid(row=0, column=3, padx=5, pady=4)
ttk.Button(actions_frame, text="Подобрать unequal fast", command=fit_unequal_mu_fast, width=34).grid(row=0, column=4, padx=5, pady=4)
ttk.Button(actions_frame, text="Просимулировать + equal", command=create_observation_and_fit, width=34).grid(row=0, column=2, padx=5, pady=4)
ttk.Button(actions_frame, text="Исходный граф", command=show_observation_plot, width=34).grid(row=1, column=0, padx=5, pady=4)
ttk.Button(actions_frame, text="Подобранный граф", command=show_best_fit_plot, width=34).grid(row=1, column=1, padx=5, pady=4)
ttk.Button(actions_frame, text="ABC posterior по mu", command=show_abc_posterior_distribution, width=34).grid(row=1, column=2, padx=5, pady=4)

status_label = ttk.Label(root, text="Готово.", anchor="w")
status_label.pack(fill="x", padx=10, pady=5)

result_label = ttk.Label(root, text="", anchor="w", wraplength=1450, justify="left")
result_label.pack(fill="x", padx=10, pady=5)

table_frame = ttk.Frame(root)
table_frame.pack(fill="both", expand=True, padx=10, pady=10)

table = ttk.Treeview(table_frame)
table.grid(row=0, column=0, sticky="nsew")

scrollbar_y = ttk.Scrollbar(table_frame, orient="vertical", command=table.yview)
scrollbar_y.grid(row=0, column=1, sticky="ns")

scrollbar_x = ttk.Scrollbar(table_frame, orient="horizontal", command=table.xview)
scrollbar_x.grid(row=1, column=0, sticky="ew")

table.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

table_frame.rowconfigure(0, weight=1)
table_frame.columnconfigure(0, weight=1)

root.mainloop()
