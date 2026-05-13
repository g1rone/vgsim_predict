from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt


# Коды событий VGsim:
# BIRTH = заражение
# DEATH = выход из infectious-состояния, в нашей интерпретации recovery
# SAMPLING = попадание в sample, тоже убирает объект из active infectious
# MUTATION = смена гаплотипа
# MIGRATION = переход заражённой линии между популяциями

BIRTH = 0
DEATH = 1
SAMPLING = 2
MUTATION = 3
SUSCCHANGE = 4
MIGRATION = 5


def export_and_load_chain_events(simulator, out_dir, basename="toy_chain_events"):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    base_path = out_dir / basename

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
        raise RuntimeError("В chain events не найдено событий с time > 0.")

    last_index = valid_indices[-1] + 1

    times = times[:last_index]
    event_types = event_types[:last_index]
    haplotypes = haplotypes[:last_index]
    populations = populations[:last_index]
    new_haplotypes = new_haplotypes[:last_index]
    new_populations = new_populations[:last_index]

    order = np.argsort(times)

    return {
        "time": times[order],
        "type": event_types[order],
        "haplotype": haplotypes[order],
        "population": populations[order],
        "new_haplotype": new_haplotypes[order],
        "new_population": new_populations[order],
        "path": npy_path,
    }


def build_epidemic_timelines(chain_events, populations_number=2, haplotypes_number=4):
    times = chain_events["time"]
    event_types = chain_events["type"]
    haplotypes = chain_events["haplotype"]
    populations = chain_events["population"]
    new_haplotypes = chain_events["new_haplotype"]
    new_populations = chain_events["new_population"]

    active_by_pop_hap = np.zeros(
        (populations_number, haplotypes_number),
        dtype=int,
    )

    # Стартуем с одного заражённого в A, гаплотип H0
    active_by_pop_hap[0, 0] = 1

    cumulative_infections = np.zeros(populations_number, dtype=int)
    cumulative_recoveries = np.zeros(populations_number, dtype=int)

    timeline_time = [0.0]
    timeline_active = [active_by_pop_hap.sum(axis=1).copy()]
    timeline_infections = [cumulative_infections.copy()]
    timeline_recoveries = [cumulative_recoveries.copy()]

    for t, event_type, hap, pop, new_hap, new_pop in zip(
        times,
        event_types,
        haplotypes,
        populations,
        new_haplotypes,
        new_populations,
    ):
        if event_type == BIRTH:
            if 0 <= pop < populations_number and 0 <= hap < haplotypes_number:
                active_by_pop_hap[pop, hap] += 1
                cumulative_infections[pop] += 1

        elif event_type == DEATH:
            if 0 <= pop < populations_number and 0 <= hap < haplotypes_number:
                active_by_pop_hap[pop, hap] -= 1
                cumulative_recoveries[pop] += 1

        elif event_type == SAMPLING:
            if 0 <= pop < populations_number and 0 <= hap < haplotypes_number:
                active_by_pop_hap[pop, hap] -= 1

        elif event_type == MUTATION:
            if 0 <= pop < populations_number:
                if 0 <= hap < haplotypes_number:
                    active_by_pop_hap[pop, hap] -= 1
                if 0 <= new_hap < haplotypes_number:
                    active_by_pop_hap[pop, new_hap] += 1

        elif event_type == MIGRATION:
            if (
                0 <= pop < populations_number
                and 0 <= new_pop < populations_number
                and 0 <= hap < haplotypes_number
            ):
                active_by_pop_hap[pop, hap] -= 1
                active_by_pop_hap[new_pop, hap] += 1

        elif event_type == SUSCCHANGE:
            pass

        active_by_pop_hap = np.maximum(active_by_pop_hap, 0)

        timeline_time.append(float(t))
        timeline_active.append(active_by_pop_hap.sum(axis=1).copy())
        timeline_infections.append(cumulative_infections.copy())
        timeline_recoveries.append(cumulative_recoveries.copy())

    return {
        "time": np.asarray(timeline_time, dtype=float),
        "active": np.asarray(timeline_active, dtype=float),
        "infections": np.asarray(timeline_infections, dtype=float),
        "recoveries": np.asarray(timeline_recoveries, dtype=float),
    }


def save_timelines_csv(timelines, out_dir, filename="epidemic_timelines.csv"):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    path = out_dir / filename

    with open(path, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "time",
            "active_A",
            "active_B",
            "cumulative_infections_A",
            "cumulative_infections_B",
            "cumulative_recoveries_A",
            "cumulative_recoveries_B",
        ])

        for t, active, infections, recoveries in zip(
            timelines["time"],
            timelines["active"],
            timelines["infections"],
            timelines["recoveries"],
        ):
            writer.writerow([
                t,
                active[0],
                active[1],
                infections[0],
                infections[1],
                recoveries[0],
                recoveries[1],
            ])

    print(f"✅ CSV с динамикой сохранён: {path}")


def plot_two_populations(time, values, title, ylabel, out_dir, filename):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(10, 5))

    plt.step(time, values[:, 0], where="post", label="Популяция A", color="red")
    plt.step(time, values[:, 1], where="post", label="Популяция B", color="blue")

    plt.title(title)
    plt.xlabel("Время симуляции")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = out_dir / filename
    plt.savefig(path, dpi=200)
    plt.close()

    print(f"✅ График сохранён: {path}")


def make_epidemic_plots_from_events(
    simulator,
    out_dir="toy_output",
    populations_number=2,
    haplotypes_number=4,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    chain_events = export_and_load_chain_events(
        simulator=simulator,
        out_dir=out_dir,
        basename="toy_chain_events",
    )

    print(f"✅ Chain events сохранены и прочитаны: {chain_events['path']}")
    print(f"Количество событий: {len(chain_events['time'])}")

    timelines = build_epidemic_timelines(
        chain_events=chain_events,
        populations_number=populations_number,
        haplotypes_number=haplotypes_number,
    )

    save_timelines_csv(
        timelines=timelines,
        out_dir=out_dir,
        filename="epidemic_timelines.csv",
    )

    plot_two_populations(
        time=timelines["time"],
        values=timelines["active"],
        title="Динамика числа активных заражённых во времени",
        ylabel="Число активных заражённых",
        out_dir=out_dir,
        filename="active_infectious_ab.png",
    )

    plot_two_populations(
        time=timelines["time"],
        values=timelines["infections"],
        title="Накопленное число заражений во времени",
        ylabel="Число заражений",
        out_dir=out_dir,
        filename="cumulative_infections_ab.png",
    )

    plot_two_populations(
        time=timelines["time"],
        values=timelines["recoveries"],
        title="Накопленное число выздоровлений во времени",
        ylabel="Число выздоровлений",
        out_dir=out_dir,
        filename="cumulative_recoveries_ab.png",
    )