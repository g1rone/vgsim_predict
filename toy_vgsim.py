from pathlib import Path
from collections import Counter

import VGsim

from vgsim_plots import make_epidemic_plots_from_events


def main():
    # 2 популяции: 0 = A, 1 = B
    number_of_sites = 1
    populations_number = 2
    number_of_susceptible_groups = 1

    simulator = VGsim.Simulator(
        number_of_sites,
        populations_number,
        number_of_susceptible_groups,
        seed=42,
    )

    # размеры популяций
    simulator.set_population_size(200_000, population=0)  # A
    simulator.set_population_size(100_000, population=1)  # B

    # эпидемиологические параметры
    simulator.set_transmission_rate(0.10)  # beta
    simulator.set_recovery_rate(0.10)      # gamma
    simulator.set_sampling_rate(0.01)      # sampling

    # миграция: A -> B сильнее, чем B -> A
    simulator.set_migration_probability(0.02, source=0, target=1)    # mu_AB
    simulator.set_migration_probability(0.002, source=1, target=0)   # mu_BA

    # запуск симуляции
    simulator.simulate(
        1_000_000,
        sample_size=500,
        method="direct",
    )

    out = Path("toy_output")
    out.mkdir(exist_ok=True)

    base = str(out / "toy")

    # основные графики для курсовой
    make_epidemic_plots_from_events(
        simulator=simulator,
        out_dir=out,
        populations_number=populations_number,
        haplotypes_number=4 ** number_of_sites,
    )

    # дерево по сэмплам
    simulator.genealogy()

    # экспорт дерева и миграций
    simulator.export_newick(base)
    simulator.export_migrations(base)

    # экспорт мутаций, если метод есть
    if hasattr(simulator, "export_mutations"):
        simulator.export_mutations(base)
    elif hasattr(simulator, "output_mutations"):
        simulator.output_mutations(base)
    else:
        print("⚠️ export/output для mutations не найден.")

    # метаданные сэмплов
    res = simulator.output_sample_data(output_print=False)

    if res is None:
        print("✅ sample_data: функция в этой сборке ничего не возвращает.")
        print("   Поэтому читаем результаты из файлов в toy_output.")
    else:
        times, pops, haps = res
        print("samples:", len(times))
        print("samples by population:", dict(Counter(pops)))
        print("first 5 samples:", list(zip(times, pops, haps))[:5])

    print("files in toy_output:", [p.name for p in out.iterdir()])


if __name__ == "__main__":
    main()