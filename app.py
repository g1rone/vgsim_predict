import numpy as np
import pandas as pd
import streamlit as st

from abc_fit import SUMMARY_COLS, fit_equal_mu, fit_unequal_mu


def simulate(mu_ab, mu_ba, seed=0):
    """
    Простая заглушка.

    Потом сюда подключим настоящий VGsim.
    Сейчас это нужно, чтобы UI и подбор уже работали.
    """

    rng = np.random.default_rng(seed)

    mig_ab = int(rng.poisson(500 * mu_ab))
    mig_ba = int(rng.poisson(500 * mu_ba))

    final_a = 500 - mig_ab + mig_ba
    final_b = 500 + mig_ab - mig_ba

    return {
        "mig_AB": mig_ab,
        "mig_BA": mig_ba,
        "final_A": final_a,
        "final_B": final_b,
        "t_end": 100,
    }


st.title("VGsim Predict")

st.sidebar.header("Настройки сетки")

mu_min = st.sidebar.number_input("min μ", value=0.001, format="%.5f")
mu_max = st.sidebar.number_input("max μ", value=0.050, format="%.5f")
steps = st.sidebar.number_input("steps", value=10, min_value=2)

mu_grid = np.linspace(mu_min, mu_max, int(steps))

tab1, tab2, tab3 = st.tabs([
    "Загрузить CSV",
    "Засимулировать",
    "Засимулировать + подобрать",
])


with tab1:
    st.header("Подбор по загруженному CSV")

    file = st.file_uploader("Загрузи observed_summary.csv", type="csv")

    if file is not None:
        df = pd.read_csv(file)

        missing = [col for col in SUMMARY_COLS if col not in df.columns]

        if missing:
            st.error(f"В файле нет колонок: {missing}")
        else:
            obs = df.iloc[0][SUMMARY_COLS].astype(float)

            st.write("Загруженные данные:")
            st.dataframe(df)

            if st.button("Подобрать параметры"):
                equal = fit_equal_mu(obs, mu_grid, simulate)
                unequal = fit_unequal_mu(obs, mu_grid, simulate)

                st.subheader("Случай μ_AB = μ_BA")
                st.dataframe(equal)

                st.subheader("Случай μ_AB ≠ μ_BA")
                st.dataframe(unequal)

                best = unequal.iloc[0]

                st.success(
                    f"Лучшие параметры: "
                    f"μ_AB = {best['mu_AB']:.5f}, "
                    f"μ_BA = {best['mu_BA']:.5f}, "
                    f"distance = {best['distance']:.5f}"
                )


with tab2:
    st.header("Засимулировать с параметрами")

    mu_ab = st.number_input("μ_AB", value=0.020, format="%.5f", key="sim_ab")
    mu_ba = st.number_input("μ_BA", value=0.050, format="%.5f", key="sim_ba")
    seed = st.number_input("seed", value=0, step=1, key="sim_seed")

    if st.button("Засимулировать"):
        sim = simulate(mu_ab, mu_ba, int(seed))
        sim_df = pd.DataFrame([sim])

        st.dataframe(sim_df)

        st.download_button(
            "Скачать CSV",
            sim_df.to_csv(index=False),
            "observed_summary.csv",
            "text/csv",
        )


with tab3:
    st.header("Засимулировать и сразу подобрать")

    true_mu_ab = st.number_input("true μ_AB", value=0.020, format="%.5f", key="true_ab")
    true_mu_ba = st.number_input("true μ_BA", value=0.050, format="%.5f", key="true_ba")
    seed = st.number_input("seed", value=0, step=1, key="true_seed")

    if st.button("Засимулировать и подобрать"):
        obs = simulate(true_mu_ab, true_mu_ba, int(seed))
        obs_df = pd.DataFrame([obs])

        st.write("Сначала получили simulated observed data:")
        st.dataframe(obs_df)

        equal = fit_equal_mu(obs, mu_grid, simulate)
        unequal = fit_unequal_mu(obs, mu_grid, simulate)

        st.subheader("Случай μ_AB = μ_BA")
        st.dataframe(equal)

        st.subheader("Случай μ_AB ≠ μ_BA")
        st.dataframe(unequal)

        best = unequal.iloc[0]

        st.info(
            f"Истинные параметры: "
            f"μ_AB = {true_mu_ab:.5f}, "
            f"μ_BA = {true_mu_ba:.5f}"
        )

        st.success(
            f"Лучшие найденные параметры: "
            f"μ_AB = {best['mu_AB']:.5f}, "
            f"μ_BA = {best['mu_BA']:.5f}, "
            f"distance = {best['distance']:.5f}"
        )

        st.download_button(
            "Скачать generated_observed_summary.csv",
            obs_df.to_csv(index=False),
            "generated_observed_summary.csv",
            "text/csv",
        )