import tkinter as tk
from tkinter import filedialog, messagebox, ttk, font as tkfont

import numpy as np
import pandas as pd

from abc_fit import SUMMARY_COLS, fit_equal_mu_refined, fit_unequal_mu


def simulate(mu_ab, mu_ba, seed=0):
    """
    Пока это простая toy-симуляция.

    Потом сюда вставим настоящий запуск VGsim.
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


def get_mu_grid():
    try:
        mu_min = float(mu_min_entry.get())
        mu_max = float(mu_max_entry.get())
        steps = int(steps_entry.get())

        if mu_min < 0:
            raise ValueError

        if mu_max < 0:
            raise ValueError

        if mu_min > mu_max:
            raise ValueError

        if steps < 2:
            raise ValueError

        return np.linspace(mu_min, mu_max, steps)

    except ValueError:
        messagebox.showerror(
            "Ошибка",
            "Проверь настройки сетки: min mu, max mu и steps."
        )
        return None


def get_simulation_params():
    try:
        mu_ab = float(mu_ab_entry.get())
        mu_ba = float(mu_ba_entry.get())
        seed = int(seed_entry.get())

        if mu_ab < 0 or mu_ba < 0:
            raise ValueError

        return mu_ab, mu_ba, seed

    except ValueError:
        messagebox.showerror(
            "Ошибка",
            "Проверь параметры симуляции: mu_AB, mu_BA и seed."
        )
        return None


def show_table(df):
    for item in table.get_children():
        table.delete(item)

    table["columns"] = list(df.columns)
    table["show"] = "headings"

    for col in df.columns:
        table.heading(col, text=col)
        table.column(col, width=120, anchor="center")

    for _, row in df.iterrows():
        values = []

        for col, value in row.items():
            if col == "stage":
                values.append(int(value))
            elif isinstance(value, (float, np.floating)):
                values.append(round(float(value), 6))
            else:
                values.append(value)

        table.insert("", "end", values=values)


def load_csv():
    global observed_data

    path = filedialog.askopenfilename(
        title="Выбери CSV-файл",
        filetypes=[("CSV files", "*.csv")]
    )

    if not path:
        return

    try:
        df = pd.read_csv(path)

        missing = [col for col in SUMMARY_COLS if col not in df.columns]

        if missing:
            messagebox.showerror("Ошибка", f"В файле нет колонок: {missing}")
            return

        observed_data = df.iloc[0][SUMMARY_COLS].astype(float).to_dict()

        status_label.config(text=f"Загружен файл: {path}")
        result_label.config(text="CSV загружен. Теперь можно нажать 'Подобрать по CSV'.")
        show_table(df)

    except Exception as error:
        messagebox.showerror("Ошибка", f"Не удалось прочитать CSV:\n{error}")


def fit_loaded_csv():
    if observed_data is None:
        messagebox.showwarning("Нет данных", "Сначала загрузи CSV.")
        return

    mu_grid = get_mu_grid()

    if mu_grid is None:
        return

    try:
        equal = fit_equal_mu_refined(
            observed_data,
            mu_grid,
            simulate,
            top_k=3,
            refine_steps=4,
            refine_points=15,
            shrink=0.35,
            seeds=(0, 1, 2),
        )

        best_equal = equal.iloc[0]

        result_label.config(
            text=(
                f"Лучший refined equal подбор: "
                f"mu = {best_equal['mu_AB']:.6f}, "
                f"distance = {best_equal['distance']:.6f}, "
                f"stage = {int(best_equal['stage'])}"
            )
        )

        status_label.config(text="Refined equal подбор выполнен.")
        show_table(equal)

    except Exception as error:
        messagebox.showerror("Ошибка", f"Не удалось подобрать параметры:\n{error}")


def run_simulation():
    global last_simulation_df

    params = get_simulation_params()

    if params is None:
        return

    mu_ab, mu_ba, seed = params

    try:
        sim = simulate(mu_ab, mu_ba, seed)
        last_simulation_df = pd.DataFrame([sim])

        result_label.config(
            text=f"Симуляция готова: mu_AB = {mu_ab:.6f}, mu_BA = {mu_ba:.6f}"
        )

        status_label.config(text="Симуляция выполнена.")
        show_table(last_simulation_df)

    except Exception as error:
        messagebox.showerror("Ошибка", f"Не удалось выполнить симуляцию:\n{error}")


def save_simulation_csv():
    if last_simulation_df is None:
        messagebox.showwarning("Нет симуляции", "Сначала нажми 'Засимулировать'.")
        return

    path = filedialog.asksaveasfilename(
        title="Сохранить CSV",
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")]
    )

    if not path:
        return

    try:
        last_simulation_df.to_csv(path, index=False)
        status_label.config(text=f"CSV сохранён: {path}")

    except Exception as error:
        messagebox.showerror("Ошибка", f"Не удалось сохранить CSV:\n{error}")


def simulate_and_fit():
    global last_simulation_df

    params = get_simulation_params()

    if params is None:
        return

    mu_grid = get_mu_grid()

    if mu_grid is None:
        return

    mu_ab, mu_ba, seed = params

    try:
        obs = simulate(mu_ab, mu_ba, seed)
        last_simulation_df = pd.DataFrame([obs])

        equal = fit_equal_mu_refined(
            obs,
            mu_grid,
            simulate,
            top_k=3,
            refine_steps=4,
            refine_points=15,
            shrink=0.35,
            seeds=(0, 1, 2),
        )

        unequal = fit_unequal_mu(obs, mu_grid, simulate)

        best_equal = equal.iloc[0]
        best_unequal = unequal.iloc[0]

        result_label.config(
            text=(
                f"Истинные: mu_AB = {mu_ab:.6f}, mu_BA = {mu_ba:.6f}. "
                f"Refined equal: mu = {best_equal['mu_AB']:.6f}, "
                f"distance = {best_equal['distance']:.6f}. "
                f"Unequal grid: mu_AB = {best_unequal['mu_AB']:.6f}, "
                f"mu_BA = {best_unequal['mu_BA']:.6f}, "
                f"distance = {best_unequal['distance']:.6f}"
            )
        )

        status_label.config(text="Симуляция и подбор выполнены.")
        show_table(equal)

    except Exception as error:
        messagebox.showerror("Ошибка", f"Не удалось выполнить симуляцию и подбор:\n{error}")


observed_data = None
last_simulation_df = None


root = tk.Tk()
root.title("VGsim Predict")
root.geometry("1200x760")

available_fonts = set(tkfont.families())

if "Noto Sans" in available_fonts:
    font_family = "Noto Sans"
elif "DejaVu Sans" in available_fonts:
    font_family = "DejaVu Sans"
elif "Liberation Sans" in available_fonts:
    font_family = "Liberation Sans"
else:
    font_family = "Arial"

for font_name in (
    "TkDefaultFont",
    "TkTextFont",
    "TkMenuFont",
    "TkHeadingFont",
    "TkCaptionFont",
    "TkSmallCaptionFont",
    "TkIconFont",
    "TkTooltipFont",
):
    try:
        tkfont.nametofont(font_name).configure(family=font_family, size=11)
    except tk.TclError:
        pass


style = ttk.Style()
style.theme_use("clam")

style.configure("TButton", padding=6, font=(font_family, 11))
style.configure("TLabel", padding=3, font=(font_family, 11))
style.configure("TLabelframe.Label", font=(font_family, 11, "bold"))
style.configure("Treeview", font=(font_family, 10), rowheight=26)
style.configure("Treeview.Heading", font=(font_family, 10, "bold"))


settings_frame = ttk.LabelFrame(root, text="Настройки сетки")
settings_frame.pack(fill="x", padx=10, pady=6)

ttk.Label(settings_frame, text="min mu").grid(row=0, column=0, padx=5, pady=5)
mu_min_entry = ttk.Entry(settings_frame, width=12)
mu_min_entry.insert(0, "0.001")
mu_min_entry.grid(row=0, column=1, padx=5, pady=5)

ttk.Label(settings_frame, text="max mu").grid(row=0, column=2, padx=5, pady=5)
mu_max_entry = ttk.Entry(settings_frame, width=12)
mu_max_entry.insert(0, "0.050")
mu_max_entry.grid(row=0, column=3, padx=5, pady=5)

ttk.Label(settings_frame, text="steps").grid(row=0, column=4, padx=5, pady=5)
steps_entry = ttk.Entry(settings_frame, width=12)
steps_entry.insert(0, "10")
steps_entry.grid(row=0, column=5, padx=5, pady=5)


params_frame = ttk.LabelFrame(root, text="Параметры симуляции")
params_frame.pack(fill="x", padx=10, pady=6)

ttk.Label(params_frame, text="mu_AB").grid(row=0, column=0, padx=5, pady=5)
mu_ab_entry = ttk.Entry(params_frame, width=12)
mu_ab_entry.insert(0, "0.020")
mu_ab_entry.grid(row=0, column=1, padx=5, pady=5)

ttk.Label(params_frame, text="mu_BA").grid(row=0, column=2, padx=5, pady=5)
mu_ba_entry = ttk.Entry(params_frame, width=12)
mu_ba_entry.insert(0, "0.050")
mu_ba_entry.grid(row=0, column=3, padx=5, pady=5)

ttk.Label(params_frame, text="seed").grid(row=0, column=4, padx=5, pady=5)
seed_entry = ttk.Entry(params_frame, width=12)
seed_entry.insert(0, "0")
seed_entry.grid(row=0, column=5, padx=5, pady=5)


buttons_frame = ttk.LabelFrame(root, text="Действия")
buttons_frame.pack(fill="x", padx=10, pady=6)

ttk.Button(
    buttons_frame,
    text="Загрузить CSV",
    command=load_csv,
    width=26
).grid(row=0, column=0, padx=5, pady=5)

ttk.Button(
    buttons_frame,
    text="Подобрать по CSV",
    command=fit_loaded_csv,
    width=26
).grid(row=0, column=1, padx=5, pady=5)

ttk.Button(
    buttons_frame,
    text="Засимулировать",
    command=run_simulation,
    width=26
).grid(row=1, column=0, padx=5, pady=5)

ttk.Button(
    buttons_frame,
    text="Сохранить CSV",
    command=save_simulation_csv,
    width=26
).grid(row=1, column=1, padx=5, pady=5)

ttk.Button(
    buttons_frame,
    text="Засимулировать + подобрать",
    command=simulate_and_fit,
    width=56
).grid(row=2, column=0, columnspan=2, padx=5, pady=5)


status_label = ttk.Label(root, text="Готово.", anchor="w")
status_label.pack(fill="x", padx=10, pady=5)

result_label = ttk.Label(
    root,
    text="",
    anchor="w",
    font=(font_family, 11, "bold"),
    wraplength=1150,
    justify="left",
)
result_label.pack(fill="x", padx=10, pady=5)


table_frame = ttk.Frame(root)
table_frame.pack(fill="both", expand=True, padx=10, pady=10)

table = ttk.Treeview(table_frame)
table.grid(row=0, column=0, sticky="nsew")

scrollbar_y = ttk.Scrollbar(table_frame, orient="vertical", command=table.yview)
scrollbar_y.grid(row=0, column=1, sticky="ns")

scrollbar_x = ttk.Scrollbar(table_frame, orient="horizontal", command=table.xview)
scrollbar_x.grid(row=1, column=0, sticky="ew")

table.configure(
    yscrollcommand=scrollbar_y.set,
    xscrollcommand=scrollbar_x.set,
)

table_frame.rowconfigure(0, weight=1)
table_frame.columnconfigure(0, weight=1)


root.mainloop()