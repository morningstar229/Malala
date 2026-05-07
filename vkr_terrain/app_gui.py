"""
Десктоп-приложение: настройка параметров генерации, предпросмотр, экспорт.

Запуск: python -m vkr_terrain.app_gui
Требуется: tkinter (в составе Python), matplotlib, numpy.
"""

from __future__ import annotations

import json
import os
import threading

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError as e:
    raise SystemExit(
        "Не установлен Tkinter (модуль tkinter / _tkinter).\n"
        "На macOS с Homebrew:  brew install python-tk\n"
        "или поставьте Python с python.org — в комплекте есть Tcl/Tk.\n"
        "Пока GUI недоступен, используйте:  python -m vkr_terrain.main --quick"
    ) from e
from typing import Any

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from .params import GenerationParams
from .terrain_3d import run_pipeline_from_params
from .visualize import render_generation_board_on_figure


def _output_dir() -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    d = os.path.join(base, "..", "output")
    return os.path.normpath(d)


class TerrainGeneratorApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Генерация ландшафта (CA Day & Night) — ВКР")
        self.root.minsize(1100, 700)
        self.root.geometry("1500x880")

        self._last_stages: Any = None
        self._last_params: GenerationParams | None = None
        self._build_vars()
        self._build_ui()

    def _build_vars(self) -> None:
        p = GenerationParams()
        self.var_rows = tk.IntVar(value=p.rows)
        self.var_cols = tk.IntVar(value=p.cols)
        self.var_seed = tk.StringVar(value=str(p.seed) if p.seed is not None else "")
        self.var_toroidal = tk.BooleanVar(value=p.toroidal)
        self.var_p_land = tk.DoubleVar(value=p.p_land)
        self.var_coast_it = tk.IntVar(value=p.coastline_iterations)
        self.var_max_layers = tk.IntVar(value=p.max_height_layers)
        self.var_layer_it = tk.IntVar(value=p.height_layer_iterations)
        self.var_noise_p = tk.DoubleVar(value=p.layer_noise_probability)
        self.var_sea_depth = tk.IntVar(value=p.sea_water_depth)
        self.var_voxel_axis = tk.IntVar(value=p.voxel_max_axis)
        self.var_surf_ds = tk.IntVar(value=p.surface_downsample)

    def _read_params(self) -> GenerationParams:
        seed_s = self.var_seed.get().strip()
        seed: int | None
        if not seed_s:
            seed = None
        else:
            try:
                seed = int(seed_s)
            except ValueError:
                raise ValueError("Сид должен быть целым числом или пустым (случайный).") from None

        return GenerationParams(
            rows=self.var_rows.get(),
            cols=self.var_cols.get(),
            seed=seed,
            toroidal=self.var_toroidal.get(),
            p_land=self.var_p_land.get(),
            coastline_iterations=self.var_coast_it.get(),
            max_height_layers=self.var_max_layers.get(),
            height_layer_iterations=self.var_layer_it.get(),
            layer_noise_probability=self.var_noise_p.get(),
            sea_water_depth=self.var_sea_depth.get(),
            voxel_max_axis=self.var_voxel_axis.get(),
            surface_downsample=self.var_surf_ds.get(),
        )

    def _apply_params_to_ui(self, p: GenerationParams) -> None:
        p.clamp()
        self.var_rows.set(p.rows)
        self.var_cols.set(p.cols)
        self.var_seed.set("" if p.seed is None else str(p.seed))
        self.var_toroidal.set(p.toroidal)
        self.var_p_land.set(p.p_land)
        self.var_coast_it.set(p.coastline_iterations)
        self.var_max_layers.set(p.max_height_layers)
        self.var_layer_it.set(p.height_layer_iterations)
        self.var_noise_p.set(p.layer_noise_probability)
        self.var_sea_depth.set(p.sea_water_depth)
        self.var_voxel_axis.set(p.voxel_max_axis)
        self.var_surf_ds.set(p.surface_downsample)

    def _build_ui(self) -> None:
        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)

        left_outer = ttk.Frame(main, width=340)
        main.add(left_outer, weight=0)

        canvas = tk.Canvas(left_outer, highlightthickness=0)
        scroll = ttk.Scrollbar(left_outer, orient=tk.VERTICAL, command=canvas.yview)
        left = ttk.Frame(canvas)
        left.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=left, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        def _on_mousewheel(event: tk.Event) -> str | None:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            return None

        for w in (canvas, left):
            w.bind("<MouseWheel>", _on_mousewheel)

        right = ttk.Frame(main)
        main.add(right, weight=1)

        self._panel_params(left)
        self._panel_plot(right)

        self.status = ttk.Label(self.root, text="Нажмите «Сгенерировать»", relief=tk.SUNKEN)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _panel_params(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Параметры генерации", font=("", 12, "bold")).pack(
            anchor=tk.W, padx=8, pady=(8, 4)
        )

        f_grid = ttk.LabelFrame(parent, text="Сетка и случайность", padding=6)
        f_grid.pack(fill=tk.X, padx=6, pady=4)
        self._row_spin(f_grid, "Строк (Y):", self.var_rows, 8, 512)
        self._row_spin(f_grid, "Столбцов (X):", self.var_cols, 8, 512)
        self._row_entry(f_grid, "Сид (пусто = случайный):", self.var_seed)
        ttk.Checkbutton(f_grid, text="Тороидальные границы (склейка краёв)", variable=self.var_toroidal).pack(
            anchor=tk.W, pady=4
        )

        f_ca = ttk.LabelFrame(parent, text="Этапы 1–2: шум и берег (Day & Night)", padding=6)
        f_ca.pack(fill=tk.X, padx=6, pady=4)
        self._row_scale(f_ca, "Вероятность суши в шуме:", self.var_p_land, 0.05, 0.95)
        self._row_spin(f_ca, "Итераций CA для берега:", self.var_coast_it, 0, 200)

        f_h = ttk.LabelFrame(parent, text="Этап 3: слои высоты", padding=6)
        f_h.pack(fill=tk.X, padx=6, pady=4)
        self._row_spin(f_h, "Число слоёв высоты:", self.var_max_layers, 1, 200)
        self._row_spin(f_h, "Итераций CA на слой:", self.var_layer_it, 1, 50)
        self._row_scale(f_h, "Порог шума на слой (0–1):", self.var_noise_p, 0.01, 0.99)

        f_w = ttk.LabelFrame(parent, text="Вода и визуализация", padding=6)
        f_w.pack(fill=tk.X, padx=6, pady=4)
        self._row_spin(f_w, "Глубина воды (вокселей в море):", self.var_sea_depth, 0, 64)
        self._row_spin(f_w, "Макс. размер воксельного превью (ось):", self.var_voxel_axis, 12, 128)
        self._row_spin(f_w, "Прореживание для 3D-поверхности:", self.var_surf_ds, 1, 16)

        bf = ttk.Frame(parent)
        bf.pack(fill=tk.X, padx=6, pady=8)
        self.btn_gen = ttk.Button(bf, text="Сгенерировать", command=self._generate_async)
        self.btn_gen.pack(fill=tk.X, pady=2)
        ttk.Button(bf, text="Новый случайный сид", command=self._random_seed).pack(fill=tk.X, pady=2)
        ttk.Button(bf, text="Сохранить PNG…", command=self._save_png).pack(fill=tk.X, pady=2)

        fm = ttk.LabelFrame(parent, text="Пресеты", padding=6)
        fm.pack(fill=tk.X, padx=6, pady=4)
        ttk.Button(fm, text="Экспорт параметров (JSON)…", command=self._export_json).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(fm, text="Импорт параметров (JSON)…", command=self._import_json).pack(
            fill=tk.X, pady=2
        )

        ttk.Label(
            parent,
            text="Подсказка: после генерации крутите 3D мышью\n(панель инструментов снизу).",
            foreground="gray",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, padx=8, pady=(4, 12))

    def _row_spin(
        self,
        parent: ttk.Frame,
        label: str,
        var: tk.IntVar,
        vmin: int,
        vmax: int,
    ) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=28).pack(side=tk.LEFT)
        sp = ttk.Spinbox(
            row,
            textvariable=var,
            from_=vmin,
            to=vmax,
            width=12,
            command=lambda: self._clamp_spin(var, vmin, vmax),
        )
        sp.pack(side=tk.RIGHT)

    def _row_entry(self, parent: ttk.Frame, label: str, var: tk.StringVar) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=28).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=var, width=14).pack(side=tk.RIGHT)

    def _row_scale(
        self,
        parent: ttk.Frame,
        label: str,
        var: tk.DoubleVar,
        vmin: float,
        vmax: float,
    ) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=28).pack(side=tk.LEFT)
        sc = ttk.Scale(
            row,
            from_=vmin,
            to=vmax,
            variable=var,
            command=lambda _: self.status.config(text=f"{label} {var.get():.3f}"),
        )
        sc.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Label(row, textvariable=var, width=6).pack(side=tk.RIGHT)

    def _clamp_spin(self, var: tk.IntVar, vmin: int, vmax: int) -> None:
        try:
            v = int(var.get())
            var.set(max(vmin, min(vmax, v)))
        except tk.TclError:
            pass

    def _panel_plot(self, parent: ttk.Frame) -> None:
        self.fig = Figure(figsize=(11, 7.2), dpi=100)
        self.fig.patch.set_facecolor("#f5f5f5")

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.draw()

        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        placeholder = (
            "Здесь появится предпросмотр после генерации.\n"
            "Настройте параметры слева и нажмите «Сгенерировать»."
        )
        self.fig.text(0.5, 0.5, placeholder, ha="center", va="center", fontsize=12, color="gray")

    def _random_seed(self) -> None:
        import random

        self.var_seed.set(str(random.randint(0, 2**31 - 1)))

    def _generate_async(self) -> None:
        try:
            params_main = self._read_params()
            params_main.clamp()
        except ValueError as e:
            messagebox.showerror("Параметры", str(e))
            return

        self.btn_gen.config(state=tk.DISABLED)
        self.status.config(text="Идёт генерация…")

        def work(p: GenerationParams) -> None:
            err: list[str | None] = [None]
            result: list[Any] = [None]

            try:
                result[0] = run_pipeline_from_params(p)
            except Exception as e:
                err[0] = str(e)

            def done_ui() -> None:
                self.btn_gen.config(state=tk.NORMAL)
                if err[0]:
                    messagebox.showerror("Ошибка генерации", err[0])
                    self.status.config(text="Ошибка.")
                    return
                self._last_stages = result[0]
                self._last_params = p
                render_generation_board_on_figure(
                    self.fig,
                    self._last_stages,
                    voxel_max_axis=p.voxel_max_axis,
                    surface_downsample=p.surface_downsample,
                )
                self.canvas.draw_idle()
                self.status.config(text="Готово. Можно сохранить PNG.")

            self.root.after(0, done_ui)

        threading.Thread(target=work, args=(params_main,), daemon=True).start()

    def _save_png(self) -> None:
        if self._last_stages is None:
            messagebox.showinfo("Нет данных", "Сначала выполните генерацию.")
            return
        os.makedirs(_output_dir(), exist_ok=True)
        path = filedialog.asksaveasfilename(
            title="Сохранить изображение",
            initialdir=_output_dir(),
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("Все файлы", "*.*")],
        )
        if not path:
            return
        try:
            self.fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=self.fig.get_facecolor())
            self.status.config(text=f"Сохранено: {path}")
        except OSError as e:
            messagebox.showerror("Ошибка записи", str(e))

    def _export_json(self) -> None:
        try:
            p = self._read_params()
            p.clamp()
        except ValueError as e:
            messagebox.showerror("Ошибка", str(e))
            return
        d = p.to_dict()
        # JSON не любит None в ключах — сериализуем seed явно
        d["seed"] = d["seed"]
        path = filedialog.asksaveasfilename(
            title="Экспорт параметров",
            initialdir=_output_dir(),
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, indent=2)
            self.status.config(text=f"Параметры сохранены: {path}")
        except OSError as e:
            messagebox.showerror("Ошибка", str(e))

    def _import_json(self) -> None:
        path = filedialog.askopenfilename(
            title="Импорт параметров",
            initialdir=_output_dir(),
            filetypes=[("JSON", "*.json"), ("Все файлы", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            messagebox.showerror("Ошибка файла", str(e))
            return
        try:
            p = GenerationParams.from_dict({str(k): v for k, v in d.items()})
        except (TypeError, ValueError) as e:
            messagebox.showerror("Неверный формат", str(e))
            return
        self._apply_params_to_ui(p)
        self.status.config(text="Параметры загружены из JSON.")

    def _on_close(self) -> None:
        self.root.quit()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    TerrainGeneratorApp().run()


if __name__ == "__main__":
    main()
