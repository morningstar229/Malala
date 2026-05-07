"""
Десктоп-приложение (Qt 6): полноценный интерфейс генерации ландшафта.

Меню, горячие клавиши, фоновая генерация, сохранение окна и параметров,
встроенный Matplotlib.

Запуск: python -m vkr_terrain.desktop_app
Зависимость: pip install PySide6
"""

from __future__ import annotations

import csv
from datetime import datetime
import json
import os
import sys
from typing import Any

# Matplotlib backend до импорта pyplot
import matplotlib

matplotlib.use("QtAgg")

try:
    from PySide6.QtCore import QSettings, QSize, Qt, QThread, Signal, QUrl
except ImportError as e:
    raise SystemExit(
        "Нужен PySide6:  pip install PySide6\n"
        "Или: pip install -r requirements.txt"
    ) from e

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtGui import QAction, QCloseEvent, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .experiments import run_all_experiments
import numpy as np

from .mesh_export import (
    DEFAULT_RELIEF_FRACTION,
    anchor_sea_level,
    build_smooth_height_surface,
    scale_relief_to_map_extent,
    snap_land_z_to_triangulated_mesh_floor,
)
from .params import GenerationParams
from .terrain_3d import run_pipeline_from_params
from .visualize import render_generation_board_on_figure

APP_ORG = "SSTU"
APP_NAME = "VKRTerrainGenerator"
APP_TITLE = "Генератор ландшафта — ВКР"
APP_VERSION = "1.0.0"


def _output_dir() -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(base, "..", "output"))


class GenerateThread(QThread):
    """Фоновая генерация без блокировки UI."""

    finished_ok = Signal(object, object)
    failed = Signal(str)

    def __init__(self, params: GenerationParams) -> None:
        super().__init__()
        self._params = params

    def run(self) -> None:
        try:
            stages = run_pipeline_from_params(self._params)
            self.finished_ok.emit(stages, self._params)
        except Exception as e:
            self.failed.emit(str(e))


class ResearchThread(QThread):
    """Фоновый запуск серий экспериментов ЛР2/ЛР3/ЛР4."""

    finished_ok = Signal(str)
    failed = Signal(str)

    def run(self) -> None:
        try:
            out = run_all_experiments()
            self.finished_ok.emit(out)
        except Exception as e:
            self.failed.emit(str(e))


class CandidateBatchThread(QThread):
    """Фоновая генерация всех кандидатов для визуального сравнения."""

    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, params_list: list[GenerationParams]) -> None:
        super().__init__()
        self._params_list = params_list

    def run(self) -> None:
        try:
            out: list[tuple[GenerationParams, object]] = []
            for p in self._params_list:
                stages = run_pipeline_from_params(p)
                out.append((p, stages))
            self.finished_ok.emit(out)
        except Exception as e:
            self.failed.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.setMinimumSize(QSize(1180, 760))
        self.resize(1520, 900)

        self._settings = QSettings(APP_ORG, APP_NAME)
        self._thread: GenerateThread | None = None
        self._research_thread: ResearchThread | None = None
        self._cand_batch_thread: CandidateBatchThread | None = None
        self._candidates: list[dict[str, object]] = []
        self._compare_image_path: str | None = None
        self._ui_surface_Z: np.ndarray | None = None
        self._ui_surface_mask: np.ndarray | None = None
        self._ui_surface_scale_xy: float = 1.0
        self._last_stages: Any = None
        self._last_params: GenerationParams | None = None

        self._build_controls()
        self._build_canvas()
        self._build_menu()
        self._build_status()

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.tabs)
        splitter.addWidget(self._plot_frame)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([520, 980])
        layout.addWidget(splitter)

        self._restore_state()
        self._wire_actions()
        self._refresh_research_results()

    def _build_controls(self) -> None:
        p = GenerationParams()
        if self._settings.contains("params_json"):
            try:
                raw = self._settings.value("params_json", "")
                if isinstance(raw, str) and raw.strip():
                    p = GenerationParams.from_dict(json.loads(raw))
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        self.sp_rows = QSpinBox()
        self.sp_rows.setRange(8, 512)
        self.sp_rows.setValue(p.rows)

        self.sp_cols = QSpinBox()
        self.sp_cols.setRange(8, 512)
        self.sp_cols.setValue(p.cols)

        self.ed_seed = QLineEdit()
        self.ed_seed.setPlaceholderText("пусто = случайный")
        self.ed_seed.setText("" if p.seed is None else str(p.seed))

        self.cb_torus = QCheckBox("Тороидальные границы (склейка краёв)")
        self.cb_torus.setChecked(p.toroidal)

        self.combo_model = QComboBox()
        self.combo_model.addItem("Убывающая высота + CA по слоям (tapered_height)", "tapered_height")
        self.combo_model.setEnabled(False)
        idx = self.combo_model.findData(p.height_model)
        self.combo_model.setCurrentIndex(idx if idx >= 0 else 0)

        self.dsp_p_land = QDoubleSpinBox()
        self.dsp_p_land.setRange(0.05, 0.95)
        self.dsp_p_land.setSingleStep(0.01)
        self.dsp_p_land.setDecimals(3)
        self.dsp_p_land.setValue(p.p_land)

        self.sp_coast = QSpinBox()
        self.sp_coast.setRange(0, 200)
        self.sp_coast.setValue(p.coastline_iterations)

        self.sp_layers = QSpinBox()
        self.sp_layers.setRange(1, 200)
        self.sp_layers.setValue(p.max_height_layers)

        self.sp_layer_it = QSpinBox()
        self.sp_layer_it.setRange(1, 50)
        self.sp_layer_it.setValue(p.height_layer_iterations)

        self.dsp_noise = QDoubleSpinBox()
        self.dsp_noise.setRange(0.01, 0.99)
        self.dsp_noise.setSingleStep(0.01)
        self.dsp_noise.setDecimals(3)
        self.dsp_noise.setValue(p.layer_noise_probability)

        self.sp_sea = QSpinBox()
        self.sp_sea.setRange(0, 64)
        self.sp_sea.setValue(p.sea_water_depth)

        self.sp_vox = QSpinBox()
        self.sp_vox.setRange(12, 128)
        self.sp_vox.setValue(p.voxel_max_axis)

        self.sp_ds = QSpinBox()
        self.sp_ds.setRange(1, 16)
        self.sp_ds.setValue(p.surface_downsample)

        form = QVBoxLayout()
        form.setSpacing(10)

        g0 = QGroupBox("Сетка и случайность")
        fl0 = QFormLayout(g0)
        fl0.addRow("Строк (Y):", self.sp_rows)
        fl0.addRow("Столбцов (X):", self.sp_cols)
        fl0.addRow("Сид:", self.ed_seed)
        fl0.addRow(self.cb_torus)
        fl0.addRow("Модель рельефа:", self.combo_model)
        form.addWidget(g0)

        g1 = QGroupBox("Этапы 1–2: шум и берег (Day & Night)")
        fl1 = QFormLayout(g1)
        fl1.addRow("Вероятность суши:", self.dsp_p_land)
        fl1.addRow("Итераций CA (берег):", self.sp_coast)
        form.addWidget(g1)

        g2 = QGroupBox("Этап 3: слои высоты")
        fl2 = QFormLayout(g2)
        fl2.addRow("Число слоёв:", self.sp_layers)
        fl2.addRow("Итераций CA на слой:", self.sp_layer_it)
        fl2.addRow("Порог шума на слой:", self.dsp_noise)
        form.addWidget(g2)

        g3 = QGroupBox("Вода и отображение")
        fl3 = QFormLayout(g3)
        fl3.addRow("Глубина воды (вокс.):", self.sp_sea)
        fl3.addRow("Превью вокселей (ось):", self.sp_vox)
        fl3.addRow("Прореж. 3D-поверхности:", self.sp_ds)
        form.addWidget(g3)

        btn_row = QHBoxLayout()
        self.btn_gen = QPushButton("Сгенерировать")
        self.btn_gen.setDefault(True)
        self.btn_gen.setMinimumHeight(36)
        self.btn_rand = QPushButton("Случайный сид")
        btn_row.addWidget(self.btn_gen)
        btn_row.addWidget(self.btn_rand)
        form.addLayout(btn_row)

        self.btn_blender = QPushButton("Сохранить OBJ для Blender…")
        self.btn_blender.setMinimumHeight(34)
        self.btn_blender.setToolTip(
            "Сетка с нормалями — откройте файл в Blender: File → Import → Wavefront (.obj)"
        )
        self.btn_blender.clicked.connect(self._export_obj)
        form.addWidget(self.btn_blender)

        hint = QLabel(
            "Ctrl+G — генерация · Ctrl+S — PNG · кнопка выше или Файл — OBJ для Blender\n"
            "Параметры и размер окна сохраняются автоматически."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: palette(mid); font-size: 11px;")
        form.addWidget(hint)

        gen_wrap = QWidget()
        gen_wrap.setLayout(form)
        gen_wrap.setMinimumWidth(320)

        exp_wrap = QWidget()
        exp_layout = QVBoxLayout(exp_wrap)
        exp_layout.setSpacing(10)

        exp_method = QTextEdit()
        exp_method.setReadOnly(True)
        exp_method.setMaximumHeight(160)
        exp_method.setPlainText(
            "Как считаются эксперименты:\n"
            "ЛР2: для плотностей 10/30/50/70% запускаются серии CA, ищется шаг входа в цикл,\n"
            "период цикла, число blobs (связных компонент) и длина границы.\n"
            "ЛР2 (симметрия): сравнивается эволюция конфигурации и её инверсии.\n"
            "ЛР3: sweep по sea_probability и iterations_base, измеряются связность и берег.\n"
            "ЛР4: sweep по параметрам 3D, считаются mean/std/max высоты, roughness, доля суши."
        )
        exp_layout.addWidget(exp_method)

        exp_intro = QLabel(
            "Автоматический прогон исследований под ЛР2/ЛР3/ЛР4.\n"
            "Результаты сохраняются в output/research в формате CSV."
        )
        exp_intro.setWordWrap(True)
        exp_layout.addWidget(exp_intro)

        self.btn_run_exp = QPushButton("Запустить эксперименты")
        self.btn_run_exp.setMinimumHeight(36)
        self.btn_open_research = QPushButton("Открыть output/research")
        self.btn_run_exp.setToolTip("Шаг 1: собрать данные ЛР2/ЛР3/ЛР4 и сформировать кандидатов.")
        self.btn_open_research.setToolTip("Открыть исходные CSV/JSON файлы исследований.")
        exp_layout.addWidget(self.btn_run_exp)
        exp_layout.addWidget(self.btn_open_research)

        self.pb_exp = QProgressBar()
        self.pb_exp.setRange(0, 0)
        self.pb_exp.hide()
        exp_layout.addWidget(self.pb_exp)

        self.lbl_exp_status = QLabel("Статус: эксперименты ещё не запускались.")
        self.lbl_exp_status.setWordWrap(True)
        self.lbl_exp_status.setStyleSheet("color: palette(mid);")
        exp_layout.addWidget(self.lbl_exp_status)

        self.btn_refresh_results = QPushButton("Обновить из CSV")
        exp_layout.addWidget(self.btn_refresh_results)

        self.txt_exp_report = QTextEdit()
        self.txt_exp_report.setReadOnly(True)
        self.txt_exp_report.setPlaceholderText("После запуска здесь появится сводка по исследованиям.")
        self.txt_exp_report.setMinimumHeight(180)
        exp_layout.addWidget(self.txt_exp_report)

        csv_lbl = QLabel("Таблица CSV-исследований:")
        csv_lbl.setStyleSheet("font-weight: 600;")
        exp_layout.addWidget(csv_lbl)
        self.combo_results_file = QComboBox()
        self.combo_results_file.addItem("lab2_random_summary.csv")
        self.combo_results_file.addItem("lab2_symmetry.csv")
        self.combo_results_file.addItem("lab3_landscape_sweep.csv")
        self.combo_results_file.addItem("lab4_3d_relief_sweep.csv")
        exp_layout.addWidget(self.combo_results_file)

        self.table_results = QTableWidget(0, 0)
        self.table_results.setMinimumHeight(220)
        exp_layout.addWidget(self.table_results)

        cand_title = QLabel("Кандидаты параметров (топ-5):")
        cand_title.setStyleSheet("font-weight: 600;")
        exp_layout.addWidget(cand_title)

        self.combo_candidates = QComboBox()
        self.combo_candidates.setToolTip("Шаг 2: выбери один из топ-5 кандидатов алгоритма.")
        exp_layout.addWidget(self.combo_candidates)
        self.lbl_candidate_desc = QLabel("Нет загруженных кандидатов. Сначала запустите эксперименты.")
        self.lbl_candidate_desc.setWordWrap(True)
        self.lbl_candidate_desc.setStyleSheet("color: palette(mid);")
        exp_layout.addWidget(self.lbl_candidate_desc)

        cand_btn_grid = QGridLayout()
        self.btn_pick_best = QPushButton("Автовыбор")
        self.btn_apply_show_candidate = QPushButton("Применить и показать")
        self.btn_show_all_candidates = QPushButton("Сравнить 5")
        self.btn_mark_manual = QPushButton("Подтвердить выбор (JSON)")
        self.btn_pick_best.setToolTip("Автоматически берёт Кандидат #1 (лучший по метрикам).")
        self.btn_apply_show_candidate.setToolTip("Шаг 3: сразу запускает генерацию для визуальной проверки.")
        self.btn_show_all_candidates.setToolTip(
            "Одним кликом генерирует все 5 кандидатов и рисует их рядом для сравнения."
        )
        self.btn_mark_manual.setToolTip("Шаг 4: фиксирует итоговый выбор в output/research/manual_selected_candidate.json.")
        cand_btn_grid.addWidget(self.btn_pick_best, 0, 0)
        cand_btn_grid.addWidget(self.btn_apply_show_candidate, 0, 1, 1, 2)
        cand_btn_grid.addWidget(self.btn_show_all_candidates, 1, 0)
        cand_btn_grid.addWidget(self.btn_mark_manual, 1, 1, 1, 2)
        cand_btn_grid.setHorizontalSpacing(6)
        cand_btn_grid.setVerticalSpacing(6)
        exp_layout.addLayout(cand_btn_grid)

        for b in (
            self.btn_run_exp,
            self.btn_open_research,
            self.btn_refresh_results,
            self.btn_pick_best,
            self.btn_apply_show_candidate,
            self.btn_show_all_candidates,
            self.btn_mark_manual,
        ):
            b.setMinimumHeight(32)
            b.setSizePolicy(b.sizePolicy().horizontalPolicy(), b.sizePolicy().verticalPolicy())

        guide = QTextEdit()
        guide.setReadOnly(True)
        guide.setMaximumHeight(110)
        guide.setPlainText(
            "Схема: запуск серии → CSV в output/research → кнопка «Обновить из CSV» → топ-5 кандидатов.\n"
            "Кандидат — набор чисел (сид, берег, слои…), не готовая сцена.\n"
            "Итог в Blender: «Применить и показать» → «Экспорт OBJ» или JSON из «Подтвердить выбор…».\n"
            "Полное руководство — вкладка «Справка» и файл README.md в папке проекта."
        )
        exp_layout.addWidget(guide)

        exp_files_hint = QLabel(
            "CSV формируются автоматически, но теперь их содержание видно прямо в таблице выше."
        )
        exp_files_hint.setStyleSheet("color: palette(mid); font-size: 11px;")
        exp_layout.addWidget(exp_files_hint)
        exp_layout.addStretch(1)

        help_wrap = QWidget()
        help_layout = QVBoxLayout(help_wrap)
        help_layout.setSpacing(8)
        help_title = QLabel("Справка по программе")
        help_title.setStyleSheet("font-weight: 600;")
        help_layout.addWidget(help_title)
        help_note = QLabel(
            "Полная инструкция установки и сдачи проекта находится в файле <b>README.md</b> "
            "в корне папки приложения."
        )
        help_note.setWordWrap(True)
        help_layout.addWidget(help_note)
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setMinimumHeight(340)
        help_text.setPlainText(
            "НАЗНАЧЕНИЕ\n"
            "Генерация процедурного ландшафта: случайное заполнение сушей/морем → автомат береговой линии "
            "(правило Day & Night) → трёхмерный рельеф (режим tapered_height). Предпросмотр всех шагов слева, "
            "итоговая картинка справа. Экспорт в Blender сохраняет ту же численную поверхность после сглаживания, "
            "плюс плоский объект воды.\n\n"
            "ЗАПУСК\n"
            "Откройте терминал в папке проекта. Один раз: python3 -m venv .venv → активировать → "
            "pip install -r requirements.txt. Затем запуск приложения:\n"
            "• macOS: двойной щелчок run.command или ./run.sh\n"
            "• Windows: run.bat после активации .venv\\Scripts\\activate\n"
            "• Универсально: python -m vkr_terrain (или python -m vkr_terrain.desktop_app)\n\n"
            "ВКЛАДКА «ГЕНЕРАЦИЯ»\n"
            "Задаются размеры сетки, сид измерений, тор, доля первичной суши p_land, число шагов уточнения берега, "
            "число слоёв рельефа и итераций на слой, прореживание поверхности (влияет на скорость и деталь 3D), "
            "ограничения воксельного блока статистики. «Сгенерировать» или Ctrl+G запускает расчёт в фоновом потоке — "
            "дождаться сообщения «Готово». Без генерации экспорт OBJ недоступен.\n"
            "После генерации можно: сохранить PNG (Ctrl+S), сохранить параметры JSON, загрузить JSON и снова сгенерировать.\n\n"
            "ЭКСПОРТ В BLENDER\n"
            "«Экспорт mesh для Blender (OBJ)…» или Ctrl+Shift+O. Сохраняются .obj и .mtl (Land/Sea). Ориентация XY "
            "совпадает с сеткой приложения (строка карты → +Y, столбец → +X). Уровень воды — горизонт чуть ниже z=0, "
            "как нуль рельефа после привязки. В Blender: Wavefront OBJ, Material Preview; если карта кажется зеркальной, "
            "смените оси в диалоге импорта или поверните объект.\n\n"
            "ВКЛАДКА «ЭКСПЕРИМЕНТЫ»\n"
            "Кнопка запускает заранее заданные серии расчётов (ЛР2–ЛР4) и сохраняет результаты в output/research/ "
            "(CSV и краткая сводка). «Обновить из CSV» перечитывает файлы и строит список кандидатов — до пяти наборов "
            "рекомендованных параметров на основе метрик (состав комбинируется из результатов таблиц; это очередь "
            "к проверке глазами, а не финальный диагноз). «Применить и показать результат» загружает выбранный кандидат "
            "и запускает полную генерацию. «Подтвердить выбор вручную (JSON)» фиксирует выбор для отчёта. При необходимости "
            "сравните сразу пять штук («Сгенерировать и сравнить всех кандидатов»).\n\n"
            "ГОРЯЧИЕ КЛАВИШИ И МЕНЮ\n"
            "Ctrl+G — генерация • Ctrl+S — PNG • Ctrl+Shift+O — OBJ • Файл: эксперименты, импорт/экспорт JSON • "
            "Справка → О программе.\n\n"
            "ФАЙЛЫ ПРИ РАБОТЕ\n"
            "Рядом с проектом появится output/ — PNG скрины, объект экспорта, папка research/ после экспериментов.\n\n"
            "КОМАНДНАЯ СТРОКА (без главного окна)\n"
            "python -m vkr_terrain.main research — только эксперименты;\n"
            "python -m vkr_terrain.main --quick — одна генерация и generation_full.png в output/;\n"
            "python -m vkr_terrain.main lab — отдельная лаборатория CA, нужен pygame."
        )
        help_layout.addWidget(help_text, stretch=1)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._wrap_tab_scroll(gen_wrap), "Генерация")
        self.tabs.addTab(self._wrap_tab_scroll(exp_wrap), "Эксперименты")
        self.tabs.addTab(self._wrap_tab_scroll(help_wrap), "Справка")
        self.tabs.setMinimumWidth(440)
        self.tabs.setMaximumWidth(640)

        self._plot_frame = QFrame()

    def _wrap_tab_scroll(self, content: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.Shape.NoFrame)
        area.setWidget(content)
        return area

    def _build_canvas(self) -> None:
        self.fig = Figure(figsize=(11, 7.2), dpi=100)
        self.fig.patch.set_facecolor("#f4f4f5")

        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self._plot_frame)

        plot_layout = QVBoxLayout(self._plot_frame)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        cap = QLabel(
            "Окно ниже — предпросмотр (Matplotlib). «Нормальное» затенённое 3D: "
            "сохраните OBJ кнопкой слева и импортируйте в Blender."
        )
        cap.setWordWrap(True)
        cap.setStyleSheet("padding: 6px 4px; color: palette(text); background: palette(base);")
        plot_layout.addWidget(cap)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)

        self.fig.text(
            0.5,
            0.5,
            "Настройте параметры слева и нажмите «Сгенерировать»\nили Ctrl+G",
            ha="center",
            va="center",
            fontsize=12,
            color="gray",
        )
        self.canvas.draw()

    def _build_menu(self) -> None:
        mb = self.menuBar()

        m_file = mb.addMenu("Файл")
        self.act_gen = QAction("Сгенерировать", self)
        self.act_gen.setShortcut(QKeySequence("Ctrl+G"))
        self.act_gen.triggered.connect(self._start_generate)
        m_file.addAction(self.act_gen)

        self.act_save = QAction("Сохранить PNG…", self)
        self.act_save.setShortcut(QKeySequence.StandardKey.Save)
        self.act_save.triggered.connect(self._save_png)
        m_file.addAction(self.act_save)

        self.act_obj = QAction("Экспорт mesh для Blender (OBJ)…", self)
        self.act_obj.setShortcut(QKeySequence("Ctrl+Shift+O"))
        self.act_obj.triggered.connect(self._export_obj)
        m_file.addAction(self.act_obj)

        self.act_research = QAction("Запустить эксперименты ЛР2–ЛР4…", self)
        self.act_research.triggered.connect(self._start_research)
        m_file.addAction(self.act_research)

        m_file.addSeparator()
        imp = QAction("Импорт параметров (JSON)…", self)
        imp.triggered.connect(self._import_json)
        m_file.addAction(imp)
        exp = QAction("Экспорт параметров (JSON)…", self)
        exp.triggered.connect(self._export_json)
        m_file.addAction(exp)

        m_file.addSeparator()
        q = QAction("Выход", self)
        q.setShortcut(QKeySequence.StandardKey.Quit)
        q.triggered.connect(self.close)
        m_file.addAction(q)

        m_help = mb.addMenu("Справка")
        about = QAction("О программе…", self)
        about.triggered.connect(self._about)
        m_help.addAction(about)

    def _build_status(self) -> None:
        self.statusBar().showMessage("Готово")

    def _wire_actions(self) -> None:
        self.btn_gen.clicked.connect(self._start_generate)
        self.btn_rand.clicked.connect(self._random_seed)
        self.btn_run_exp.clicked.connect(self._start_research)
        self.btn_open_research.clicked.connect(self._open_research_dir)
        self.btn_refresh_results.clicked.connect(self._refresh_research_results)
        self.combo_results_file.currentTextChanged.connect(self._refresh_results_table)
        self.combo_candidates.currentIndexChanged.connect(self._on_candidate_changed)
        self.btn_pick_best.clicked.connect(self._pick_best_candidate)
        self.btn_apply_show_candidate.clicked.connect(self._apply_and_generate_selected_candidate)
        self.btn_show_all_candidates.clicked.connect(self._generate_and_compare_all_candidates)
        self.btn_mark_manual.clicked.connect(self._mark_manual_selection)
        self.tabs.currentChanged.connect(self._on_tab_changed)

    def _on_tab_changed(self, idx: int) -> None:
        # На вкладке «Эксперименты» всегда подхватываем свежие CSV и список кандидатов.
        if idx == 1:
            self._refresh_research_results()

    def _read_params(self) -> GenerationParams:
        seed_s = self.ed_seed.text().strip()
        seed: int | None = None
        if seed_s:
            try:
                seed = int(seed_s)
            except ValueError as e:
                raise ValueError("Сид: целое число или пусто (случайный).") from e
        return GenerationParams(
            rows=self.sp_rows.value(),
            cols=self.sp_cols.value(),
            seed=seed,
            toroidal=self.cb_torus.isChecked(),
            p_land=self.dsp_p_land.value(),
            coastline_iterations=self.sp_coast.value(),
            max_height_layers=self.sp_layers.value(),
            height_layer_iterations=self.sp_layer_it.value(),
            layer_noise_probability=self.dsp_noise.value(),
            sea_water_depth=self.sp_sea.value(),
            voxel_max_axis=self.sp_vox.value(),
            surface_downsample=self.sp_ds.value(),
            height_model="tapered_height",
        )

    def _candidate_dict_to_params_dict(self, c: dict[str, object]) -> dict[str, object]:
        """Преобразовать short candidate JSON в полный preset для стабильного импорта."""
        base = GenerationParams().to_dict()
        mapping = {
            "seed": "seed",
            "p_land": "p_land",
            "coastline_iterations": "coastline_iterations",
            "height_model": "height_model",
            "max_height_layers": "max_height_layers",
            "height_layer_iterations": "height_layer_iterations",
        }
        for src, dst in mapping.items():
            if src in c:
                base[dst] = c[src]
        return base

    def _apply_params(self, p: GenerationParams) -> None:
        p.clamp()
        self.sp_rows.setValue(p.rows)
        self.sp_cols.setValue(p.cols)
        self.ed_seed.setText("" if p.seed is None else str(p.seed))
        self.cb_torus.setChecked(p.toroidal)
        self.dsp_p_land.setValue(p.p_land)
        self.sp_coast.setValue(p.coastline_iterations)
        self.sp_layers.setValue(p.max_height_layers)
        self.sp_layer_it.setValue(p.height_layer_iterations)
        self.dsp_noise.setValue(p.layer_noise_probability)
        self.sp_sea.setValue(p.sea_water_depth)
        self.sp_vox.setValue(p.voxel_max_axis)
        self.sp_ds.setValue(p.surface_downsample)
        idx = self.combo_model.findData(p.height_model)
        self.combo_model.setCurrentIndex(idx if idx >= 0 else 0)

    def _random_seed(self) -> None:
        import random

        self.ed_seed.setText(str(random.randint(0, 2**31 - 1)))

    def _worker_busy(self) -> bool:
        """True, если фоновый поток ещё работает (без обращения к удалённому QThread)."""
        t = self._thread
        if t is None:
            return False
        try:
            return t.isRunning()
        except RuntimeError:
            self._thread = None
            return False

    def _on_thread_finished(self) -> None:
        """После завершения QThread сбрасываем ссылку (иначе libshiboken ругается на deleteLater)."""
        self._thread = None

    def _on_research_finished(self) -> None:
        self._research_thread = None

    def _set_research_controls(self, enabled: bool) -> None:
        self.btn_run_exp.setEnabled(enabled)
        self.btn_open_research.setEnabled(enabled)
        self.act_research.setEnabled(enabled)
        self.pb_exp.setVisible(not enabled)

    def _start_generate(self) -> None:
        if self._worker_busy():
            return
        try:
            p = self._read_params()
            p.clamp()
        except ValueError as e:
            QMessageBox.warning(self, "Параметры", str(e))
            return

        self.btn_gen.setEnabled(False)
        self.act_gen.setEnabled(False)
        self.statusBar().showMessage("Идёт генерация…")

        thr = GenerateThread(p)
        self._thread = thr
        thr.finished_ok.connect(self._on_generated)
        thr.failed.connect(self._on_failed)
        thr.finished.connect(self._on_thread_finished)
        thr.start()

    def _on_generated(self, stages: object, params: GenerationParams) -> None:
        self._last_stages = stages
        self._last_params = params
        self.fig.clear()
        render_generation_board_on_figure(
            self.fig,
            stages,
            voxel_max_axis=params.voxel_max_axis,
            surface_downsample=params.surface_downsample,
        )
        # 1-в-1 поверхность для экспорта: та же сетка и те же значения, что и в UI plot_surface.
        z_smooth = build_smooth_height_surface(stages.heights, stages.land_mask)
        z_smooth = scale_relief_to_map_extent(
            z_smooth, stages.land_mask, scale_xy=1.0, relief_fraction=DEFAULT_RELIEF_FRACTION
        )
        step = max(1, int(params.surface_downsample))
        z_smooth = anchor_sea_level(z_smooth, stages.land_mask, surface_downsample=step)
        lmds = stages.land_mask[::step, ::step].astype(bool)
        z_coarse = np.where(lmds, z_smooth[::step, ::step].astype(np.float64), 0.0)
        z_coarse = snap_land_z_to_triangulated_mesh_floor(z_coarse, lmds)
        self._ui_surface_Z = np.where(lmds, z_coarse, np.nan)
        self._ui_surface_mask = lmds
        self._ui_surface_scale_xy = float(step)
        self.canvas.draw()
        self.btn_gen.setEnabled(True)
        self.act_gen.setEnabled(True)
        self.statusBar().showMessage(
            f"Готово. Модель: {params.height_model}. Можно сохранить PNG (Ctrl+S) или OBJ."
        )
        self._save_params_to_settings()

    def _on_failed(self, msg: str) -> None:
        self.btn_gen.setEnabled(True)
        self.act_gen.setEnabled(True)
        self.statusBar().showMessage("Ошибка.")
        QMessageBox.critical(self, "Ошибка генерации", msg)

    def _start_research(self) -> None:
        if self._research_thread is not None:
            return
        self._set_research_controls(False)
        self.lbl_exp_status.setText("Статус: выполняется серия экспериментов, подождите...")
        self.statusBar().showMessage("Идут эксперименты ЛР2/ЛР3/ЛР4…")
        thr = ResearchThread()
        self._research_thread = thr
        thr.finished_ok.connect(self._on_research_done)
        thr.failed.connect(self._on_research_failed)
        thr.finished.connect(self._on_research_finished)
        thr.start()

    def _on_research_done(self, out_dir: str) -> None:
        self._set_research_controls(True)
        self.lbl_exp_status.setText(
            "Статус: готово. Сформированы CSV/README в output/research.\n"
            f"Папка: {os.path.abspath(out_dir)}"
        )
        self.statusBar().showMessage("Эксперименты завершены.")
        self._refresh_research_results()
        QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(out_dir)))

    def _on_research_failed(self, msg: str) -> None:
        self._set_research_controls(True)
        self.lbl_exp_status.setText("Статус: ошибка запуска экспериментов.")
        self.statusBar().showMessage("Ошибка экспериментов.")
        QMessageBox.critical(self, "Ошибка экспериментов", msg)

    def _open_research_dir(self) -> None:
        out = os.path.join(_output_dir(), "research")
        os.makedirs(out, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(out)))

    def _read_csv_rows(self, name: str) -> list[dict[str, str]]:
        path = os.path.join(_output_dir(), "research", name)
        if not os.path.exists(path):
            return []
        with open(path, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    def _refresh_results_table(self) -> None:
        rows = self._read_csv_rows(self.combo_results_file.currentText())
        self.table_results.clear()
        if not rows:
            self.table_results.setRowCount(0)
            self.table_results.setColumnCount(0)
            return
        cols = list(rows[0].keys())
        show = rows[:120]
        self.table_results.setColumnCount(len(cols))
        self.table_results.setHorizontalHeaderLabels(cols)
        self.table_results.setRowCount(len(show))
        for i, row in enumerate(show):
            for j, col in enumerate(cols):
                self.table_results.setItem(i, j, QTableWidgetItem(str(row.get(col, ""))))
        self.table_results.resizeColumnsToContents()

    def _build_research_report_text(self) -> str:
        lab2 = self._read_csv_rows("lab2_random_summary.csv")
        sym = self._read_csv_rows("lab2_symmetry.csv")
        lab3 = self._read_csv_rows("lab3_landscape_sweep.csv")
        lab4 = self._read_csv_rows("lab4_3d_relief_sweep.csv")
        if not lab2:
            return "Нет результатов. Нажмите «Запустить эксперименты ЛР2–ЛР4»."

        def fmin(rows: list[dict[str, str]], key: str) -> tuple[float, dict[str, str]]:
            best = min(rows, key=lambda r: float(r[key]))
            return float(best[key]), best

        def fmax(rows: list[dict[str, str]], key: str) -> tuple[float, dict[str, str]]:
            best = max(rows, key=lambda r: float(r[key]))
            return float(best[key]), best

        s_min, r_min = fmin(lab2, "mean_stabilization")
        s_max, r_max = fmax(lab2, "mean_stabilization")

        txt = [
            "Сводка исследований (автоматически):",
            "",
            "ЛР2 — начальная плотность и стабилизация:",
            f"- Быстрее всего стабилизируется плотность {r_min['density']} (среднее шагов: {s_min:.2f}).",
            f"- Медленнее всего: плотность {r_max['density']} (среднее шагов: {s_max:.2f}).",
        ]
        if sym:
            ok = sym[0].get("invariant_holds", "0")
            txt.append(
                f"- Симметрия (инверсия, toroidal): {'подтверждена' if ok == '1' else 'не подтверждена'}."
            )
        if lab3:
            _, b3 = fmin(lab3, "blob_count")
            txt.extend(
                [
                    "",
                    "ЛР3 — 2D береговая карта:",
                    f"- Наиболее цельная суша (минимум blobs): sea_prob={b3['sea_probability']}, "
                    f"iterations_base={b3['iterations_base']} (seed={b3['seed']}).",
                ]
            )
        if lab4:
            _, b4 = fmax(lab4, "mean_height")
            _, r4 = fmax(lab4, "roughness")
            txt.extend(
                [
                    "",
                    "ЛР4 — 3D рельеф:",
                    f"- Максимальная средняя высота: model={b4['height_model']}, "
                    f"layers={b4['max_height_layers']}, it={b4['height_layer_iterations']}.",
                    f"- Наибольшая шероховатость: model={r4['height_model']}, "
                    f"layers={r4['max_height_layers']}, it={r4['height_layer_iterations']}.",
                ]
            )
        txt.extend(
            [
                "",
                "Кандидаты (top-5) формируются на основе lab3+lab4; доступны ниже в выпадающем списке.",
                "",
                "Метрики:",
                "- stabilization_step: шаг входа в цикл/стабилизацию",
                "- period: длина цикла",
                "- blob_count: число связных компонент суши",
                "- boundary_length: длина границы суша/море",
                "- roughness: средний модуль перепада высоты между соседями",
            ]
        )
        return "\n".join(txt)

    def _refresh_research_results(self) -> None:
        self.txt_exp_report.setPlainText(self._build_research_report_text())
        self._refresh_results_table()
        self._refresh_candidates()
        rows_now = len(self._read_csv_rows(self.combo_results_file.currentText()))
        self.statusBar().showMessage(
            f"CSV перечитаны: {self.combo_results_file.currentText()} ({rows_now} строк), "
            f"кандидатов: {len(self._candidates)}."
        )

    def _refresh_candidates(self) -> None:
        lab3 = self._read_csv_rows("lab3_landscape_sweep.csv")
        lab4 = self._read_csv_rows("lab4_3d_relief_sweep.csv")
        self._candidates = []
        self.combo_candidates.clear()
        if not lab3 or not lab4:
            self.lbl_candidate_desc.setText("Нет загруженных кандидатов. Сначала запустите эксперименты.")
            self._set_candidate_controls(False)
            return

        l3_pool = [
            r
            for r in lab3
            if 0.35 <= float(r["land_ratio"]) <= 0.80 and float(r["blob_count"]) >= 1
        ]
        if not l3_pool:
            l3_pool = lab3
        l3_sorted = sorted(
            l3_pool,
            key=lambda r: (
                float(r["blob_count"]) * 1.6
                + float(r["boundary_length"]) * 0.01
                + abs(float(r["land_ratio"]) - 0.55) * 100.0
            ),
        )[:8]
        l4_pool = [
            r
            for r in lab4
            if 0.30 <= float(r["land_ratio"]) <= 0.85 and float(r["max_height"]) >= 6
        ]
        if not l4_pool:
            l4_pool = lab4
        tapered_only = [r for r in l4_pool if str(r.get("height_model", "")) == "tapered_height"]
        if tapered_only:
            l4_pool = tapered_only
        l4_sorted = sorted(
            l4_pool,
            key=lambda r: (
                -(
                    float(r["mean_height"]) * 0.95
                    - float(r["roughness"]) * 0.45
                    - abs(float(r["land_ratio"]) - 0.55) * 6.5
                ),
                -float(r["max_height"]),
            ),
        )[:8]

        for i, r4 in enumerate(l4_sorted[:5]):
            r3 = l3_sorted[i % len(l3_sorted)]
            sea_prob = float(r3["sea_probability"])
            p_land = max(0.05, min(0.95, 1.0 - sea_prob))
            cand = {
                "name": f"Кандидат #{i + 1}",
                "seed": int(r4["seed"]),
                "p_land": p_land,
                "coastline_iterations": int(r3["iterations_base"]),
                "height_model": str(r4["height_model"]),
                "max_height_layers": int(r4["max_height_layers"]),
                "height_layer_iterations": int(r4["height_layer_iterations"]),
                "source_lab3": r3,
                "source_lab4": r4,
            }
            self._candidates.append(cand)
            self.combo_candidates.addItem(str(cand["name"]))
        self._set_candidate_controls(True)
        self.lbl_exp_status.setText(
            f"Статус: кандидаты загружены ({len(self._candidates)} шт.). "
            "Выбери один и нажми «Применить и показать результат»."
        )
        self._on_candidate_changed()

    def _set_candidate_controls(self, enabled: bool) -> None:
        self.combo_candidates.setEnabled(enabled)
        self.btn_pick_best.setEnabled(enabled)
        self.btn_apply_show_candidate.setEnabled(enabled)
        self.btn_show_all_candidates.setEnabled(enabled)
        self.btn_mark_manual.setEnabled(enabled)

    def _on_candidate_changed(self) -> None:
        idx = self.combo_candidates.currentIndex()
        if idx < 0 or idx >= len(self._candidates):
            self.lbl_candidate_desc.setText("Нет загруженных кандидатов. Сначала запустите эксперименты.")
            return
        c = self._candidates[idx]
        self.lbl_candidate_desc.setText(
            "Параметры: "
            f"seed={c['seed']}, p_land={float(c['p_land']):.2f}, coast={c['coastline_iterations']}, "
            f"model={c['height_model']}, layers={c['max_height_layers']}, "
            f"layer_it={c['height_layer_iterations']}"
        )

    def _pick_best_candidate(self) -> None:
        if not self._candidates:
            QMessageBox.information(self, "Кандидаты", "Сначала запустите эксперименты.")
            return
        self.combo_candidates.setCurrentIndex(0)
        self.statusBar().showMessage("Выбран лучший кандидат по метрикам (Кандидат #1).")

    def _candidate_params_list(self) -> list[GenerationParams]:
        plist: list[GenerationParams] = []
        for c in self._candidates[:5]:
            plist.append(self._candidate_to_params(c))
        return plist

    def _generate_and_compare_all_candidates(self) -> None:
        if not self._candidates:
            QMessageBox.information(self, "Кандидаты", "Сначала запустите эксперименты.")
            return
        if self._cand_batch_thread is not None:
            return
        self._set_candidate_controls(False)
        self.btn_gen.setEnabled(False)
        self.act_gen.setEnabled(False)
        self.statusBar().showMessage("Генерирую все 5 кандидатов для сравнения…")
        self.lbl_exp_status.setText("Статус: пакетная генерация 5 кандидатов, подождите...")
        self.pb_exp.setVisible(True)
        thr = CandidateBatchThread(self._candidate_params_list())
        self._cand_batch_thread = thr
        thr.finished_ok.connect(self._on_all_candidates_done)
        thr.failed.connect(self._on_all_candidates_failed)
        thr.finished.connect(self._on_all_candidates_finished)
        thr.start()

    def _on_all_candidates_finished(self) -> None:
        self._cand_batch_thread = None

    def _on_all_candidates_failed(self, msg: str) -> None:
        self._set_candidate_controls(True)
        self.btn_gen.setEnabled(True)
        self.act_gen.setEnabled(True)
        self.pb_exp.setVisible(False)
        self.statusBar().showMessage("Ошибка пакетной генерации кандидатов.")
        self.lbl_exp_status.setText("Статус: ошибка пакетной генерации.")
        QMessageBox.critical(self, "Ошибка генерации кандидатов", msg)

    def _on_all_candidates_done(self, items: object) -> None:
        pairs = list(items)  # list[(GenerationParams, stages)]
        import numpy as np

        self.fig.clear()
        n = len(pairs)
        for i, (p, stages) in enumerate(pairs, start=1):
            ax = self.fig.add_subplot(2, 3, i)
            h = np.where(stages.land_mask, stages.heights.astype(float), np.nan)
            ax.imshow(h, cmap="terrain", origin="upper")
            ax.set_title(
                f"#{i}: {p.height_model}, coast={p.coastline_iterations}\n"
                f"layers={p.max_height_layers}, it={p.height_layer_iterations}"
            )
            ax.axis("off")

        # 6-я ячейка — легенда/инструкция.
        ax_info = self.fig.add_subplot(2, 3, 6)
        ax_info.axis("off")
        ax_info.text(
            0.02,
            0.98,
            "Сравнение кандидатов (одним запуском)\n\n"
            "Что смотреть:\n"
            "1) связность берегов/архипелагов\n"
            "2) перепады высот и реалистичность формы\n"
            "3) отсутствие артефактов\n\n"
            "Дальше:\n"
            "• выбери номер в списке кандидатов\n"
            "• нажми «Подтвердить выбор вручную (JSON)»",
            va="top",
            fontsize=10,
        )
        self.fig.tight_layout()
        self.canvas.draw()
        out_dir = os.path.join(_output_dir(), "research")
        os.makedirs(out_dir, exist_ok=True)
        self._compare_image_path = os.path.join(out_dir, "candidates_compare.png")
        self.fig.savefig(self._compare_image_path, dpi=150, bbox_inches="tight")

        self._set_candidate_controls(True)
        self.btn_gen.setEnabled(True)
        self.act_gen.setEnabled(True)
        self.pb_exp.setVisible(False)
        self.statusBar().showMessage("Готово: все 5 кандидатов показаны на экране.")
        self.lbl_exp_status.setText("Статус: все кандидаты сгенерированы и показаны в интерфейсе.")

    def _candidate_to_params(self, c: dict[str, object]) -> GenerationParams:
        p = self._read_params()
        p.seed = int(c["seed"])
        p.p_land = float(c["p_land"])
        p.coastline_iterations = int(c["coastline_iterations"])
        p.height_model = str(c["height_model"])
        p.max_height_layers = int(c["max_height_layers"])
        p.height_layer_iterations = int(c["height_layer_iterations"])
        p.clamp()
        return p

    def _apply_candidate_by_index(self, idx: int, *, run_generate: bool) -> None:
        if idx < 0 or idx >= len(self._candidates):
            QMessageBox.information(self, "Кандидаты", "Сначала запустите эксперименты и обновите результаты.")
            return
        p = self._candidate_to_params(self._candidates[idx])
        self._apply_params(p)
        self.statusBar().showMessage(f"Применён {self._candidates[idx]['name']}.")
        if run_generate:
            self._start_generate()

    def _apply_and_generate_selected_candidate(self) -> None:
        self._apply_candidate_by_index(self.combo_candidates.currentIndex(), run_generate=True)

    def _mark_manual_selection(self) -> None:
        idx = self.combo_candidates.currentIndex()
        if idx < 0 or idx >= len(self._candidates):
            QMessageBox.information(self, "Кандидаты", "Сначала выберите кандидат из списка.")
            return
        try:
            params = self._read_params()
            params.clamp()
        except ValueError as e:
            QMessageBox.warning(self, "Параметры", str(e))
            return

        out_dir = os.path.join(_output_dir(), "research")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "manual_selected_candidate.json")
        payload = {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "selected_candidate_index": idx,
            "selected_candidate_name": self._candidates[idx]["name"],
            "selected_candidate": self._candidates[idx],
            "candidate_params": self._candidate_to_params(self._candidates[idx]).to_dict(),
            "manual_params": params.to_dict(),
            "note": "Алгоритм предложил кандидат; пользователь подтвердил выбор после визуальной проверки.",
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            self.statusBar().showMessage(f"Ручной выбор сохранён: {path}")
            QMessageBox.information(
                self,
                "Выбор зафиксирован",
                "Сохранено в output/research/manual_selected_candidate.json",
            )
        except OSError as e:
            QMessageBox.critical(self, "Ошибка записи", str(e))

    def _save_png(self) -> None:
        if self._last_stages is None:
            QMessageBox.information(self, "Нет данных", "Сначала выполните генерацию.")
            return
        os.makedirs(_output_dir(), exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить изображение",
            os.path.join(_output_dir(), "generation_full.png"),
            "PNG (*.png);;Все файлы (*)",
        )
        if not path:
            return
        try:
            self.fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=self.fig.get_facecolor())
            self.statusBar().showMessage(f"Сохранено: {path}")
        except OSError as e:
            QMessageBox.critical(self, "Ошибка записи", str(e))

    def _export_obj(self) -> None:
        if self._last_stages is None:
            QMessageBox.information(self, "Нет данных", "Сначала выполните генерацию.")
            return
        os.makedirs(_output_dir(), exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить mesh для Blender (Wavefront OBJ)",
            os.path.join(_output_dir(), "terrain.obj"),
            "Wavefront OBJ (*.obj);;Все файлы (*)",
        )
        if not path:
            return
        try:
            from .mesh_export import export_prepared_surface_obj

            if self._ui_surface_Z is None or self._ui_surface_mask is None:
                QMessageBox.warning(self, "Нет данных", "Сначала выполните генерацию.")
                return
            export_prepared_surface_obj(
                path,
                self._ui_surface_Z,
                self._ui_surface_mask,
                scale_xy=self._ui_surface_scale_xy,
            )
            self.statusBar().showMessage(f"OBJ сохранён: {path}")
            folder = os.path.dirname(os.path.abspath(path))
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder))
            QMessageBox.information(
                self,
                "Импорт в Blender",
                "Сохранены .obj и .mtl — в сцене два объекта: terrain (суша) и water_surface (океан).\n"
                "Держите .mtl рядом с .obj. Режим: Material Preview или Rendered.\n"
                "Если не видно воду — выделите water_surface и проверьте, что камера смотрит сверху (Z вверх).\n"
                f"Текущая модель рельефа: {self._last_params.height_model if self._last_params else 'unknown'}",
            )
        except OSError as e:
            QMessageBox.critical(self, "Ошибка записи", str(e))

    def _export_json(self) -> None:
        try:
            p = self._read_params()
            p.clamp()
        except ValueError as e:
            QMessageBox.warning(self, "Параметры", str(e))
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Экспорт параметров",
            os.path.join(_output_dir(), "preset.json"),
            "JSON (*.json)",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(p.to_dict(), f, ensure_ascii=False, indent=2)
            self.statusBar().showMessage(f"Параметры: {path}")
        except OSError as e:
            QMessageBox.critical(self, "Ошибка", str(e))

    def _import_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Импорт параметров",
            _output_dir(),
            "JSON (*.json);;Все файлы (*)",
        )
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            payload = d
            # Поддержка не только "чистого" пресета, но и manual_selected_candidate.json.
            if isinstance(d, dict):
                if isinstance(d.get("candidate_params"), dict):
                    payload = d["candidate_params"]
                elif isinstance(d.get("manual_params"), dict):
                    payload = d["manual_params"]
                elif isinstance(d.get("selected_candidate"), dict):
                    payload = self._candidate_dict_to_params_dict(d["selected_candidate"])
            if not isinstance(payload, dict):
                raise ValueError("JSON должен содержать словарь параметров.")
            p = GenerationParams.from_dict(payload)
            self._apply_params(p)
            self.statusBar().showMessage(f"Загружено: {path}")
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
            QMessageBox.critical(self, "Ошибка файла", str(e))

    def _about(self) -> None:
        QMessageBox.about(
            self,
            "О программе",
            f"<b>{APP_TITLE}</b><br>Версия {APP_VERSION}<br><br>"
            "Процедурная генерация: шум → берег (Day & Night) → рельеф (tapered_height). "
            "Предпросмотр этапов, экспорт OBJ+MTL в Blender.<br><br>"
            "Пошаговая установка и сдача описаны в файле <b>README.md</b> в каталоге проекта.<br><br>"
            "Учебный проект (ВКР).",
        )

    def _save_params_to_settings(self) -> None:
        try:
            p = self._read_params()
            p.clamp()
            self._settings.setValue("params_json", json.dumps(p.to_dict(), ensure_ascii=False))
        except ValueError:
            pass

    def _restore_state(self) -> None:
        geo = self._settings.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._settings.setValue("geometry", self.saveGeometry())
        try:
            p = self._read_params()
            p.clamp()
            self._settings.setValue("params_json", json.dumps(p.to_dict(), ensure_ascii=False))
        except ValueError:
            pass
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_ORG)
    app.setApplicationDisplayName(APP_TITLE)
    # Theme-aware colors for dark/light system themes.
    app.setStyleSheet(
        """
        QMenuBar { color: palette(windowText); background: palette(window); }
        QMenuBar::item { color: palette(windowText); }
        QMenu { color: palette(windowText); background: palette(window); }
        QTabBar::tab { color: palette(windowText); }
        QPushButton { color: palette(button-text); }
        QLabel { color: palette(windowText); }
        """
    )

    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
