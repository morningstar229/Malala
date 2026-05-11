"""Автоматические эксперименты для ЛР2/ЛР3/ЛР4 и выгрузка таблиц в output/research."""

from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass

import numpy as np

from . import day_and_night as dn
from .params import GenerationParams
from .terrain_3d import run_pipeline_from_params


def _research_dir() -> str:
    out = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "output", "research"))
    os.makedirs(out, exist_ok=True)
    return out


def _save_csv(path: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _count_components(mask: np.ndarray) -> int:
    rows, cols = mask.shape
    vis = np.zeros_like(mask, dtype=bool)
    comps = 0
    for i in range(rows):
        for j in range(cols):
            if not mask[i, j] or vis[i, j]:
                continue
            comps += 1
            stack = [(i, j)]
            vis[i, j] = True
            while stack:
                y, x = stack.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if ny < 0 or ny >= rows or nx < 0 or nx >= cols:
                        continue
                    if mask[ny, nx] and not vis[ny, nx]:
                        vis[ny, nx] = True
                        stack.append((ny, nx))
    return comps


def _boundary_length(mask: np.ndarray) -> int:
    m = mask.astype(np.uint8)
    return int(np.sum(m[:, 1:] != m[:, :-1]) + np.sum(m[1:, :] != m[:-1, :]))


def _simulate_until_cycle(
    grid: np.ndarray, *, max_steps: int = 500, toroidal: bool = False
) -> tuple[int, int, np.ndarray]:
    seen: dict[bytes, int] = {}
    g = grid.copy().astype(np.uint8)
    for step in range(max_steps + 1):
        key = g.tobytes()
        prev = seen.get(key)
        if prev is not None:
            return prev, step - prev, g
        seen[key] = step
        g = dn.next_generation(g, toroidal=toroidal).astype(np.uint8)
    return max_steps, -1, g


def _try_plot(path: str, x: list[float], y: list[float], *, title: str, xlabel: str, ylabel: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, marker="o")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run_lab2_random_conditions(rows: int = 64, cols: int = 64) -> list[dict[str, object]]:
    out = _research_dir()
    densities = [0.10, 0.30, 0.50, 0.70]
    seeds = list(range(6))
    table: list[dict[str, object]] = []
    for p in densities:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            g0 = dn.random_binary(rows, cols, p, rng)
            st_step, period, last = _simulate_until_cycle(g0, max_steps=250, toroidal=False)
            table.append(
                {
                    "density": p,
                    "seed": seed,
                    "stabilization_step": st_step,
                    "period": period,
                    "final_land_ratio": float(last.mean()),
                    "blob_count": _count_components(last.astype(bool)),
                    "boundary_length": _boundary_length(last.astype(bool)),
                }
            )
    _save_csv(os.path.join(out, "lab2_random_conditions.csv"), table)

    summary: list[dict[str, object]] = []
    for p in densities:
        vals = [r for r in table if abs(float(r["density"]) - p) < 1e-12]
        summary.append(
            {
                "density": p,
                "mean_stabilization": float(np.mean([int(v["stabilization_step"]) for v in vals])),
                "mean_blobs": float(np.mean([float(v["blob_count"]) for v in vals])),
                "mean_boundary": float(np.mean([float(v["boundary_length"]) for v in vals])),
            }
        )
    _save_csv(os.path.join(out, "lab2_random_summary.csv"), summary)
    _try_plot(
        os.path.join(out, "lab2_density_vs_stabilization.png"),
        [float(s["density"]) for s in summary],
        [float(s["mean_stabilization"]) for s in summary],
        title="ЛР2: плотность vs время стабилизации",
        xlabel="Начальная плотность (доля 1)",
        ylabel="Среднее число шагов до цикла",
    )
    return table


def run_lab2_symmetry(seed: int = 7, rows: int = 64, cols: int = 64, steps: int = 80) -> dict[str, object]:
    out = _research_dir()
    rng = np.random.default_rng(seed)
    base = dn.random_binary(rows, cols, 0.47, rng).astype(np.uint8)
    inv = (1 - base).astype(np.uint8)
    mismatches = 0
    b = base.copy()
    i = inv.copy()
    for _ in range(steps):
        nb = dn.next_generation(b, toroidal=True).astype(np.uint8)
        ni = dn.next_generation(i, toroidal=True).astype(np.uint8)
        mismatches += int(np.sum((1 - nb) != ni))
        b, i = nb, ni
    row = {
        "seed": seed,
        "steps": steps,
        "toroidal": 1,
        "total_cells_checked": steps * rows * cols,
        "mismatch_cells": mismatches,
        "invariant_holds": int(mismatches == 0),
    }
    _save_csv(os.path.join(out, "lab2_symmetry.csv"), [row])
    return row


def run_lab3_2d_landscape_sweep(rows: int = 96, cols: int = 96) -> list[dict[str, object]]:
    out = _research_dir()
    sea_probs = [0.3, 0.4, 0.5, 0.6, 0.7]
    base_iters = [50, 100, 150, 200]
    seeds = [0, 1]
    table: list[dict[str, object]] = []
    for sea in sea_probs:
        for it in base_iters:
            for seed in seeds:
                p = GenerationParams(
                    rows=rows,
                    cols=cols,
                    seed=seed,
                    p_land=1.0 - sea,
                    coastline_iterations=it,
                    max_height_layers=1,
                    height_layer_iterations=1,
                    height_model="tapered_height",
                )
                stages = run_pipeline_from_params(p)
                lm = stages.land_mask
                table.append(
                    {
                        "sea_probability": sea,
                        "iterations_base": it,
                        "seed": seed,
                        "land_ratio": float(lm.mean()),
                        "blob_count": _count_components(lm),
                        "boundary_length": _boundary_length(lm),
                    }
                )
    _save_csv(os.path.join(out, "lab3_landscape_sweep.csv"), table)
    return table


@dataclass
class Lab4Row:
    seed: int
    height_model: str
    max_height_layers: int
    height_layer_iterations: int
    mean_height: float
    std_height: float
    max_height: int
    land_ratio: float
    roughness: float


def run_lab4_3d_relief_sweep(rows: int = 96, cols: int = 96) -> list[dict[str, object]]:
    out = _research_dir()
    table: list[dict[str, object]] = []
    seeds = [0, 1]
    models = ["tapered_height", "column_layers"]
    layers = [20, 28, 36]
    h_iters = [2, 4, 6]
    for model in models:
        for mhl in layers:
            for hi in h_iters:
                for seed in seeds:
                    p = GenerationParams(
                        rows=rows,
                        cols=cols,
                        seed=seed,
                        height_model=model,
                        max_height_layers=mhl,
                        height_layer_iterations=hi,
                    )
                    s = run_pipeline_from_params(p)
                    h = s.heights.astype(float)
                    lm = s.land_mask
                    if lm.any():
                        hh = h[lm]
                        rough = float(np.mean(np.abs(np.diff(h, axis=0))) + np.mean(np.abs(np.diff(h, axis=1))))
                        row = Lab4Row(
                            seed=seed,
                            height_model=model,
                            max_height_layers=mhl,
                            height_layer_iterations=hi,
                            mean_height=float(np.mean(hh)),
                            std_height=float(np.std(hh)),
                            max_height=int(np.max(hh)),
                            land_ratio=float(lm.mean()),
                            roughness=rough,
                        )
                        table.append(asdict(row))
    _save_csv(os.path.join(out, "lab4_3d_relief_sweep.csv"), table)
    return table


def run_all_experiments() -> str:
    out = _research_dir()
    print("ЛР2: случайные начальные условия...", flush=True)
    run_lab2_random_conditions()
    print("ЛР2: симметрия и инверсия...", flush=True)
    run_lab2_symmetry()
    print("ЛР3: sweep по sea_probability/iterations...", flush=True)
    run_lab3_2d_landscape_sweep()
    print("ЛР4: sweep по параметрам высоты...", flush=True)
    run_lab4_3d_relief_sweep()
    report = os.path.join(out, "README_EXPERIMENTS.txt")
    with open(report, "w", encoding="utf-8") as f:
        f.write(
            "Автоэксперименты ЛР2/ЛР3/ЛР4\n"
            "============================\n\n"
            "ЛР2:\n"
            "  - lab2_random_conditions.csv: прогоны по плотностям 10/30/50/70%\n"
            "  - lab2_random_summary.csv: средние по плотности\n"
            "  - lab2_density_vs_stabilization.png: график стабилизации (если доступен matplotlib)\n"
            "  - lab2_symmetry.csv: проверка инвариантности к инверсии (toroidal=1)\n\n"
            "ЛР3:\n"
            "  - lab3_landscape_sweep.csv: sea_probability × iterations_base\n\n"
            "ЛР4:\n"
            "  - lab4_3d_relief_sweep.csv: height_model × max_height_layers × height_layer_iterations\n\n"
            "Запуск:\n"
            "  python -m vkr_terrain.main research\n"
        )
    return out

