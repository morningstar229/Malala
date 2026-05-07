"""
Двумерный клеточный автомат: правило «День и ночь» (B3678/S34678) и настраиваемые B/S.

Окрестность Мура (8 соседей). Границы: «мертвые» нули или тор (склейка краёв).
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def count_neighbors_moore(
    grid: np.ndarray,
    i: int,
    j: int,
    toroidal: bool = False,
) -> int:
    rows, cols = grid.shape
    s = 0
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            if toroidal:
                ni = (i + di) % rows
                nj = (j + dj) % cols
            else:
                ni, nj = i + di, j + dj
                if ni < 0 or ni >= rows or nj < 0 or nj >= cols:
                    continue
            s += int(grid[ni, nj] != 0)
    return s


def next_generation(
    grid: np.ndarray,
    birth: Iterable[int] | None = None,
    survival: Iterable[int] | None = None,
    toroidal: bool = False,
) -> np.ndarray:
    """
    Один шаг. По умолчанию Day & Night: B3678 / S34678.
    alive: ненулевые клетки.
    """
    b = set(birth if birth is not None else (3, 6, 7, 8))
    s = set(survival if survival is not None else (3, 4, 6, 7, 8))
    rows, cols = grid.shape
    alive = grid != 0
    new_alive = np.zeros_like(alive, dtype=bool)
    for i in range(rows):
        for j in range(cols):
            n = count_neighbors_moore(alive.astype(np.uint8), i, j, toroidal=toroidal)
            if not alive[i, j]:
                new_alive[i, j] = n in b
            else:
                new_alive[i, j] = n in s
    return new_alive.astype(grid.dtype)


def run_iterations(
    grid: np.ndarray,
    steps: int,
    birth: Sequence[int] | None = None,
    survival: Sequence[int] | None = None,
    toroidal: bool = False,
) -> np.ndarray:
    g = grid.copy()
    for _ in range(steps):
        g = next_generation(g, birth=birth, survival=survival, toroidal=toroidal)
    return g


def random_binary(rows: int, cols: int, p_land: float, rng: np.random.Generator) -> np.ndarray:
    return (rng.random((rows, cols)) < p_land).astype(np.uint8)
