"""
Альтернативная модель высоты: 3D-объём, наращивание суши с убывающей вероятностью по высоте,
затем правило Day & Night на горизонтальных срезах (воздух / суша).
"""

from __future__ import annotations

import numpy as np

from . import day_and_night as dn
from .terrain_3d import GenerationStages, build_earth_and_water

AIR, LAND, SEA = 0, 1, 2
WARNING = {3, 6, 7, 8}


def _neighbor_land_air_counts(
    plane: np.ndarray,
    r: int,
    c: int,
    rows: int,
    cols: int,
    toroidal: bool,
) -> tuple[int, int]:
    land_n = 0
    air_n = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if toroidal:
                nr %= rows
                nc %= cols
            elif nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            v = plane[nr, nc]
            if v == LAND:
                land_n += 1
            else:
                air_n += 1
    return land_n, air_n


def _grow_vertical_land(
    vol: np.ndarray,
    rng: np.random.Generator,
    H: int,
    rows: int,
    cols: int,
) -> None:
    """Наращивание столбов суши с убывающей вероятностью по мере подъёма."""
    for r in range(rows):
        for c in range(cols):
            if vol[0, r, c] != LAND:
                continue
            step = 100.0 / max(H, 1)
            ver = step
            for yi in range(1, H):
                r99 = int(rng.integers(1, 101))
                if r99 > ver:
                    vol[yi, r, c] = LAND
                    ver += step
                else:
                    break


def next_generation_high_slice(
    vol: np.ndarray,
    yi: int,
    rows: int,
    cols: int,
    toroidal: bool,
) -> np.ndarray:
    """Day & Night на одном горизонтальном уровне yi (воздух / суша)."""
    H = vol.shape[0]
    plane = vol[yi]
    new_plane = plane.copy()
    for r in range(rows):
        for c in range(cols):
            cur = plane[r, c]
            if cur != LAND and cur != AIR:
                continue
            land_n, air_n = _neighbor_land_air_counts(plane, r, c, rows, cols, toroidal)
            if cur == LAND:
                if air_n in WARNING:
                    if yi + 1 < H and vol[yi + 1, r, c] == AIR:
                        new_plane[r, c] = AIR
            else:
                if land_n in WARNING:
                    if yi > 0 and vol[yi - 1, r, c] == LAND:
                        new_plane[r, c] = LAND
    out = vol.copy()
    out[yi] = new_plane
    return out


def run_tapered_height_stages(
    rows: int,
    cols: int,
    H: int,
    rng: np.random.Generator,
    *,
    pct_sea: float,
    coast_iterations: int,
    day_night_height_iterations: int,
    sea_water_depth: int,
    toroidal: bool = False,
) -> GenerationStages:
    """
    pct_sea: 0–100, ожидаемая доля моря на нижнем слое (доля «морских» клеток).
    """
    pct_sea = max(0.0, min(100.0, float(pct_sea)))
    H = max(4, min(200, int(H)))

    initial = np.zeros((rows, cols), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            if rng.integers(1, 101) > pct_sea:
                initial[r, c] = 1

    base = initial.astype(np.uint8)
    land2d = dn.run_iterations(base.copy(), coast_iterations, toroidal=toroidal)
    land_bottom = land2d.astype(bool)

    vol = np.zeros((H, rows, cols), dtype=np.uint8)
    vol[0] = np.where(land_bottom, LAND, SEA)

    _grow_vertical_land(vol, rng, H, rows, cols)

    for _ in range(day_night_height_iterations):
        for yi in range(1, H):
            vol = next_generation_high_slice(vol, yi, rows, cols, toroidal)

    heights = np.zeros((rows, cols), dtype=np.int16)
    for r in range(rows):
        for c in range(cols):
            hz = 0
            for yi in range(H):
                if vol[yi, r, c] == LAND:
                    hz = max(hz, yi + 1)
            heights[r, c] = hz

    land_mask = heights > 0
    earth, water = build_earth_and_water(land_mask, heights, water_depth=sea_water_depth)

    coast = land_bottom.astype(np.float32)

    return GenerationStages(
        initial_noise=initial,
        land_mask=land_mask,
        heights=heights,
        earth_voxels=earth,
        water_voxels=water,
        coast_plane=coast,
    )
