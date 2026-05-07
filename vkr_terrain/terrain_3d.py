"""
Генерация 3D-ландшафта по схеме из темы ВКР:
  этап 1 — случайная суша/море;
  этап 2 — правило Day & Night для береговой линии;
  этап 3 — наростание высоты слоями (тот же CA на масках над сушей).

Итог: карта высот, воксельный объём суши и отдельно вода под морем (для 3D-сцены).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import day_and_night as dn
from .params import GenerationParams
from .settings import (
    COASTLINE_ITERATIONS,
    HEIGHT_LAYER_ITERATIONS,
    INITIAL_LAND_PROB,
    LAYER_NOISE_PROBABILITY,
    MAX_HEIGHT_LAYERS,
    SEA_WATER_DEPTH,
)


def generate_land_mask(
    rows: int,
    cols: int,
    rng: np.random.Generator,
    coastline_iterations: int = COASTLINE_ITERATIONS,
    p_land: float = INITIAL_LAND_PROB,
    toroidal: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает (начальный шум 0/1, маска суши после CA)."""
    initial = dn.random_binary(rows, cols, p_land, rng)
    evolved = dn.run_iterations(initial.copy(), coastline_iterations, toroidal=toroidal)
    return initial.astype(np.uint8), evolved.astype(bool)


def generate_height_map(
    land_mask: np.ndarray,
    rng: np.random.Generator,
    max_layers: int = MAX_HEIGHT_LAYERS,
    layer_iterations: int = HEIGHT_LAYER_ITERATIONS,
    toroidal: bool = False,
    layer_noise_probability: float = LAYER_NOISE_PROBABILITY,
) -> np.ndarray:
    """Высота в клетках над уровнем моря (целые >= 0)."""
    rows, cols = land_mask.shape
    p = float(layer_noise_probability)
    height = np.zeros((rows, cols), dtype=np.int16)
    for _ in range(max_layers):
        noise = (rng.random((rows, cols)) < p) & land_mask
        noise = noise.astype(np.uint8)
        evolved = dn.run_iterations(noise, layer_iterations, toroidal=toroidal)
        height += evolved.astype(np.int16)
    # На суше минимум один слой «земли», иначе остров без вокселей
    height = np.where(land_mask, np.maximum(height, np.int16(1)), height)
    return height


def build_voxel_volume(
    land_mask: np.ndarray,
    height_map: np.ndarray,
    max_z: int | None = None,
) -> np.ndarray:
    """Только суша: world[z, y, x] — столбик от дна вверх."""
    rows, cols = land_mask.shape
    h_max = int(height_map.max()) if height_map.size else 0
    if max_z is not None:
        h_max = min(h_max, max_z)
    depth = max(h_max, 1)
    world = np.zeros((depth, rows, cols), dtype=bool)
    for y in range(rows):
        for x in range(cols):
            if not land_mask[y, x]:
                continue
            hz = int(height_map[y, x])
            world[:hz, y, x] = True
    return world


def build_earth_and_water(
    land_mask: np.ndarray,
    height_map: np.ndarray,
    water_depth: int = SEA_WATER_DEPTH,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Общий объём по Z: max(макс. высота суши, water_depth).
    earth и water не пересекаются (море — вода внизу, суша — столбик земли).
    """
    rows, cols = land_mask.shape
    h_max = int(height_map.max()) if height_map.size else 1
    depth = max(h_max, max(water_depth, 1))
    earth = np.zeros((depth, rows, cols), dtype=bool)
    water = np.zeros((depth, rows, cols), dtype=bool)
    wd = max(0, min(water_depth, depth))
    for y in range(rows):
        for x in range(cols):
            if land_mask[y, x]:
                hz = int(height_map[y, x])
                earth[:hz, y, x] = True
            else:
                water[:wd, y, x] = True
    return earth, water


@dataclass
class GenerationStages:
    """Все промежуточные данные для отчёта и отладки."""

    initial_noise: np.ndarray  # 0/1 после этапа 1
    land_mask: np.ndarray  # итоговая суша (есть рельеф)
    heights: np.ndarray  # карта высот (этап 3)
    earth_voxels: np.ndarray  # bool [Z,Y,X]
    water_voxels: np.ndarray  # bool [Z,Y,X], море
    coast_plane: np.ndarray | None = None  # 2D маска суша/море сразу после CA (этап 2 в отчёте)


def run_pipeline(
    rows: int,
    cols: int,
    seed: int | None = None,
    toroidal: bool = False,
    *,
    p_land: float | None = None,
    coastline_iterations: int | None = None,
    max_height_layers: int | None = None,
    height_layer_iterations: int | None = None,
    layer_noise_probability: float | None = None,
    sea_water_depth: int | None = None,
) -> GenerationStages:
    rng = np.random.default_rng(seed)
    pl = INITIAL_LAND_PROB if p_land is None else p_land
    ci = COASTLINE_ITERATIONS if coastline_iterations is None else coastline_iterations
    mhl = MAX_HEIGHT_LAYERS if max_height_layers is None else max_height_layers
    hli = HEIGHT_LAYER_ITERATIONS if height_layer_iterations is None else height_layer_iterations
    lnp = LAYER_NOISE_PROBABILITY if layer_noise_probability is None else layer_noise_probability
    swd = SEA_WATER_DEPTH if sea_water_depth is None else sea_water_depth

    initial_noise, land = generate_land_mask(
        rows, cols, rng, coastline_iterations=ci, p_land=pl, toroidal=toroidal
    )
    heights = generate_height_map(
        land,
        rng,
        max_layers=mhl,
        layer_iterations=hli,
        toroidal=toroidal,
        layer_noise_probability=lnp,
    )
    earth, water = build_earth_and_water(land, heights, water_depth=swd)
    return GenerationStages(
        initial_noise=initial_noise,
        land_mask=land,
        heights=heights,
        earth_voxels=earth,
        water_voxels=water,
        coast_plane=land.astype(np.float32),
    )


def run_pipeline_from_params(p: GenerationParams) -> GenerationStages:
    p.clamp()
    from .tapered_height_pipeline import run_tapered_height_stages

    rng = np.random.default_rng(p.seed)
    pct_sea = (1.0 - p.p_land) * 100.0
    return run_tapered_height_stages(
        p.rows,
        p.cols,
        p.max_height_layers,
        rng,
        pct_sea=pct_sea,
        coast_iterations=p.coastline_iterations,
        day_night_height_iterations=p.height_layer_iterations,
        sea_water_depth=p.sea_water_depth,
        toroidal=p.toroidal,
    )


def pipeline_full(
    rows: int,
    cols: int,
    seed: int | None = None,
    toroidal: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Обратная совместимость: (land, heights, earth_only_volume)."""
    g = run_pipeline(rows, cols, seed, toroidal)
    vol = build_voxel_volume(g.land_mask, g.heights)
    return g.land_mask, g.heights, vol
