"""Экспорт рельефа в Wavefront OBJ + MTL (цвета суши/моря для Blender)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# Доля высоты от размера карты (пик ≈ relief × max(width,height)); совпадает с предпросмотром и Blender.
DEFAULT_RELIEF_FRACTION = 0.15


def _np_gaussian_filter2d(a: np.ndarray, sigma: float) -> np.ndarray:
    """Сепарабельный Гаусс без scipy (fallback)."""
    a = np.asarray(a, dtype=np.float64)
    if sigma <= 1e-9:
        return a.copy()
    r = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float64)
    g = np.exp(-(x**2) / (2.0 * sigma**2))
    g /= g.sum()
    tmp = np.apply_along_axis(lambda row: np.convolve(row, g, mode="same"), 1, a)
    return np.apply_along_axis(lambda col: np.convolve(col, g, mode="same"), 0, tmp)


def build_smooth_height_surface(
    heights: np.ndarray,
    land_mask: np.ndarray,
    *,
    scale_z: float = 0.72,
    median_ksize: int = 3,
    sigma_a: float = 2.2,
    sigma_b: float = 1.4,
    height_power: float = 1.08,
) -> np.ndarray:
    """
    Подготовка высот для mesh / 3D: дискретный CA даёт «иголки» — убираем медианой,
    двойным Гауссом и лёгким сжатием профиля (height_power > 1 прижимает перепады).
    """
    lm = land_mask.astype(bool)
    z = np.where(lm, heights.astype(np.float64) * scale_z, 0.0)

    def _gf(arr: np.ndarray, sig: float) -> np.ndarray:
        if sig <= 0:
            return arr
        try:
            from scipy.ndimage import gaussian_filter

            return gaussian_filter(arr, sigma=sig, mode="nearest")
        except ImportError:
            return _np_gaussian_filter2d(arr, sig)

    def _masked_gf(arr: np.ndarray, mask: np.ndarray, sig: float) -> np.ndarray:
        """Сглаживаем только сушу, без «перетекания» через море."""
        if sig <= 0:
            return arr
        m = mask.astype(np.float64)
        num = _gf(arr * m, sig)
        den = np.maximum(_gf(m, sig), 1e-9)
        return np.where(mask, num / den, 0.0)

    if median_ksize >= 3:
        try:
            from scipy.ndimage import median_filter

            z = np.where(lm, median_filter(z, size=median_ksize, mode="nearest"), 0.0)
        except ImportError:
            pass

    z = _masked_gf(z, lm, sigma_a)
    z = _masked_gf(z, lm, sigma_b)

    mx = float(z[lm].max()) if lm.any() else 0.0
    if mx > 1e-9 and height_power > 1.0:
        zn = np.clip(z / mx, 0.0, 1.0)
        z = np.where(lm, mx * (zn**height_power), 0.0)

    return z


def scale_relief_to_map_extent(
    Z: np.ndarray,
    land_mask: np.ndarray,
    *,
    scale_xy: float,
    relief_fraction: float = DEFAULT_RELIEF_FRACTION,
) -> np.ndarray:
    """
    После сглаживания высоты ~1–3, а карта ~128×128 — в Blender это «блин».
    Растягиваем Z так, чтобы пик был ~relief_fraction от размера карты (те же единицы, что XY).
    """
    lm = land_mask.astype(bool)
    rows, cols = Z.shape
    lw = (cols - 1) * scale_xy
    lh = (rows - 1) * scale_xy
    extent = max(lw, lh, 1e-9)
    zmx = float(Z[lm].max()) if lm.any() else 0.0
    if zmx < 1e-12:
        return Z.astype(np.float64, copy=False)
    target_peak = extent * max(0.04, min(0.35, relief_fraction))
    return Z * (target_peak / zmx)


def _vertices_in_full_land_quads(land_mask: np.ndarray) -> np.ndarray:
    """Угол (i,j) участвует в mesh, если он — вершина хотя бы одной ячейки 2×2 из суши."""
    lm = land_mask.astype(bool)
    rows, cols = lm.shape
    used = np.zeros_like(lm, dtype=bool)
    if rows < 2 or cols < 2:
        return used
    cell_ok = lm[:-1, :-1] & lm[:-1, 1:] & lm[1:, :-1] & lm[1:, 1:]
    used[:-1, :-1] |= cell_ok
    used[:-1, 1:] |= cell_ok
    used[1:, :-1] |= cell_ok
    used[1:, 1:] |= cell_ok
    return used


def _mesh_sea_level_floor(Z: np.ndarray, land_mask: np.ndarray) -> float:
    """
    Уровень «дна» видимого меша (не вся матрица суше): только вершины полных квадов.
    Иначе низ берега часто маской попадает в клетки без 4/4 углов суши — по ним брали min,
    делали большой сдвиг, а реально отрисовывались только более высокие внутренние вершины.
    """
    lm = land_mask.astype(bool)
    z = np.asarray(Z, dtype=np.float64)
    used = _vertices_in_full_land_quads(lm)
    if np.any(used):
        return float(np.min(z[used]))
    if np.any(lm):
        return float(np.min(z[lm]))
    return 0.0


def snap_land_z_to_triangulated_mesh_floor(Z: np.ndarray, land_mask: np.ndarray) -> np.ndarray:
    """
    Сдвигает рельеф на **этой же** сетке, чтобы минимальная высота по углам **реально
    выпускаемых** суходильных четырёхугольников (полные квады 2×2) была 0.

    ``anchor_sea_level`` задаёт общий порог по решётке ``Z[::step]`` на полном массиве; после выборки
    ``[::step]`` нижний угол живого mesh может всё же оказаться заметно выше 0, а ``water_surface``
    остаётся у −eps — отсюда «остров в воздухе» в Blender относительно воды.

    Если полных суходильных ячеек нет, массив не меняют.
    """
    lm = land_mask.astype(bool)
    z = np.asarray(Z, dtype=np.float64).copy()
    rows, cols = z.shape
    vmin = np.inf
    for i in range(rows - 1):
        for j in range(cols - 1):
            if not (lm[i, j] and lm[i, j + 1] and lm[i + 1, j + 1] and lm[i + 1, j]):
                continue
            vmin = min(
                vmin,
                float(z[i, j]),
                float(z[i, j + 1]),
                float(z[i + 1, j + 1]),
                float(z[i + 1, j]),
            )
    if not np.isfinite(vmin):
        return z
    return np.where(lm, np.maximum(z - vmin, 0.0), 0.0)


def anchor_sea_level(
    Z: np.ndarray,
    land_mask: np.ndarray,
    *,
    surface_downsample: int = 1,
) -> np.ndarray:
    """
    Уровень моря у z≈0. Сдвиг считается по той же сетке, что и 3D/OBJ после surface_downsample,
    и только по вершинам триангуляции — как в экспорте.
    """
    lm = land_mask.astype(bool)
    z = np.asarray(Z, dtype=np.float64)
    if not np.any(lm):
        return z.copy()
    step = max(1, int(surface_downsample))
    z_floor = _mesh_sea_level_floor(z[::step, ::step], lm[::step, ::step])
    return np.where(lm, np.maximum(z - z_floor, 0.0), 0.0)


def _water_plane_z_for_mesh_land(
    zf: np.ndarray,
    land_mask: np.ndarray,
    *,
    scale_xy: float,
) -> float:
    """
    Горизонт воды в тех же координатах, что и суша после ``anchor_sea_level``:
    «море» на условном z≈0, чуть ниже нуля против z-fighting.
    Высота по маске не подрезается под «макс. берег» — это ломало соответствие предпросмотру (осям Z от 0).
    """
    lm = land_mask.astype(bool)
    z_peak = float(np.nanmax(zf[lm])) if lm.any() else 1.0
    xy = float(max(scale_xy, 1e-9))
    return -float(max(z_peak * 0.002, xy * 0.003, 1e-9))


def _vertex_normals_from_height(Z: np.ndarray, scale_xy: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Нормали к z = f(x,y), x = j·s, y = i·s (как в Matplotlib plot_surface: строка i → +Y)."""
    Zi = np.gradient(Z, axis=0)
    Zj = np.gradient(Z, axis=1)
    s = max(scale_xy, 1e-9)
    dZdx = Zj / s
    dZdy = Zi / s
    nx = -dZdx
    ny = -dZdy
    nz = np.ones_like(Z)
    ln = np.sqrt(nx * nx + ny * ny + nz * nz)
    ln = np.maximum(ln, 1e-12)
    return nx / ln, ny / ln, nz / ln


def _write_mtl(mtl_path: str) -> None:
    """Диффузные цвета: суша — зелёно-коричневый, море — синее (видно в Material Preview)."""
    text = """# VKR terrain — в Blender включите Viewport Shading: Material Preview или Rendered
newmtl Land
Ka 0.15 0.18 0.12
Kd 0.28 0.52 0.24
Ks 0.12 0.15 0.10
Ns 32
d 1.0

newmtl Sea
Ka 0.08 0.22 0.45
Kd 0.10 0.48 0.88
Ks 0.45 0.55 0.65
Ns 120
d 1.0
"""
    with open(mtl_path, "w", encoding="utf-8") as f:
        f.write(text)


def export_heightmap_obj(
    path: str,
    heights: np.ndarray,
    land_mask: np.ndarray,
    *,
    scale_xy: float = 1.0,
    scale_z: float = 0.72,
    median_ksize: int = 3,
    sigma_a: float = 2.2,
    sigma_b: float = 1.4,
    height_power: float = 1.08,
    relief_fraction: float = DEFAULT_RELIEF_FRACTION,
    surface_downsample: int = 1,
) -> None:
    """
    Регулярная сетка: вся карта (и море z≈0, и суша) — в Blender будет и вода, и земля.
    MTL рядом с OBJ: зелёная суша, синее море (материалы назначаются по треугольникам).
    Высота по Z подогнана к размеру карты (relief_fraction), как в предпросмотре приложения.
    """
    path = os.path.abspath(path)
    step = max(1, int(surface_downsample))
    h_src = heights[::step, ::step]
    lm = land_mask.astype(bool)[::step, ::step]
    rows, cols = h_src.shape

    Z = build_smooth_height_surface(
        h_src,
        lm,
        scale_z=scale_z,
        median_ksize=median_ksize,
        sigma_a=sigma_a,
        sigma_b=sigma_b,
        height_power=height_power,
    )
    sx = scale_xy * step
    Z = scale_relief_to_map_extent(Z, lm, scale_xy=sx, relief_fraction=relief_fraction)
    Z = anchor_sea_level(Z, lm, surface_downsample=1)
    Z = snap_land_z_to_triangulated_mesh_floor(Z, lm)
    nx, ny, nz = _vertex_normals_from_height(Z, sx)

    base = Path(path)
    mtl_name = base.with_suffix(".mtl").name
    mtl_full = str(base.with_suffix(".mtl"))
    _write_mtl(mtl_full)

    lines: list[str] = [
        "# Blender: импорт Wavefront OBJ, ориентация по умолчанию; ось высот — Z.",
        "# XY совпадает с порядком сетки в приложении (j→X, i→Y, без переворота картинки).",
        f"mtllib {mtl_name}",
        "o terrain",
    ]

    def vid(i: int, j: int) -> int:
        return i * cols + j + 1

    for i in range(rows):
        for j in range(cols):
            z = float(Z[i, j])
            lines.append(f"v {j * sx:.6f} {i * sx:.6f} {z:.6f}")

    for i in range(rows):
        for j in range(cols):
            lines.append(f"vn {nx[i,j]:.6f} {ny[i,j]:.6f} {nz[i,j]:.6f}")

    # Треугольники суши строим только для ПОЛНОЙ сухой ячейки (все 4 угла суша).
    # Это максимально близко к plot_surface в UI (где ячейки с NaN на углах не рисуются).
    tris: list[tuple[str, int, int, int]] = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            a = vid(i, j)
            b = vid(i, j + 1)
            c = vid(i + 1, j + 1)
            d = vid(i + 1, j)

            full_land_cell = bool(
                lm[i, j] and lm[i, j + 1] and lm[i + 1, j + 1] and lm[i + 1, j]
            )
            if not full_land_cell:
                continue
            tris.append(("Land", a, b, c))
            tris.append(("Land", a, c, d))

    cur_mtl = ""
    for tag, v1, v2, v3 in tris:
        if tag != cur_mtl:
            lines.append(f"usemtl {tag}")
            cur_mtl = tag
        lines.append(f"f {v1}//{v1} {v2}//{v2} {v3}//{v3}")

    # Цельная вода: чуть ниже z=0, чтобы не мерцало с берегом; запас по краям карты.
    n_terrain_verts = rows * cols
    n_terrain_vn = rows * cols
    xm = (cols - 1) * sx
    ym = (rows - 1) * sx
    margin = sx * 2.5
    x0, x1 = -margin, xm + margin
    y0, y1 = -margin, ym + margin
    zw = _water_plane_z_for_mesh_land(Z, lm, scale_xy=sx)

    lines.append("o water_surface")
    lines.append("usemtl Sea")
    lines.append(f"v {x0:.6f} {y0:.6f} {zw:.6f}")
    lines.append(f"v {x1:.6f} {y0:.6f} {zw:.6f}")
    lines.append(f"v {x1:.6f} {y1:.6f} {zw:.6f}")
    lines.append(f"v {x0:.6f} {y1:.6f} {zw:.6f}")
    wn_idx = n_terrain_vn + 1
    lines.append("vn 0.0 0.0 1.0")
    w0 = n_terrain_verts + 1
    w1 = n_terrain_verts + 2
    w2 = n_terrain_verts + 3
    w3 = n_terrain_verts + 4
    lines.append(f"f {w0}//{wn_idx} {w1}//{wn_idx} {w2}//{wn_idx}")
    lines.append(f"f {w0}//{wn_idx} {w2}//{wn_idx} {w3}//{wn_idx}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def export_prepared_surface_obj(
    path: str,
    Z: np.ndarray,
    land_mask: np.ndarray,
    *,
    scale_xy: float = 1.0,
) -> None:
    """
    Экспорт уже подготовленной поверхности (1-в-1 с UI).
    Z и land_mask должны быть той же сеткой, что использована в plot_surface.
    """
    path = os.path.abspath(path)
    rows, cols = Z.shape
    lm = land_mask.astype(bool)
    zf = np.where(lm, np.nan_to_num(Z.astype(float), nan=0.0), 0.0)
    zf = snap_land_z_to_triangulated_mesh_floor(zf, lm)
    nx, ny, nz = _vertex_normals_from_height(zf, scale_xy)

    base = Path(path)
    mtl_name = base.with_suffix(".mtl").name
    mtl_full = str(base.with_suffix(".mtl"))
    _write_mtl(mtl_full)

    lines: list[str] = [
        f"mtllib {mtl_name}",
        "# XY как в UI: колонка j → X, строка i → Y (предпросмотр без зеркала по вертикали).",
        "o terrain",
    ]

    def vid(i: int, j: int) -> int:
        return i * cols + j + 1

    for i in range(rows):
        for j in range(cols):
            lines.append(f"v {j * scale_xy:.6f} {i * scale_xy:.6f} {float(zf[i, j]):.6f}")
    for i in range(rows):
        for j in range(cols):
            lines.append(f"vn {nx[i,j]:.6f} {ny[i,j]:.6f} {nz[i,j]:.6f}")

    lines.append("usemtl Land")
    for i in range(rows - 1):
        for j in range(cols - 1):
            if not (lm[i, j] and lm[i, j + 1] and lm[i + 1, j + 1] and lm[i + 1, j]):
                continue
            a = vid(i, j)
            b = vid(i, j + 1)
            c = vid(i + 1, j + 1)
            d = vid(i + 1, j)
            lines.append(f"f {a}//{a} {b}//{b} {c}//{c}")
            lines.append(f"f {a}//{a} {c}//{c} {d}//{d}")

    # Единая водная плоскость (как в основном экспорте)
    n_terrain_verts = rows * cols
    n_terrain_vn = rows * cols
    xm = (cols - 1) * scale_xy
    ym = (rows - 1) * scale_xy
    margin = scale_xy * 2.5
    x0, x1 = -margin, xm + margin
    y0, y1 = -margin, ym + margin
    zw = _water_plane_z_for_mesh_land(zf, lm, scale_xy=scale_xy)

    lines.append("o water_surface")
    lines.append("usemtl Sea")
    lines.append(f"v {x0:.6f} {y0:.6f} {zw:.6f}")
    lines.append(f"v {x1:.6f} {y0:.6f} {zw:.6f}")
    lines.append(f"v {x1:.6f} {y1:.6f} {zw:.6f}")
    lines.append(f"v {x0:.6f} {y1:.6f} {zw:.6f}")
    wn_idx = n_terrain_vn + 1
    lines.append("vn 0.0 0.0 1.0")
    w0 = n_terrain_verts + 1
    w1 = n_terrain_verts + 2
    w2 = n_terrain_verts + 3
    w3 = n_terrain_verts + 4
    lines.append(f"f {w0}//{wn_idx} {w1}//{wn_idx} {w2}//{wn_idx}")
    lines.append(f"f {w0}//{wn_idx} {w2}//{wn_idx} {w3}//{wn_idx}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
