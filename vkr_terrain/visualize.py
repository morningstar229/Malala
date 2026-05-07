"""2D (Pygame) и полная доска генерации: этапы + 3D (Matplotlib)."""

from __future__ import annotations

import os

import numpy as np

from .mesh_export import (
    DEFAULT_RELIEF_FRACTION,
    anchor_sea_level,
    build_smooth_height_surface,
    scale_relief_to_map_extent,
    snap_land_z_to_triangulated_mesh_floor,
)

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
except ImportError:
    plt = None
    Figure = None  # type: ignore[misc, assignment]
    LightSource = None  # type: ignore[misc, assignment]


def height_to_rgb(height: np.ndarray, land_mask: np.ndarray) -> np.ndarray:
    """Карта высот: море — синий, суша — от зелёного к коричневому."""
    rows, cols = height.shape
    rgb = np.zeros((rows, cols, 3), dtype=np.float32)
    h = height.astype(np.float32)
    if h.max() > 0:
        hn = h / (h.max() + 1e-6)
    else:
        hn = np.zeros_like(h, dtype=np.float32)
    rgb[..., 0] = np.where(land_mask, 0.2 + 0.5 * hn, 0.05)
    rgb[..., 1] = np.where(land_mask, 0.45 + 0.35 * hn, 0.25)
    rgb[..., 2] = np.where(land_mask, 0.15 + 0.1 * hn, 0.55 + 0.2 * (1 - land_mask.astype(float)))
    return rgb


def _subsample_volumes(
    earth: np.ndarray,
    water: np.ndarray,
    max_axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    z, y, x = earth.shape
    sy = max(1, (y + max_axis - 1) // max_axis)
    sx = max(1, (x + max_axis - 1) // max_axis)
    sz = max(1, (z + max_axis - 1) // max_axis)
    return earth[::sz, ::sy, ::sx], water[::sz, ::sy, ::sx]


def hillshade_rgba(heights: np.ndarray, land_mask: np.ndarray) -> np.ndarray:
    """Псевдо-рельеф с освещением; море — только по маске суше/морю (не подрезать внутреннюю сушу по Z)."""
    if LightSource is None or plt is None:
        return np.zeros((*heights.shape, 4))
    lm = land_mask.astype(bool)
    z = np.where(lm, heights.astype(float), 0.0)
    ls = LightSource(azdeg=315, altdeg=58)
    cmap = plt.cm.gist_earth
    rgba = ls.shade(z, cmap=cmap, vert_exag=0.045, blend_mode="soft")
    sea_pixels = ~lm
    rgba[sea_pixels, 0] = rgba[sea_pixels, 0] * 0.15 + 0.03
    rgba[sea_pixels, 1] = rgba[sea_pixels, 1] * 0.15 + 0.14
    rgba[sea_pixels, 2] = rgba[sea_pixels, 2] * 0.25 + 0.62
    rgba[sea_pixels, 3] = 1.0
    return rgba


def _voxel_facecolors(earth: np.ndarray, water: np.ndarray) -> np.ndarray:
    """RGBA для ax.voxels: земля по высоте, вода полупрозрачная."""
    assert earth.shape == water.shape
    import matplotlib.cm as cm

    cmap = cm.get_cmap("terrain")
    nz, ny, nx = earth.shape
    zz = np.indices((nz, ny, nx), dtype=float)[0]
    rgba = cmap(zz / max(nz - 1, 1))
    colors = np.zeros((nz, ny, nx, 4), dtype=float)
    colors[earth] = rgba[earth]
    water_rgba = (0.15, 0.35, 0.85, 0.55)
    colors[water & ~earth] = water_rgba
    return colors


def render_generation_board_on_figure(
    fig: "Figure",
    stages: "GenerationStages",
    *,
    voxel_max_axis: int = 48,
    surface_downsample: int = 2,
    suptitle: str | None = None,
) -> None:
    """
    Рисует полную доску на существующей Figure (для GUI и экспорта).
    """
    ini = stages.initial_noise.astype(float)
    coast = stages.coast_plane
    if coast is not None:
        plane2 = coast.astype(float)
    else:
        plane2 = stages.land_mask.astype(float)
    hmap = stages.heights.astype(float)
    z_smooth = build_smooth_height_surface(stages.heights, stages.land_mask)
    # Те же единицы, что и в OBJ: высота сопоставима с шириной карты (не «блин» в Blender).
    z_smooth = scale_relief_to_map_extent(
        z_smooth, stages.land_mask, scale_xy=1.0, relief_fraction=DEFAULT_RELIEF_FRACTION
    )
    step = max(1, surface_downsample)
    # Привязка уровня моря по вершинам mesh на той сетке, что 3D и экспорт после step.
    z_smooth = anchor_sea_level(z_smooth, stages.land_mask, surface_downsample=step)

    earth = stages.earth_voxels
    water = stages.water_voxels
    hs_rgba = hillshade_rgba(z_smooth, stages.land_mask)

    fig.clear()
    fig.suptitle(
        suptitle
        or "Пайплайн: шум → берег (Day & Night) → рельеф",
        fontsize=10,
    )

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(ini, cmap="Greys", origin="upper")
    ax1.set_title("Этап 1\nСлучайная суша/море", fontsize=10)
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(plane2, cmap="Blues_r", origin="upper", vmin=0, vmax=1)
    ax2.set_title("Этап 2\nБерег (CA Day&Night)", fontsize=10)
    ax2.axis("off")

    ax3 = fig.add_subplot(2, 3, 3)
    im = ax3.imshow(hmap.astype(float), cmap="terrain", origin="upper")
    ax3.set_title("Этап 3\nКарта высот", fontsize=10)
    ax3.axis("off")
    fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    lmds = stages.land_mask[::step, ::step].astype(bool)
    z_coarse = np.where(lmds, z_smooth[::step, ::step].astype(np.float64), 0.0)
    z_coarse = snap_land_z_to_triangulated_mesh_floor(z_coarse, lmds)
    h_ds = np.where(lmds, z_coarse, np.nan)
    ys = np.arange(h_ds.shape[0])
    xs = np.arange(h_ds.shape[1])
    X, Y = np.meshgrid(xs, ys)
    ax4 = fig.add_subplot(2, 3, 4, projection="3d")
    ax4.plot_surface(
        X,
        Y,
        h_ds,
        cmap="terrain",
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        alpha=0.98,
        shade=True,
    )
    xs0 = float(np.nanmax(X) - np.nanmin(X)) or 1.0
    ys0 = float(np.nanmax(Y) - np.nanmin(Y)) or 1.0
    z_land_max = float(np.nanmax(h_ds))
    z_land_min = float(np.nanmin(h_ds))
    # Иначе mpl поджимает Z к данным острова и он «парит» без отметки условного z=0 (уровня якоря).
    z_top = max(z_land_max * 1.02, z_land_min + 1e-6)
    ax4.set_zlim(0.0, z_top)
    z_span = max(z_top, 1e-9)
    try:
        ax4.set_box_aspect((xs0, ys0, z_span))
    except Exception:
        pass
    ax4.set_title("3D поверхность\n(ось Z от 0 = якорь; как в Blender)", fontsize=10)
    ax4.set_xlabel("X")
    ax4.set_ylabel("Y")
    ax4.set_zlabel("Высота")

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(hs_rgba, origin="upper", interpolation="bilinear")
    ax5.set_title("Рельеф с освещением\n(псевдо-3D)", fontsize=10)
    ax5.axis("off")

    n_land = int(np.sum(stages.land_mask))
    n_sea = int(stages.land_mask.size - n_land)
    vol_earth = int(np.sum(earth))
    vol_water = int(np.sum(water))
    zmax = earth.shape[0]
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")
    ax6.text(
        0.02,
        0.95,
        "Статистика генерации\n"
        f"Клеток суши: {n_land}\n"
        f"Клеток моря: {n_sea}\n"
        f"Макс. высота: {int(stages.heights.max())} (вокселей)\n"
        f"Размер объёма Z×Y×X: {zmax}×{earth.shape[1]}×{earth.shape[2]}\n"
        f"Вокселей земли: {vol_earth}\n"
        f"Вокселей воды: {vol_water}\n"
        "\nЭкспорт OBJ: низ меша совмещён с горизонтом воды (доп. якорь под прореживание сетки); вода чуть ниже z=0.\n"
        "2D hillshade и этапы — маска суше/море по сетке (без затопления внутренних низин по высоте).\n"
        "Режим рельефа: tapered_height.",
        fontsize=9,
        verticalalignment="top",
        fontfamily="sans-serif",
        transform=ax6.transAxes,
    )

    fig.subplots_adjust(top=0.9, hspace=0.42, wspace=0.32)


def save_generation_board(
    stages: "GenerationStages",
    path_png: str,
    *,
    voxel_max_axis: int = 48,
    surface_downsample: int = 2,
    show: bool = False,
) -> None:
    """Сохраняет PNG с полной доской (этапы + 3D)."""
    if plt is None:
        return

    fig = plt.figure(figsize=(16, 10))
    render_generation_board_on_figure(
        fig,
        stages,
        voxel_max_axis=voxel_max_axis,
        surface_downsample=surface_downsample,
    )
    os.makedirs(os.path.dirname(path_png) or ".", exist_ok=True)
    fig.savefig(path_png, dpi=140, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def save_3d_plots(
    height: np.ndarray,
    land_mask: np.ndarray,
    path_surface: str,
    path_voxels: str | None = None,
    downsample: int = 2,
) -> None:
    """Устаревшее: два отдельных файла (оставлено для совместимости)."""
    if plt is None:
        return
    h = height.astype(float)
    h = np.where(land_mask, h, np.nan)
    step = max(1, downsample)
    h_ds = h[::step, ::step]
    ys = np.arange(h_ds.shape[0])
    xs = np.arange(h_ds.shape[1])
    X, Y = np.meshgrid(xs, ys)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, h_ds, cmap="terrain", linewidth=0, antialiased=True, alpha=0.95)
    ax.set_title("Поверхность рельефа")
    plt.tight_layout()
    plt.savefig(path_surface, dpi=150)
    plt.close(fig)

    if path_voxels and land_mask.any():
        # мини-фрагмент
        r, c = land_mask.shape
        rz = min(32, r)
        cz = min(32, c)
        lm = land_mask[:rz, :cz]
        zmax = min(int(height[:rz, :cz].max()), 16)
        if zmax <= 0:
            return
        vol = np.zeros((zmax, rz, cz), dtype=bool)
        for y in range(rz):
            for x in range(cz):
                if not lm[y, x]:
                    continue
                for z in range(min(int(height[y, x]), zmax)):
                    vol[z, y, x] = True
        fig2 = plt.figure(figsize=(8, 6))
        ax2 = fig2.add_subplot(111, projection="3d")
        ax2.voxels(vol, facecolors="0.65", edgecolor="0.3", alpha=0.85)
        ax2.set_title("Фрагмент вокселей (legacy)")
        plt.tight_layout()
        plt.savefig(path_voxels, dpi=120)
        plt.close(fig2)
