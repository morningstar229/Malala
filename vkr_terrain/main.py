"""
Точка входа: полный пайплайн генерации 3D-мира (CA Day & Night) + доска этапов.

Запуск из папки «project generate»:
  python -m vkr_terrain   или  python -m vkr_terrain.desktop_app  — Qt-приложение
  python -m vkr_terrain.main gui — то же (Qt)
  python -m vkr_terrain.app_gui — запасной интерфейс на Tkinter
  python -m vkr_terrain.main      — окно Matplotlib: все этапы + 3D
  python -m vkr_terrain.main lab  — лаборатория CA (нужен Pygame)
  python -m vkr_terrain.main research — автосерии экспериментов ЛР2/ЛР3/ЛР4 в CSV
  python -m vkr_terrain.main --quick — только сохранить PNG, без окна

PNG: output/generation_full.png (полный отчёт), при --legacy также relief_*.png
"""

from __future__ import annotations

import os
import sys

import numpy as np

from . import settings as S
from .params import GenerationParams
from .terrain_3d import run_pipeline_from_params
from .visualize import height_to_rgb, save_3d_plots, save_generation_board


def _run_pygame_height_preview(rgb: np.ndarray) -> None:
    import pygame

    pygame.init()
    px = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    px = np.transpose(px, (1, 0, 2))
    w, h = S.GRID_COLS * S.CELL_SIZE, S.GRID_ROWS * S.CELL_SIZE
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("ВКР — быстрый 2D просмотр карты высот (полный отчёт в output/)")
    surf = pygame.surfarray.make_surface(px)
    surf = pygame.transform.scale(surf, (w, h))
    screen.blit(surf, (0, 0))
    font = pygame.font.SysFont("sans", 15)
    hint = font.render("ESC — выход  |  смотри generation_full.png", True, (255, 255, 255))
    screen.blit(hint, (8, 8))
    pygame.display.flip()
    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                running = False
        clock.tick(S.FPS)
    pygame.quit()


def run_terrain_demo(
    toroidal: bool = False,
    *,
    quick: bool = False,
    legacy_png: bool = False,
    pygame_after: bool = False,
) -> None:
    stages = run_pipeline_from_params(GenerationParams(seed=42, toroidal=toroidal))
    out_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(out_dir, exist_ok=True)

    full_path = os.path.join(out_dir, "generation_full.png")
    save_generation_board(
        stages,
        full_path,
        voxel_max_axis=S.VOXEL_VIEW_MAX_AXIS,
        surface_downsample=S.VOXEL_DOWNSAMPLE,
        show=not quick,
    )
    print(f"Сохранено: {os.path.abspath(full_path)}")

    if legacy_png:
        save_3d_plots(
            stages.heights,
            stages.land_mask,
            os.path.join(out_dir, "relief_surface.png"),
            os.path.join(out_dir, "relief_voxels.png"),
            downsample=S.VOXEL_DOWNSAMPLE,
        )

    if quick and pygame_after:
        rgb = height_to_rgb(stages.heights, stages.land_mask)
        try:
            _run_pygame_height_preview(rgb)
        except (ImportError, OSError):
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(rgb, origin="upper")
            ax.set_title("Карта высот (режим --quick)")
            plt.tight_layout()
            plt.show()


def main() -> None:
    argv = [a.lower() for a in sys.argv[1:]]
    if "research" in argv or "exp" in argv or "experiments" in argv:
        from .experiments import run_all_experiments

        out = run_all_experiments()
        print(f"Готово. Эксперименты сохранены в: {os.path.abspath(out)}")
        return
    if "gui" in argv or "app" in argv:
        from .desktop_app import main as desktop_main

        desktop_main()
        return
    if "lab" in argv:
        try:
            from .lab_app import run_lab
        except ImportError as e:
            print(
                "Режим lab требует Pygame: pip install pygame\n"
                "(на Python 3.13+ может понадобиться Python 3.11–3.12 или pygame-ce)",
                file=sys.stderr,
            )
            raise SystemExit(1) from e
        run_lab()
        return

    toroidal = "--torus" in argv or "--toroidal" in argv
    quick = "--quick" in argv
    legacy = "--legacy" in argv
    pygame_after = "--pygame" in argv

    if pygame_after and not quick:
        print(
            "Подсказка: --pygame имеет смысл вместе с --quick (иначе сначала откроется полная доска).",
            file=sys.stderr,
        )

    run_terrain_demo(
        toroidal=toroidal,
        quick=quick,
        legacy_png=legacy,
        pygame_after=pygame_after,
    )


if __name__ == "__main__":
    main()
