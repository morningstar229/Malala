"""
Режим «лабораторная»: пошаговая эволюция CA, настраиваемые B/S, тор, сохранение кадра.

Соответствует идее из переписки: параметры правил, демонстрация влияния на результат.
"""

from __future__ import annotations

import sys

import numpy as np

try:
    import pygame
except ImportError as err:
    raise ImportError(
        "Установите Pygame: pip install pygame (удобнее Python 3.11–3.12)"
    ) from err

from . import day_and_night as dn
from .settings import CELL_SIZE, FPS, GRID_COLS, GRID_ROWS


def parse_rule_list(text: str) -> list[int]:
    text = text.replace(" ", "").strip()
    if not text:
        return []
    return [int(c) for c in text if c.isdigit()]


def run_lab() -> None:
    pygame.init()
    rows, cols = GRID_ROWS, GRID_COLS
    rng = np.random.default_rng()
    grid = dn.random_binary(rows, cols, 0.45, rng)

    birth = [3, 6, 7, 8]
    survival = [3, 4, 6, 7, 8]
    toroidal = False

    w, h = cols * CELL_SIZE + 280, rows * CELL_SIZE + 40
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("ВКР — лаборатория CA (Day & Night и др.)")
    font = pygame.font.SysFont("consolas", 14)
    clock = pygame.time.Clock()

    birth_str = "3678"
    surv_str = "34678"
    active_field = "birth"
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    grid = dn.next_generation(
                        grid,
                        birth=birth,
                        survival=survival,
                        toroidal=toroidal,
                    )
                elif event.key == pygame.K_n:
                    grid = dn.random_binary(rows, cols, 0.45, rng)
                elif event.key == pygame.K_t:
                    toroidal = not toroidal
                elif event.key == pygame.K_TAB:
                    active_field = "surv" if active_field == "birth" else "birth"
                elif event.key == pygame.K_BACKSPACE:
                    if active_field == "birth":
                        birth_str = birth_str[:-1]
                    else:
                        surv_str = surv_str[:-1]
                    birth = parse_rule_list(birth_str) or [3, 6, 7, 8]
                    survival = parse_rule_list(surv_str) or [3, 4, 6, 7, 8]
                elif event.unicode and event.unicode.isdigit():
                    if active_field == "birth":
                        birth_str += event.unicode
                    else:
                        surv_str += event.unicode
                    birth = parse_rule_list(birth_str) or [3, 6, 7, 8]
                    survival = parse_rule_list(surv_str) or [3, 4, 6, 7, 8]

        screen.fill((20, 22, 28))
        surf_w, surf_h = cols * CELL_SIZE, rows * CELL_SIZE
        surf = pygame.Surface((surf_w, surf_h))

        for i in range(rows):
            for j in range(cols):
                color = (40, 90, 140) if grid[i, j] == 0 else (120, 200, 100)
                pygame.draw.rect(surf, color, (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        screen.blit(surf, (10, 10))
        panel_x = surf_w + 24

        lines = [
            "Space — шаг",
            "N — новое поле",
            "T — тор: " + ("вкл" if toroidal else "выкл"),
            "Tab — поле B / S",
            "Цифры — правило B/S",
            "Backspace — назад",
            "",
            f"B [{birth_str}] -> {birth}",
            f"S [{surv_str}] -> {survival}",
            "",
            f"Активно: {'Birth' if active_field == 'birth' else 'Survival'}",
        ]
        y = 12
        for line in lines:
            screen.blit(font.render(line, True, (220, 220, 225)), (panel_x, y))
            y += 18

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    run_lab()
    sys.exit(0)
