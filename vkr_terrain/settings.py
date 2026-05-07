"""Параметры по умолчанию для генерации и окна."""

# Сетка (меньше — быстрее на слабых ПК)
GRID_ROWS = 128
GRID_COLS = 128

# Вероятность «суши» на этапе 1 (~половина — выразительные моря и острова)
INITIAL_LAND_PROB = 0.50

# Число итераций CA для береговой линии (этап 2) — выше → плавнее берег
COASTLINE_ITERATIONS = 26

# Высота рельефа: слои и итераций на слой (этап 3)
MAX_HEIGHT_LAYERS = 32
HEIGHT_LAYER_ITERATIONS = 5

# Шум по слоям (tapered_height)
LAYER_NOISE_PROBABILITY = 0.40

# Модель рельефа (единственная в приложении)
DEFAULT_HEIGHT_MODEL = "tapered_height"

# Окно Pygame
CELL_SIZE = 4
FPS = 60

# 3D-визуализация в отчёте / Qt
VOXEL_DOWNSAMPLE = 1  # 1 — более детальная сглаженная поверхность в окне предпросмотра
VOXEL_VIEW_MAX_AXIS = 56
SEA_WATER_DEPTH = 6  # глубина столбца воды под морем (воксели)
