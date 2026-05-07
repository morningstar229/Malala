"""Настраиваемые параметры генерации (для GUI и API)."""

from __future__ import annotations

from dataclasses import dataclass, fields

from . import settings as S


@dataclass
class GenerationParams:
    """Все параметры, которые можно менять в приложении."""

    rows: int = S.GRID_ROWS
    cols: int = S.GRID_COLS
    seed: int | None = 42
    toroidal: bool = False
    p_land: float = S.INITIAL_LAND_PROB
    coastline_iterations: int = S.COASTLINE_ITERATIONS
    max_height_layers: int = S.MAX_HEIGHT_LAYERS
    height_layer_iterations: int = S.HEIGHT_LAYER_ITERATIONS
    layer_noise_probability: float = S.LAYER_NOISE_PROBABILITY
    sea_water_depth: int = S.SEA_WATER_DEPTH
    # Оставлена только модель tapered_height (убывающая высота + CA по слоям)
    height_model: str = S.DEFAULT_HEIGHT_MODEL
    # только визуализация
    voxel_max_axis: int = S.VOXEL_VIEW_MAX_AXIS
    surface_downsample: int = S.VOXEL_DOWNSAMPLE

    def clamp(self) -> None:
        """Подрезать значения в безопасные диапазоны (мутирует поля)."""
        self.rows = max(8, min(512, int(self.rows)))
        self.cols = max(8, min(512, int(self.cols)))
        self.p_land = max(0.05, min(0.95, float(self.p_land)))
        self.coastline_iterations = max(0, min(200, int(self.coastline_iterations)))
        self.max_height_layers = max(1, min(200, int(self.max_height_layers)))
        self.height_layer_iterations = max(1, min(50, int(self.height_layer_iterations)))
        self.layer_noise_probability = max(0.01, min(0.99, float(self.layer_noise_probability)))
        self.sea_water_depth = max(0, min(64, int(self.sea_water_depth)))
        self.voxel_max_axis = max(12, min(128, int(self.voxel_max_axis)))
        self.surface_downsample = max(1, min(16, int(self.surface_downsample)))
        hm = str(self.height_model).lower().strip()
        if hm == "habr_article":
            hm = "tapered_height"
        self.height_model = "tapered_height"

    def to_dict(self) -> dict[str, object]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> GenerationParams:
        """Восстановление из JSON (импорт пресетов)."""
        defaults = cls()
        kw: dict[str, object] = {}
        for f in fields(cls):
            if f.name in d:
                kw[f.name] = d[f.name]
            else:
                kw[f.name] = getattr(defaults, f.name)
        if kw.get("height_model") == "habr_article":
            kw["height_model"] = "tapered_height"
        if kw.get("seed") is not None:
            kw["seed"] = int(kw["seed"])  # type: ignore[arg-type]
        p = cls(**kw)  # type: ignore[arg-type]
        p.clamp()
        return p
