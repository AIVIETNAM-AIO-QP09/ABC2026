"""Experiment defaults transcribed from final_code_cgcr.ipynb."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Cycle:
    name: str
    threshold: float | None
    weight: float | None


@dataclass(frozen=True)
class Config:
    fill_value: float = -120.0
    window_sec: float = 5.0
    context_window: int = 5
    smooth_k: int = 5
    min_points: int = 1
    random_state: int = 42
    n_estimators: int = 1000


CYCLES = (
    Cycle("C0", None, None),
    Cycle("C1", 0.85, 0.30),
    Cycle("C2", 0.90, 0.60),
    Cycle("C3", 0.95, 0.45),
)

BEACON_MAP = {
    1: (1, 10), 2: (3, 10), 3: (5, 10), 5: (7, 10), 6: (9, 10),
    19: (10, 10), 7: (13, 10), 8: (15, 10), 10: (17, 10),
    11: (19, 10), 12: (21, 10), 13: (1, 0), 15: (3, 0),
    16: (5, 0), 17: (7, 0), 14: (8, 2), 18: (13, 0),
    20: (15, 0), 21: (17, 0), 22: (19, 0), 23: (21, 0),
    24: (9, 5), 4: (11, 5), 9: (13, 5), 25: (10, 5),
}
NORTH_IDS = frozenset((1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 19))
SOUTH_IDS = frozenset((13, 15, 16, 17, 18, 20, 21, 22, 23, 14))
SIGNATURE_ROOMS = ("501", "513", "516", "nurse station", "523", "506", "kitchen", "cafeteria")
WEAK_CLASSES = frozenset(("501", "502", "503", "505", "510", "513", "515", "516", "517", "518", "cleaning", "cafeteria"))
