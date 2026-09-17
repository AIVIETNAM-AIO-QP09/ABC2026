"""Input contract for the wide CSVs produced by the missing preprocessing stage."""

from pathlib import Path
import re

import pandas as pd


RSSI_RE = re.compile(r"^rssi_(\d+)$")


def rssi_columns(frame: pd.DataFrame) -> list[str]:
    cols = [c for c in frame if RSSI_RE.fullmatch(c)]
    if not cols:
        raise ValueError("Expected at least one rssi_<beacon_id> column")
    return sorted(cols, key=lambda c: int(c.split("_")[1]))


def load_wide(path: str | Path, *, labeled: bool = False,
              reference_rssi: list[str] | None = None) -> pd.DataFrame:
    """Load a wide BLE table; zero means beacon not observed in the source notebooks."""
    frame = pd.read_csv(path)
    if "timestamp" not in frame:
        raise ValueError(f"{path}: missing timestamp")
    if labeled:
        if "room" not in frame and "location" in frame:
            frame = frame.rename(columns={"location": "room"})
        if "room" not in frame or frame["room"].isna().any():
            raise ValueError(f"{path}: missing or null room/location label")
        frame["room"] = frame["room"].astype(str)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    if frame["timestamp"].isna().any():
        raise ValueError(f"{path}: null timestamp")
    cols = reference_rssi or rssi_columns(frame)
    for col in cols:
        if col not in frame:
            frame[col] = 0.0
        frame[col] = pd.to_numeric(frame[col], errors="raise").fillna(0.0)
    return frame


def load_synthetic(path: str | Path, reference_rssi: list[str]) -> pd.DataFrame:
    frame = load_wide(path, labeled=True, reference_rssi=reference_rssi)
    frame[reference_rssi] = frame[reference_rssi].replace(-110.0, 0.0)
    for col in ("rssi_24", "rssi_25"):
        if col in reference_rssi:
            frame[col] = 0.0
    return frame
