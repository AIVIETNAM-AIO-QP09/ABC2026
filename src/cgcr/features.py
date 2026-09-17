"""Window, spatial, signature, and causal context features from the final notebook."""

import numpy as np
import pandas as pd

from .config import BEACON_MAP, NORTH_IDS, SOUTH_IDS, SIGNATURE_ROOMS, Config


def room_signatures(frame: pd.DataFrame, rssi_cols: list[str], config: Config) -> pd.DataFrame:
    labeled = frame.loc[frame["room"].ne("hallway")].copy()
    labeled[rssi_cols] = labeled[rssi_cols].replace(0, config.fill_value)
    return labeled.groupby("room")[rssi_cols].mean()


def build_features(frame: pd.DataFrame, rssi_cols: list[str], config: Config,
                   signatures: pd.DataFrame) -> pd.DataFrame:
    """Extract one row per nonoverlapping 5-second window.

    A labeled window crossing a room boundary is discarded, as in the notebook.
    Unlabeled windows use exactly the same feature path, fixing its pseudo-label bug.
    """
    if frame.empty:
        return pd.DataFrame()
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    values = frame[rssi_cols].replace(0, config.fill_value).to_numpy(dtype=float)
    # Pandas 2.x can store datetime64[us]; normalize before nanosecond arithmetic.
    timestamps = frame["timestamp"].to_numpy(dtype="datetime64[ns]").astype("int64")
    labels = frame["room"].astype(str).to_numpy() if "room" in frame else None
    beacon_ids = [int(c.split("_")[1]) for c in rssi_cols]
    positions = {bid: i for i, bid in enumerate(beacon_ids)}
    targets = {r: signatures.loc[r, rssi_cols].to_numpy(dtype=float)
               for r in SIGNATURE_ROOMS if r in signatures.index}
    rows = []
    i = 0
    while i < len(frame):
        end = timestamps[i] + int(config.window_sec * 1e9)
        j = int(np.searchsorted(timestamps, end, side="left"))
        j = max(j, i + 1)
        if j - i < config.min_points:
            i = j
            continue
        if labels is not None and (labels[i] == "hallway" or np.any(labels[i:j] != labels[i])):
            i = j
            continue
        mean = values[i:j].mean(axis=0)
        row = {"win_id": len(rows), "win_start": frame["timestamp"].iloc[i]}
        if labels is not None:
            row["label"] = labels[i]
        for room, signature in targets.items():
            norm = np.linalg.norm(mean) * np.linalg.norm(signature)
            row[f"cos_sim_{room}"] = float(mean @ signature / norm) if norm else 0.0
        if "cos_sim_501" in row and "cos_sim_513" in row:
            row["sim_diff_501_513"] = row["cos_sim_501"] - row["cos_sim_513"]

        def val(beacon: int) -> float:
            return float(mean[positions[beacon]]) if beacon in positions else config.fill_value

        pairs = {
            "diff_15_13": (15, 13), "diff_1_15": (1, 15),
            "diff_1_2": (1, 2), "diff_15_16": (15, 16),
            "diff_1_13": (1, 13), "diff_kit_cafe_core": (14, 4),
            "diff_kit_gate": (24, 25), "diff_clean_nurse": (19, 9),
            "diff_513_515": (13, 15), "diff_511_512": (11, 12),
            "diff_510_7": (10, 7),
        }
        row.update({name: val(a) - val(b) for name, (a, b) in pairs.items()})
        sw = sx = sy = north = south = 0.0
        for bid, signal in zip(beacon_ids, mean):
            if signal <= -115 or bid not in BEACON_MAP:
                continue
            weight = (signal + 120) ** 2
            x, y = BEACON_MAP[bid]
            sw += weight
            sx += weight * x
            sy += weight * y
            if bid in NORTH_IDS:
                north += weight
            if bid in SOUTH_IDS:
                south += weight
        row["geo_x"] = sx / sw if sw else -1.0
        row["geo_y"] = sy / sw if sw else -1.0
        row["ns_balance"] = north - south
        row["ns_balance_v2"] = north - south  # Retained for model compatibility.
        row.update({f"mean_{col}": float(value) for col, value in zip(rssi_cols, mean)})
        rows.append(row)
        i = j
    return pd.DataFrame(rows)


def add_context(frame: pd.DataFrame, config: Config) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.sort_values("win_start", kind="stable").reset_index(drop=True)
    cols = [c for c in frame if c not in ("win_id", "win_start", "label")]
    means = frame[cols].rolling(config.context_window, min_periods=1).mean()
    return pd.concat([frame, means.add_prefix("ctx_mean_"),
                      (frame[cols] - means).add_prefix("ctx_delta_")], axis=1)


def feature_matrix(frame: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    data = frame.drop(columns=["win_id", "win_start", "label", "confidence",
                               "is_pseudo"], errors="ignore")
    if columns is not None:
        data = data.reindex(columns=columns)
    return data.fillna(-120.0)


def majority_smooth(prediction: np.ndarray, k: int) -> np.ndarray:
    if k < 1 or k % 2 != 1:
        raise ValueError("smooth_k must be a positive odd integer")
    result = np.array(prediction, copy=True)
    half = k // 2
    for i in range(len(result)):
        result[i] = np.bincount(prediction[max(0, i-half):i+half+1]).argmax()
    return result
