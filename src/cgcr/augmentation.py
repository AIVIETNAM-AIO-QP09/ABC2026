"""Fold-aware frequency-based statistical synthesis from augmentation_fake.ipynb.

This module needs the already-wide labeled table; it does not recreate the
missing raw-to-wide preprocessing notebook.
"""

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_wide, rssi_columns


FOLD_DAYS = {1: 10, 2: 11, 3: 12, 4: 13}
PROXIES = {(3, "503"): "516", (4, "510"): "511"}


def segments(frame: pd.DataFrame, room: str) -> pd.DataFrame:
    ordered = frame.sort_values("timestamp", kind="stable").copy()
    ordered["segment_id"] = ordered["room"].ne(ordered["room"].shift()).cumsum()
    selected = ordered.loc[ordered["room"].eq(str(room))]
    return selected.groupby("segment_id").agg(
        start=("timestamp", "min"), end=("timestamp", "max")
    ).reset_index(drop=True)


def minority_seeds(frame: pd.DataFrame, fold: int, tau: int = 4) -> list[tuple[str, pd.Timestamp, pd.Timestamp, str]]:
    """Select >2-second training-day seeds for scarce validation classes."""
    if fold not in FOLD_DAYS:
        raise ValueError("fold must be 1..4")
    day = FOLD_DAYS[fold]
    contest_days = frame["timestamp"].dt.day.isin(FOLD_DAYS.values())
    validation = frame.loc[contest_days & frame["timestamp"].dt.day.eq(day)]
    training = frame.loc[contest_days & ~frame["timestamp"].dt.day.eq(day)]
    result = []
    for room in sorted(validation["room"].unique()):
        if room == "hallway":
            continue
        own = segments(training, room)
        if len(own) >= tau:
            continue
        source = PROXIES.get((fold, room), room) if own.empty else room
        for row in segments(training, source).itertuples(index=False):
            if (row.end - row.start).total_seconds() > 2:
                result.append((source, row.start, row.end, room))
    return result


def synthesize(seed: pd.DataFrame, rssi_cols: list[str], rng: np.random.Generator,
               sharpen: float = 2.0) -> pd.DataFrame:
    """Simulate dominant-beacon blocks with empirical durations and transitions."""
    seed = seed.sort_values("timestamp", kind="stable")
    if len(seed) < 2:
        return pd.DataFrame()
    interval = seed["timestamp"].diff().dropna().dt.total_seconds().mean()
    step = pd.Timedelta(milliseconds=max(4, round(interval * 1000)))
    signal = seed[rssi_cols].replace(0, -110.0)
    active = signal.where(signal > -109).idxmax(axis=1)
    valid = signal.gt(-109).any(axis=1)
    active = active.loc[valid]
    if active.empty:
        return pd.DataFrame()
    blocks = []
    for _, indexes in active.groupby(active.ne(active.shift()).cumsum()):
        beacon = indexes.iloc[0]
        values = signal.loc[indexes.index, beacon].to_numpy(dtype=float)
        blocks.append((beacon, values))
    by_beacon = {b: [v for name, v in blocks if name == b] for b in active.unique()}
    counts = Counter(name for name, _ in blocks)
    names = sorted(counts)
    probability = np.array([counts[name] for name in names], dtype=float) ** sharpen
    probability /= probability.sum()
    transitions = {}
    for (before, _), (after, _) in zip(blocks, blocks[1:]):
        transitions.setdefault(before, Counter())[after] += 1
    start = seed["timestamp"].iloc[0]
    end = seed["timestamp"].iloc[-1]
    now = start
    beacon = rng.choice(names, p=probability)
    generated = []
    while now < end:
        examples = by_beacon[beacon]
        example = examples[int(rng.integers(len(examples)))]
        value = float(example[0])
        bounds = (min(map(np.min, examples)), max(map(np.max, examples)))
        observed_deltas = [np.diff(v) for v in examples if len(v) > 1]
        deltas = np.concatenate(observed_deltas) if observed_deltas else np.array([0.0])
        lengths = [len(v) for v in examples]
        for _ in range(int(rng.choice(lengths))):
            if now >= end:
                break
            row = {c: -110.0 for c in rssi_cols}
            row[beacon] = value
            row["timestamp"] = now
            generated.append(row)
            now += step
            value = float(np.clip(value + rng.choice(deltas), *bounds))
        next_counts = transitions.get(beacon)
        if next_counts:
            options = list(next_counts)
            p = np.array([next_counts[b] for b in options], dtype=float) ** sharpen
            beacon = rng.choice(options, p=p / p.sum())
        else:
            beacon = rng.choice(names, p=probability)
    return pd.DataFrame(generated)


def augment(path: Path, output: Path, repeats: int = 6, seed: int = 42) -> pd.DataFrame:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    frame = load_wide(path, labeled=True)
    cols = rssi_columns(frame)
    rng = np.random.default_rng(seed)
    products = []
    for fold in FOLD_DAYS:
        for source, start, end, target in minority_seeds(frame, fold):
            subset = frame.loc[frame["timestamp"].between(start, end) & frame["room"].eq(source)]
            for run in range(repeats):
                generated = synthesize(subset, cols, rng)
                if generated.empty:
                    continue
                generated["location"] = target
                generated["fold"] = fold
                generated["is_augmented"] = True
                generated["source_room"] = source
                generated["run"] = run
                products.append(generated)
    if not products:
        raise ValueError("No synthetic rows generated; inspect input days and room segments")
    result = pd.concat(products, ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result
