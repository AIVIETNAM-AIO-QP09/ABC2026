"""Training and inference orchestration."""

from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from .config import CYCLES, WEAK_CLASSES, Config
from .data import load_synthetic, load_wide, rssi_columns
from .features import add_context, build_features, feature_matrix, majority_smooth, room_signatures


def train(labeled_path: Path, synthetic_path: Path, unlabeled_path: Path,
          model_path: Path, config: Config = Config()) -> dict:
    labeled = load_wide(labeled_path, labeled=True)
    labeled = labeled.loc[labeled["room"] != "hallway"].copy()
    cols = rssi_columns(labeled)
    synthetic = load_synthetic(synthetic_path, cols)
    synthetic = synthetic.loc[synthetic["room"] != "hallway"].copy()
    unlabeled = load_wide(unlabeled_path, reference_rssi=cols)
    combined = pd.concat([labeled, synthetic], ignore_index=True)
    signatures = room_signatures(combined, cols, config)
    base = add_context(build_features(combined, cols, config, signatures), config)
    if base.empty:
        raise ValueError("No homogeneous labeled windows were produced")
    pseudo = pd.DataFrame()
    summary = []
    for index, cycle in enumerate(CYCLES):
        training = pd.concat([base, pseudo], ignore_index=True) if not pseudo.empty else base
        matrix = feature_matrix(training)
        encoder = LabelEncoder()
        target = encoder.fit_transform(training["label"])
        weights = compute_sample_weight(class_weight="balanced", y=target)
        if not pseudo.empty:
            weights[len(base):] = cycle.weight
        weak = np.isin(training["label"].astype(str).to_numpy(), list(WEAK_CLASSES))
        weights[weak] *= 2.5
        model = XGBClassifier(
            n_estimators=config.n_estimators, max_depth=5, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.8, objective="multi:softprob",
            n_jobs=-1, random_state=config.random_state,
        )
        model.fit(matrix, target, sample_weight=weights)
        summary.append({"cycle": cycle.name, "training_windows": len(training),
                        "pseudo_windows": len(pseudo)})
        if index + 1 < len(CYCLES):
            candidate = add_context(build_features(unlabeled, cols, config, signatures), config)
            if candidate.empty:
                pseudo = pd.DataFrame()
                continue
            probabilities = model.predict_proba(feature_matrix(candidate, list(matrix.columns)))
            next_cycle = CYCLES[index + 1]
            confidence = probabilities.max(axis=1)
            selected = confidence >= next_cycle.threshold
            pseudo = candidate.loc[selected].copy()
            if not pseudo.empty:
                pseudo["label"] = encoder.inverse_transform(probabilities[selected].argmax(axis=1))
                pseudo["confidence"] = confidence[selected]
                pseudo["is_pseudo"] = True
    bundle = {"model": model, "encoder": encoder, "features": list(matrix.columns),
              "rssi_cols": cols, "signatures": signatures, "config": asdict(config),
              "summary": summary}
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    return bundle


def predict(model_path: Path, test_path: Path, output_dir: Path) -> tuple[Path, Path]:
    bundle = joblib.load(model_path)
    config = Config(**bundle["config"])
    test = load_wide(test_path, reference_rssi=bundle["rssi_cols"])
    windows = add_context(build_features(test, bundle["rssi_cols"], config,
                                         bundle["signatures"]), config)
    if windows.empty:
        raise ValueError("No test windows were produced")
    matrix = feature_matrix(windows, bundle["features"])
    probabilities = bundle["model"].predict_proba(matrix)
    prediction = majority_smooth(probabilities.argmax(axis=1), config.smooth_k)
    labels = bundle["encoder"].inverse_transform(prediction)
    window_result = pd.DataFrame({"window_id": windows["win_id"],
                                  "timestamp": windows["win_start"],
                                  "predicted_room": labels,
                                  "confidence": probabilities.max(axis=1)})
    for index, room in enumerate(bundle["encoder"].classes_):
        window_result[f"prob_{room}"] = probabilities[:, index]
    lookup = window_result[["timestamp", "predicted_room"]].copy()
    lookup["window_end"] = lookup["timestamp"] + pd.Timedelta(seconds=config.window_sec)
    original = test.copy()
    original["_original_order"] = np.arange(len(original))
    rows = pd.merge_asof(original.sort_values("timestamp"), lookup.sort_values("timestamp"),
                         on="timestamp", direction="backward")
    # Keep all source rows and fail explicitly if an unexpected gap appears.
    if rows["predicted_room"].isna().any() or (rows["timestamp"] >= rows["window_end"]).any():
        raise ValueError("Some test rows could not be mapped to a prediction window")
    rows = rows.sort_values("_original_order").drop(columns=["_original_order", "window_end"])
    output_dir.mkdir(parents=True, exist_ok=True)
    window_path = output_dir / "predictions_windows.csv"
    row_path = output_dir / "predictions_rows.csv"
    window_result.to_csv(window_path, index=False)
    rows.to_csv(row_path, index=False)
    return window_path, row_path
