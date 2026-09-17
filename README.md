# ABC 2026 — BLE indoor location recognition

![From raw BLE readings to location predictions](docs/assets/pipeline.svg)

[Tiếng Việt](README.vi.md) · [Data contract](docs/data-contract.md) · [Architecture](docs/architecture.md) · [Research evidence](docs/research.md) · [Migration notes](docs/migration.md)

This repository turns the ABC 2026 research notebooks into a Python package for preparing fifth-floor Bluetooth Low Energy (BLE) data, training a **Confidence-Guided Cycle Retraining (CGCR)** location classifier, and predicting rooms from new RSSI readings. It follows the available final-training workflow from *From Noisy Beacons to Precise Location: A Confidence-Guided Cycle Retraining Strategy*.

> **Evidence boundary:** the paper's Leave-One-Day-Out (LODO) scores are reported research results. This repository has a documented final-training pipeline; it does not currently reproduce the paper's LODO evaluation. The lost historical preprocessing notebook was reconstructed from the organizer's raw data and tutorials, with the remaining assumptions stated below.

## Contents

- [Problem and data](#problem-and-data)
- [Method](#method)
- [Quick start](#quick-start)
- [Inputs and outputs](#inputs-and-outputs)
- [Research results and reproducibility](#research-results-and-reproducibility)
- [Repository guide](#repository-guide)
- [Development](#development)

## Problem and data

The task is to infer a room or shared area on the fifth floor from BLE signal strength (RSSI). A receiver carried by **user 90** observed **25 mapped beacons**; the organizer's location records for **user 97** provide training labels. Measurements span April 10–13, 2023. Signal strength changes with distance, walls, device orientation, and collection day, so individual readings can be noisy.

![Organizer's fifth-floor plan with numbered BLE beacons](docs/floor-map-source.png)

The raw archive contains headerless BLE event files and interval labels. `config/beacon_map.json` preserves the MAC-to-beacon order in the organizer's tutorial. The `prepare` command streams the ZIP and builds one row per second with `rssi_1` through `rssi_25`. A zero means the beacon was not observed in that second. The reconstruction uses the **strongest reading per beacon per second** and assigns a room only when the entire second is inside one unambiguous label interval; other seconds remain unlabeled.

For the supplied archive, this filter selected **1,673,395 BLE events**, matching the organizer tutorial checkpoint, and produced **23,202 labeled** plus **13,818 unlabeled** wide rows. These counts describe the current reconstruction; they do not prove that the missing historical preprocessing notebook used the same aggregation or label-join rules. See the [data contract](docs/data-contract.md) for schemas and provenance.

## Method

![Paper figure: augmented data, feature extraction, pseudo-labels, and cycle retraining](docs/assets/paper-framework.png)

1. **Prepare data.** Filter the raw ZIP by organizer user, time, and beacon rules; convert events and labels to wide, one-second tables.
2. **Augment minority rooms.** Generate synthetic signal segments from selected training-day runs. The generator retains the historical fold-specific proxy rules for rooms 503 and 510. Its seeded output is a reconstruction of the missing generator output, not a byte-for-byte copy.
3. **Extract shared features.** Build nonoverlapping five-second windows for labeled and unlabeled rows through the same code path. Features include mean RSSI, room-signature cosine similarity, selected beacon contrasts, signal-weighted spatial coordinates, north–south balance, and rolling context. Labeled windows crossing room boundaries and `hallway` windows are excluded.
4. **Fit C0–C3.** Train an XGBoost classifier on labeled and synthetic windows (C0). Each fitted cycle predicts unlabeled windows; only predictions above the **next** cycle's confidence threshold enter that cycle with its specified sample weight. The pseudo-label set is refreshed after each fit.
5. **Predict.** Save the final model with its feature order, label encoder, room signatures, and configuration. At inference, smooth window classes with a five-window majority vote and map predictions back to test rows in their original order.

![Paper figure: adaptation, expansion, and stabilization across retraining cycles](docs/assets/paper-cycle-retraining.png)

| Cycle fitted | Pseudo-label threshold used to select its input | Pseudo-label sample weight |
| --- | ---: | ---: |
| C0 | — | — |
| C1 | 0.85 | 0.30 |
| C2 | 0.90 | 0.60 |
| C3 | 0.95 | 0.45 |

These settings come from Table I of the supplied paper and are implemented in [`src/cgcr/config.py`](src/cgcr/config.py). The method also uses class balancing and an additional weight for selected weak classes. [Architecture notes](docs/architecture.md) describe the module boundaries.

## Quick start

Requires **Python 3.10+** and [`uv`](https://docs.astral.sh/uv/). Run from the repository root in PowerShell. Keep the raw archive wherever you prefer; data and model outputs below are ignored by Git.

```powershell
uv venv
uv pip install -e .

$archive = 'C:\path\to\ABC2026.zip'
uv run cgcr inspect-raw --archive $archive --report outputs/mac_inventory.csv
uv run cgcr prepare --archive $archive --beacon-map config/beacon_map.json --output-dir data/prepared
uv run cgcr augment --wide data/prepared/labeled_wide.csv --output data/prepared/synthetic.csv --repeats 6 --seed 42
uv run cgcr train --labeled data/prepared/labeled_wide.csv --synthetic data/prepared/synthetic.csv --unlabeled data/prepared/unlabeled_wide.csv --model models/cgcr.joblib
```

`inspect-raw` is optional and writes a MAC inventory for auditing. `prepare` reads the ZIP directly; extraction is unnecessary. Training can take substantially longer than preparation because it fits four XGBoost models.

When a separate test wide CSV is available:

```powershell
uv run cgcr predict --model models/cgcr.joblib --test data/test_file.csv --output-dir outputs
```

The organizer ZIP **does not contain** the final notebook's `test_file.csv`. The historical final-training notebook used `fakedata_labeled_yes_2.csv` as its synthetic input; pass that file to `--synthetic` if you have it and want to use the original intermediate artifact. The `augment` command recreates its generation rules but is not guaranteed to produce identical rows.

## Inputs and outputs

| Stage | Main input | Output |
| --- | --- | --- |
| `inspect-raw` | Raw organizer ZIP | `outputs/mac_inventory.csv` |
| `prepare` | ZIP + `config/beacon_map.json` | `data/prepared/labeled_wide.csv`, `unlabeled_wide.csv`, `preprocessing_summary.json` |
| `augment` | Labeled wide CSV | Synthetic CSV with `location`, RSSI columns, and generation metadata |
| `train` | Labeled, synthetic, and unlabeled wide CSVs | `models/cgcr.joblib` model bundle |
| `predict` | Model bundle + separate test wide CSV | `outputs/predictions_windows.csv` and `predictions_rows.csv` |

Wide CSVs need a parseable `timestamp` and beacon columns named `rssi_<id>`; labeled input needs `room` or `location`. The synthetic loader also accepts the historical generator format. `predictions_windows.csv` contains room probabilities and the highest **unsmoothed** class probability per window; the final room label may change during majority smoothing. `predictions_rows.csv` preserves the test rows and adds `predicted_room`. See [data contract](docs/data-contract.md) for how missing columns and inactive beacons are handled.

## Research results and reproducibility

The supplied paper reports the following **LODO means across held-out days** (Table II). These numbers were **not generated by this refactored package**.

| Cycle | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: |
| C0 | 0.6986 | 0.5742 | 0.7071 |
| C1 | 0.7036 | 0.5848 | 0.7101 |
| C2 | 0.7079 | 0.6131 | 0.7150 |
| **C3** | **0.7091** | **0.6220** | **0.7166** |

![Paper-reported LODO Macro F1 across cycles C0–C3](docs/assets/paper-macro-f1.svg)

The paper's ablation table also reports Macro F1 of **0.4978** for features alone and **0.6220** for the full framework. These are research claims from the supplied PDF; see [research evidence](docs/research.md) and the original paper for evaluation details and fold variability.

| What is checked in this repository | What remains an open reproduction step |
| --- | --- |
| MAC mapping matches the organizer tutorial; raw filtering reaches its BLE event checkpoint. | Exact historical raw-to-wide aggregation and train/test split are unknown because the preprocessing notebook is missing. |
| Raw preparation and seeded synthetic generation ran on the supplied archive; core contract tests pass. | The original synthetic CSV is not reproduced byte-for-byte. |
| Final-training and inference code paths are organized in the package. | The four-cycle XGBoost run and paper LODO table have not been rerun here. |

A fair LODO benchmark needs a day-wise split and fold-local preparation. The reconstructed augmentation procedure selects minority seeds using fold information; validation labels must be kept out of any blind evaluation pipeline. The `train` command instead fits the final model on the supplied labeled data and does **not** print an accuracy score. See [migration notes](docs/migration.md) for other known differences from the notebooks.

## Repository guide

```text
config/beacon_map.json       Verified organizer MAC → beacon ID mapping
src/cgcr/raw.py              ZIP inspection and one-second wide preparation
src/cgcr/augmentation.py     Synthetic minority-room segments
src/cgcr/features.py         Window, signal, spatial, and context features
src/cgcr/pipeline.py         C0–C3 training and row/window inference
src/cgcr/cli.py              cgcr command-line entry point
tests/                       Data and pipeline contract tests
docs/                        Architecture, data contract, research, migration
notebooks/legacy/            Original research notebooks, preserved
notebooks/organizer/         Original organizer tutorials, preserved
```

The notebooks remain as provenance; maintained code is in `src/cgcr/`. Generated data, model bundles, and prediction files are excluded by `.gitignore`.

## Development

```powershell
uv pip install -e '.[dev]'
uv run pytest
uv run ruff check src tests
```

GitHub Actions runs the core tests on Python 3.11 and 3.12. The tests check data contracts and key transformations; they are not a substitute for an end-to-end LODO reproduction.
