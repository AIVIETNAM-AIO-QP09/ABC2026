# Notebook migration map

| Historical code | New location | Notes |
| --- | --- | --- |
| Organizer label and BLE tutorials | `src/cgcr/raw.py`, `config/beacon_map.json` | User/time filters plus exact MAC mapping follow tutorials; one-second aggregation and conservative interval joining are reconstruction choices. |
| `augmentation_fake.ipynb`: fold segment search, proxy rooms, simulation | `src/cgcr/augmentation.py` | Pure functions with explicit RNG seed and output path; historical manually edited timestamp list replaced by deterministic segment discovery. |
| `final_code_cgcr.ipynb`: configuration | `src/cgcr/config.py` | Original cycle values and beacon geometry retained. |
| `final_code_cgcr.ipynb`: CSV loading | `src/cgcr/data.py` | Validates timestamps, labels, RSSI names; normalizes synthetic sentinel values. |
| `final_code_cgcr.ipynb`: V47 feature functions | `src/cgcr/features.py` | One labeled/unlabeled implementation avoids train/test drift. |
| `final_code_cgcr.ipynb`: C0-C3, test inference | `src/cgcr/pipeline.py` | Explicit functions, persisted model bundle, stable feature order. |
| Notebook execution cells | `src/cgcr/cli.py` | `augment`, `train`, `predict` commands with paths supplied by user. |

The old notebooks are kept unchanged in `notebooks/legacy/`; organizer tutorials are kept unchanged in `notebooks/organizer/`. Outputs saved in notebook cells are historical evidence, not current test results. The supplied floor map is copied to `docs/floor-map-source.png` for interpreting physical beacon numbers.

## Corrections made during migration

- The old pseudo-label function passed an unlabeled table to a feature function that unconditionally read `room`. The shared feature function handles both schemas.
- Pandas may store timestamps in microseconds. Window arithmetic now explicitly converts to nanoseconds.
- The old row mapping discarded any test row outside a prediction window without failing. The new inference raises an error and preserves source row order.
- Paths are command arguments instead of Colab/Kaggle drive paths.
- A model bundle stores the exact RSSI and feature columns with signatures and label encoder, so inference does not silently use a different schema.

## Reproducibility boundary

The supplied paper reports LODO cross-validation. The only final modeling notebook here performs full-data training and test inference. Its earlier LODO training script and the original raw-to-wide preprocessing script are absent. The new package therefore cannot verify or reproduce the paper's fold metrics from this repository alone. `augment` follows the historical fold rules and requires labeled wide data; using validation labels to choose seed targets is not appropriate for a blind validation claim. The package's training command is intended for final model fitting with already prepared inputs.
