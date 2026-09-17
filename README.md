# ABC 2026 BLE indoor localization (CGCR)

[Tiếng Việt](README.vi.md) · [Data contract](docs/data-contract.md) · [Architecture](docs/architecture.md) · [Migration notes](docs/migration.md)

Code for the confidence-guided cycle retraining approach described in
*From Noisy Beacons to Precise Location: A Confidence-Guided Cycle Retraining Strategy*.
The original notebooks remain in `notebooks/legacy/` as historical research records; organizer tutorials are in `notebooks/organizer/`.
The runnable package lives in `src/cgcr`.

```text
config/beacon_map.json     Exact organizer MAC-to-beacon mapping
src/cgcr/raw.py            Stream ZIP, prepare labeled/unlabeled wide CSVs
src/cgcr/augmentation.py   Fold-aware synthetic segments
src/cgcr/features.py       Shared labeled/unlabeled feature extraction
src/cgcr/pipeline.py       C0-C3 training and prediction
src/cgcr/cli.py            Command-line interface
tests/                     Contract tests
docs/                      Design and provenance notes
notebooks/                 Historical sources, kept unchanged
```

## Pipeline

1. **Raw preparation**: the supplied ZIP contains headerless BLE event files and interval labels. `inspect-raw` reports observed MACs; `prepare` uses the organizer's verified 25-beacon map, groups RSSI per second and joins unambiguous labels.
2. **Prepared wide data**: one row per timestamp, one `rssi_<id>` column per beacon, and `room` (or `location`) on labeled rows. Zero represents an unseen beacon.
3. **Augmentation**: select minority-class segments from training days and synthesize dominant-beacon blocks. Fold-specific proxies reproduce the notebook's 503←516 and 510←511 rules.
4. **Features**: five-second windows, room-signature cosine similarity, beacon differences, spatial centroid, and causal rolling context.
5. **Training**: XGBoost with class weights and cycles C0–C3. Confidence thresholds are 0.85, 0.90, 0.95; pseudo-label weights are 0.30, 0.60, 0.45.
6. **Prediction**: majority smoothing over window predictions, then mapping back to every test row. The saved model bundle includes feature order, room signatures, and label mapping.

## Install and run

```powershell
uv venv
uv pip install -e .
$archive = 'C:\path\to\ABC2026.zip'
uv run cgcr inspect-raw --archive $archive --report outputs/mac_inventory.csv
uv run cgcr prepare --archive $archive --beacon-map config/beacon_map.json --output-dir data/prepared
uv run cgcr augment --wide data/prepared/labeled_wide.csv --output data/prepared/synthetic.csv
uv run cgcr train --labeled data/prepared/labeled_wide.csv --synthetic data/prepared/synthetic.csv --unlabeled data/prepared/unlabeled_wide.csv --model models/cgcr.joblib
uv run cgcr predict --model models/cgcr.joblib --test data/test_file.csv --output-dir outputs
```

The historical notebook used `fakedata_labeled_yes_2.csv` for final training. Use that file with `--synthetic` to reproduce that input. The augmentation command is a reconstruction of its generator, not a claim of byte-identical output; synthetic generation is stochastic. `test_file.csv` is not in the supplied ZIP and must be supplied separately for inference. See [data contract](docs/data-contract.md) and [migration notes](docs/migration.md).

## Preprocessing provenance and limits

The source notebook that built the wide CSVs is missing. The supplied organizer tutorials establish the raw schema, user/time filters, and the ordered MAC-to-beacon map. The map is reproduced in `config/beacon_map.json` and checked against the tutorial by a test. `src/cgcr/raw.py` uses local Japan time, per-second maximum RSSI, and only seconds wholly inside one non-conflicting label interval. These aggregation and join choices are documented reconstructions; the exact choices of the lost notebook and its train/test split remain unknown. The raw conversion matches the tutorial's 1,673,395 selected BLE events and currently yields 23,202 labeled and 13,818 unlabeled seconds.

The final notebook's `train` procedure is full-data training plus test inference. The paper's Leave-One-Day-Out results are separate experiments; this package does not claim to reproduce those metrics. In particular, validation labels must never feed augmentation seed selection in a benchmark intended to be blind. The historical notebooks should be treated as the record for exact original experiment details.

## Development

```powershell
uv pip install -e '.[dev]'
uv run pytest
uv run ruff check src tests
```

Generated data, models, and predictions are excluded from Git by `.gitignore`.
The GitHub Actions workflow runs the core contract tests on Python 3.11 and 3.12.
