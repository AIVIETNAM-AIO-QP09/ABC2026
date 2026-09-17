# Research context and evidence

Source: *From Noisy Beacons to Precise Location: A Confidence-Guided Cycle Retraining Strategy*, supplied as `ABC_2026.pdf` with this project. The PDF is not committed to the repository.

The author-supplied [overall training framework](assets/paper-framework.png) and [cycle retraining diagram](assets/paper-cycle-retraining.png) are embedded in both READMEs. They depict the research method, while `src/cgcr/` is the maintained implementation of the available final-training workflow.

The paper studies BLE indoor location recognition across different collection days. Its CGCR method combines synthetic minority-class data, RSSI and spatial features, XGBoost, and iterative pseudo-labeling. The evaluation protocol is Leave-One-Day-Out (LODO) validation. The maintained package includes final training and inference, but the original cross-validation script and historical raw-to-wide notebook were not supplied.

## Cycle configuration (paper Table I)

| Cycle | Confidence threshold | Pseudo-label weight |
| --- | ---: | ---: |
| C0 | — | — |
| C1 | 0.85 | 0.30 |
| C2 | 0.90 | 0.60 |
| C3 | 0.95 | 0.45 |

## Paper-reported LODO results (Table II)

| Cycle | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: |
| C0 | 0.6986 | 0.5742 | 0.7071 |
| C1 | 0.7036 | 0.5848 | 0.7101 |
| C2 | 0.7079 | 0.6131 | 0.7150 |
| C3 | 0.7091 | 0.6220 | 0.7166 |

The README chart uses only the Macro F1 column. These values are paper-reported means, not output from a run of the refactored package. The supplied paper also reports fold variability; consult its Table II for the uncertainty values and its evaluation details.
