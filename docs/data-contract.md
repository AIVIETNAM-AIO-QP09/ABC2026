# Wide-data contract

The supplied ZIP establishes the raw file formats. Two organizer tutorials establish the user/time filters and the exact MAC-to-beacon mapping. The missing preprocessing notebook produced the historical wide tables, but its aggregation choices are unknown.

| Organizer source | Confirmed rule |
| --- | --- |
| `1_Label location_train.ipynb` | Keep user 97, non-null timestamps, non-deleted `Location` records. |
| `2_BLE_train data_5f.ipynb` | Keep the listed 25 MACs and user 90 rows from April 10 at 13:00 through April 13 at 17:29:59.999 Japan time. |
| `config/beacon_map.json` | MAC list copied in order and checked against the tutorial by a test. |

| Input | Required columns | Historical filename | Use |
| --- | --- | --- | --- |
| Labeled | `timestamp`, `room`, `rssi_<id>` | `wide_data_set.csv` | Room signatures and supervised training |
| Synthetic | `timestamp`, `location`, `rssi_<id>` | `fakedata_labeled_yes_2.csv` | Additional labeled training rows |
| Unlabeled | `timestamp`, `rssi_<id>` | `ble_unlabeled_data_wide.csv` | Pseudo-label generation |
| Test | `timestamp`, `rssi_<id>` | `test_file.csv` | Inference |
| Augmentation source | `timestamp`, `location`, `rssi_<id>` | `final_wide_dataset_23beacons.csv` | Segment selection and synthesis |

`room` and `location` are normalized to string labels on load. RSSI column names must match `rssi_<integer>`. A missing beacon reading was represented by `0` in the input notebooks and changed to `-120 dBm` only during feature extraction. The augmentation notebook emitted `-110` for inactive beacons; final training converted it to `0`. The final notebook also set synthetic beacon 24 and 25 values to zero, which the loader preserves.

All tables need parseable timestamps. The training table establishes beacon order and schema. Missing beacon columns in the other tables are filled with zero. Any other columns are ignored by feature extraction. Keep original input files outside Git under `data/`.

## Raw ZIP and reconstructed preparation

The ZIP contains `Dataset/5f_label_loc_train.csv`, a fifth-floor image, and 4,107 headerless files under `Dataset/BLE Data/`. BLE fields are `user_id,timestamp,name,mac address,RSSI,power` in that order. The conversion uses user 90 rows and cleaned user 97 labels in local Japan time. It groups events into one-second buckets, takes the maximum RSSI for each beacon/second, and labels only complete seconds inside one non-conflicting label interval. Ambiguous seconds stay unlabeled. The filters and mapping follow the tutorials; the bucket and join policies are reconstruction choices.

The ZIP alone does not contain the MAC-to-number map, but the BLE tutorial does. `config/beacon_map.json` contains the exact 25 pairs. Filtering the archive gives **1,673,395 BLE events**, matching the tutorial checkpoint. The current conversion yields **23,202 labeled** and **13,818 unlabeled** wide rows. `cgcr inspect-raw` can also produce a per-day MAC inventory.

## Still unknown

No source establishes the historical aggregation interval, label join policy, duplicate policy, train/test split construction, or whether outliers were removed after organizer filtering. The original test CSV is absent from the ZIP. If a historical wide CSV becomes available, compare several manually traced rows before claiming exact reproduction.
