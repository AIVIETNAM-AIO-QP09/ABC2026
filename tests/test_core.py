"""Small contract tests for the reconstructed, data-independent stages."""

import unittest
from pathlib import Path
import ast
import json
import zipfile

import numpy as np
import pandas as pd

from cgcr.augmentation import minority_seeds, synthesize
from cgcr.config import Config
from cgcr.features import add_context, build_features, majority_smooth, room_signatures
from cgcr.raw import inspect_macs, prepare_wide


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "timestamp": pd.to_datetime([
                "2023-04-10 00:00:00", "2023-04-10 00:00:01",
                "2023-04-10 00:00:03", "2023-04-10 00:00:06",
            ]),
            "room": ["501"] * 4,
            "rssi_1": [-60, -61, -62, -63],
            "rssi_2": [0, 0, -80, -80],
        })
        self.cols = ["rssi_1", "rssi_2"]

    def test_labeled_unlabeled_feature_schema(self):
        config = Config()
        signature = room_signatures(self.frame, self.cols, config)
        labeled = add_context(build_features(self.frame, self.cols, config, signature), config)
        unlabeled = add_context(build_features(self.frame.drop(columns="room"),
                                               self.cols, config, signature), config)
        self.assertEqual(len(labeled), 2)
        self.assertEqual(labeled.drop(columns="label").columns.tolist(), unlabeled.columns.tolist())
        self.assertEqual(labeled["mean_rssi_2"].iloc[0], (-120 - 120 - 80) / 3)

    def test_minority_seed_stays_in_training_days(self):
        frame = self.frame.copy()
        frame.loc[len(frame)] = [pd.Timestamp("2023-04-11 00:00:00"), "501", -61, 0]
        seeds = minority_seeds(frame, 2)
        self.assertEqual(len(seeds), 1)
        self.assertEqual(seeds[0][1].day, 10)

    def test_synthesis_and_smoothing(self):
        generated = synthesize(self.frame, self.cols, np.random.default_rng(42))
        self.assertFalse(generated.empty)
        self.assertEqual(set(self.cols).issubset(generated.columns), True)
        np.testing.assert_array_equal(majority_smooth(np.array([0, 1, 0]), 3), [0, 0, 0])

    def test_raw_zip_to_wide_and_label_boundary(self):
        root = Path("outputs/testfixture")
        root.mkdir(parents=True, exist_ok=True)
        archive = root / "input.zip"
        labels = ("user_id,activity,deleted_at,room,started_at,finished_at\n"
                  "97,Location,,501,2023-04-10 14:00:00+09:00,2023-04-10 14:00:02+09:00\n"
                  "97,Location,2023-04-10,502,2023-04-10 14:00:00+09:00,2023-04-10 14:00:02+09:00\n"
                  "97,Other,,503,2023-04-10 14:00:00+09:00,2023-04-10 14:00:02+09:00\n")
        observations = (
            "90,2023-04-10T14:00:00.100+0900,null,AA:BB:CC:DD:EE:FF,-70,0\n"
            "90,2023-04-10T14:00:00.200+0900,null,AA:BB:CC:DD:EE:FF,-60,0\n"
            "90,2023-04-10T14:00:02.100+0900,null,AA:BB:CC:DD:EE:FF,-65,0\n"
        )
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("Dataset/5f_label_loc_train.csv", labels)
            output.writestr("Dataset/BLE Data/user-ble-id_90_2023-04-10T14_00_00.csv",
                            observations)
        report = root / "macs.csv"
        self.assertEqual(inspect_macs(archive, report), 1)
        mapping = root / "mapping.json"
        mapping.write_text(json.dumps({"AA:BB:CC:DD:EE:FF": 1}), encoding="utf-8")
        summary = prepare_wide(archive, mapping, root / "prepared")
        self.assertEqual(summary["labeled_seconds"], 1)
        self.assertEqual(summary["unlabeled_seconds"], 1)
        labeled = pd.read_csv(root / "prepared/labeled_wide.csv")
        self.assertEqual(labeled.loc[0, "rssi_1"], -60)
        self.assertEqual(labeled.loc[0, "room"], 501)

    def test_committed_beacon_map_matches_organizer_tutorial(self):
        notebook = json.loads(Path("notebooks/organizer/2_BLE_train data_5f.ipynb").read_text(
            encoding="utf-8"))
        code = ast.parse("".join(notebook["cells"][4]["source"]))
        addresses = next(ast.literal_eval(node.value) for node in code.body
                         if isinstance(node, ast.Assign)
                         and any(isinstance(target, ast.Name) and target.id == "mac_list"
                                 for target in node.targets))
        mapping = json.loads(Path("config/beacon_map.json").read_text(encoding="utf-8"))
        self.assertEqual(len(addresses), 25)
        self.assertEqual(mapping, {mac: index for index, mac in enumerate(addresses, 1)})


if __name__ == "__main__":
    unittest.main()
