from __future__ import annotations

"""Parity and leakage smoke tests for the frozen portable inference bundle."""

import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.inference import (  # noqa: E402
    AUDIT_ONLY_COLUMNS,
    FrozenYouBikeModel,
    load_demo_source,
    select_eligible_june,
)


AUDITED_EXPECTED = [
    ("2026-06-18 17:30:00", 416, 0.06167725473642349, 0, 0),
    ("2026-06-03 07:00:00", 183, 0.5668018460273743, 0, 0),
    ("2026-06-04 17:00:00", 995, 0.5668501853942871, 1, 0),
    ("2026-06-17 08:30:00", 636, 0.571580171585083, 1, 0),
    ("2026-06-19 19:30:00", 1034, 0.651775598526001, 1, 0),
    ("2026-06-16 07:30:00", 1025, 0.6517868638038635, 1, 1),
    ("2026-06-11 17:00:00", 106, 0.8826912045478821, 1, 1),
    ("2026-06-13 18:00:00", 413, 0.5568886995315552, 0, 0),
    ("2026-06-04 09:30:00", 960, 0.3605349659919739, 0, 0),
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FrozenBundleSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = FrozenYouBikeModel()
        cls.source = load_demo_source()

    def test_manifest_files(self) -> None:
        manifest = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
        for item in manifest["files"]:
            path = ROOT / item["path"]
            self.assertTrue(path.is_file(), item["path"])
            self.assertEqual(path.stat().st_size, item["bytes"], item["path"])
            self.assertEqual(_sha256(path), item["sha256"], item["path"])

    def test_feature_allowlist_has_no_outcomes(self) -> None:
        self.assertEqual(len(self.engine.features), 68)
        self.assertFalse(AUDIT_ONLY_COLUMNS.intersection(self.engine.features))

    def test_nine_audited_rows_and_threshold_sides(self) -> None:
        expected = pd.DataFrame(
            AUDITED_EXPECTED,
            columns=["datetime", "station_id", "expected_p", "expected_balanced", "expected_strict"],
        )
        expected["datetime"] = pd.to_datetime(expected["datetime"])
        sample = self.source.merge(
            expected[["datetime", "station_id"]],
            on=["datetime", "station_id"],
            how="inner",
            validate="one_to_one",
        )
        self.assertEqual(len(sample), 9)
        actual = self.engine.predict_frame(sample, attach_stations=False)
        actual = actual.merge(expected, on=["datetime", "station_id"], validate="one_to_one")
        max_diff = float(np.max(np.abs(actual["p_lgbm_full"] - actual["expected_p"])))
        self.assertLessEqual(max_diff, 2e-7)
        np.testing.assert_array_equal(
            actual["alert_secondary_balanced"], actual["expected_balanced"]
        )
        np.testing.assert_array_equal(actual["alert_primary_strict"], actual["expected_strict"])
        self.assertEqual(set(actual["expected_balanced"]), {0, 1})
        self.assertEqual(set(actual["expected_strict"]), {0, 1})

    def test_unknown_station_is_safe_missing_category(self) -> None:
        row = self.source.iloc[[0]].copy()
        row["station_id"] = max(self.engine.category_schema["station_id"]) + 10000
        matrix = self.engine.model_matrix(row)
        self.assertTrue(pd.isna(matrix.iloc[0]["station_id"]))
        prediction = self.engine.predict_frame(row, attach_stations=False)
        self.assertTrue(np.isfinite(prediction["p_lgbm_full"]).all())

    def test_full_eligible_population_matches_frozen_reference(self) -> None:
        eligible = select_eligible_june(self.source)
        actual = self.engine.predict_frame(eligible, attach_stations=False)
        reference = pd.read_parquet(
            ROOT / "data/reference/june_all_eligible_decisions.parquet",
            columns=[
                "datetime",
                "station_id",
                "p_lgbm_full",
                "alert_secondary_balanced",
                "alert_primary_strict",
            ],
        ).rename(columns={"p_lgbm_full": "expected_p"})
        compared = actual.merge(
            reference,
            on=["datetime", "station_id"],
            suffixes=("", "_expected"),
            validate="one_to_one",
        )
        self.assertEqual(len(eligible), 27_962)
        self.assertEqual(len(compared), len(reference))
        max_diff = float(np.max(np.abs(compared["p_lgbm_full"] - compared["expected_p"])))
        self.assertLessEqual(max_diff, 2e-7, f"full-population max probability diff={max_diff}")
        np.testing.assert_array_equal(
            compared["alert_secondary_balanced"],
            compared["alert_secondary_balanced_expected"],
        )
        np.testing.assert_array_equal(
            compared["alert_primary_strict"], compared["alert_primary_strict_expected"]
        )
        print(f"Verified {len(compared):,} eligible rows; max probability diff={max_diff:.3g}")


if __name__ == "__main__":
    unittest.main(verbosity=2)

