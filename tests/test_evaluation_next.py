from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_evaluation_next", ROOT / "scripts/run_evaluation_next.py"
)
assert SPEC and SPEC.loader
evaluation_next = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation_next)


class EvaluationNextTest(unittest.TestCase):
    def test_ablation_feature_sets_are_independent_and_have_fixed_counts(self) -> None:
        freeze = json.loads(
            (ROOT / "config/final_policy_freeze_before_may.json").read_text(encoding="utf-8")
        )
        full = freeze["features"]["lgbm_full"]
        feature_sets = evaluation_next._feature_sets(full)
        self.assertEqual(
            {name: len(features) for name, features in feature_sets.items()},
            {"A0": 68, "A1": 65, "A2": 30, "A3": 60, "A4": 67, "A5": 65},
        )
        self.assertIn("station_id", feature_sets["A1"])
        self.assertNotIn("red_duration_capped_6", feature_sets["A1"])
        self.assertNotIn("station_id", feature_sets["A5"])
        self.assertNotIn("longitude", feature_sets["A5"])
        self.assertIn("district_code", feature_sets["A5"])

    def test_long_first_policy_prioritizes_long_events_then_model_score(self) -> None:
        frame = pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2026-06-01 08:00"] * 4),
                "station_id": [1, 2, 3, 4],
                "long_flag": [0, 1, 1, 0],
                "p_lgbm_full": [0.99, 0.40, 0.80, 0.70],
            }
        )
        ranked = evaluation_next._rank(frame, "LONG_FIRST_F0")
        self.assertEqual(ranked["station_id"].tolist(), [3, 2, 1, 4])

    def test_exchange_analysis_preserves_equal_exclusive_counts(self) -> None:
        base = {
            "datetime": pd.to_datetime(["2026-06-01 08:00"] * 2),
            "duration_bin": ["B0", "B3"],
            "red_duration_steps": [0, 6],
            "y_same_30": [0, 1],
        }
        f0 = pd.DataFrame({**base, "station_id": [1, 2]})
        duration = pd.DataFrame({**base, "station_id": [1, 3]})
        summary, rows = evaluation_next._exchange_analysis(f0, duration, 2)
        self.assertEqual(int(rows["membership"].eq("f0_only").sum()), 1)
        self.assertEqual(int(rows["membership"].eq("duration_only").sum()), 1)
        self.assertEqual(
            int(summary.loc[summary["membership"].eq("both") & summary["duration_bin"].eq("ALL"), "rows"].iloc[0]),
            1,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
