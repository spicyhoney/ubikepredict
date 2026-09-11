from __future__ import annotations

import importlib.util
import math
import os
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_evaluation", ROOT / "scripts/run_evaluation.py"
)
assert SPEC and SPEC.loader
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


class EvaluationHelpersTest(unittest.TestCase):
    def test_average_precision_matches_hand_calculation(self) -> None:
        actual = evaluation.average_precision_score([1, 0, 1], [0.9, 0.8, 0.7])
        self.assertAlmostEqual(actual, (1.0 + 2.0 / 3.0) / 2.0)

    def test_average_precision_treats_equal_scores_as_one_threshold(self) -> None:
        actual = evaluation.average_precision_score([1, 0], [0.5, 0.5])
        self.assertAlmostEqual(actual, 0.5)

    def test_duration_bins_are_fixed(self) -> None:
        actual = evaluation._duration_bin(pd.Series([0, 1, 2, 3, 5, 6, 100])).tolist()
        self.assertEqual(actual, ["B0", "B1", "B1", "B2", "B2", "B3", "B3"])

    def test_rank_rows_uses_station_id_for_fixed_ties(self) -> None:
        frame = pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2026-06-01 08:00"] * 3),
                "station_id": [3, 1, 2],
                "score": [0.8, 0.8, 0.9],
            }
        )
        ranked = evaluation._rank_rows(
            frame, ["datetime"], [("score", False)], "score"
        )
        self.assertEqual(ranked["station_id"].tolist(), [2, 1, 3])
        self.assertEqual(ranked["_rank"].tolist(), [1, 2, 3])

    def test_probability_metrics(self) -> None:
        scores = evaluation._probability_metrics(
            np.array([0.0, 1.0]), np.array([0.25, 0.75])
        )
        self.assertAlmostEqual(scores["brier"], 0.0625)
        self.assertTrue(math.isfinite(scores["log_loss"]))

    def test_equal_width_calibration_bins_include_boundary_values(self) -> None:
        bins = evaluation._calibration_bins(
            np.array([0, 0, 1, 1, 1], dtype=np.int8),
            np.array([0.0, 0.099, 0.1, 0.9, 1.0]),
            score_version="test",
            scheme="equal_width",
        ).set_index("bin")
        self.assertEqual(int(bins.loc[0, "count"]), 2)
        self.assertEqual(int(bins.loc[1, "count"]), 1)
        self.assertEqual(int(bins.loc[9, "count"]), 2)

    def test_shared_priority_produces_nested_top_k(self) -> None:
        frame = pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2026-06-01 08:00"] * 5),
                "station_id": [1, 2, 3, 4, 5],
                "random_priority": np.random.Generator(np.random.PCG64(7)).random(5),
            }
        )
        ranked = evaluation._rank_rows(
            frame,
            ["datetime"],
            [("random_priority", True)],
            "random_priority",
        )
        top_two = set(ranked.loc[ranked["_rank"].le(2), "station_id"])
        top_four = set(ranked.loc[ranked["_rank"].le(4), "station_id"])
        self.assertTrue(top_two < top_four)

    def test_real_population_reproduces_frozen_e22(self) -> None:
        source_path = ROOT / "data/source/dynamic_red_empty_2026_06_input.parquet"
        source = evaluation.load_demo_source(path=source_path, mode="empty")
        self.assertNotIn("y_same_30", source.columns)
        eligible = evaluation.select_eligible_june(source).sort_values(
            ["datetime", "station_id"], kind="mergesort"
        )
        self.assertEqual(len(eligible), 27_962)
        predicted = evaluation.FrozenYouBikeModel(mode="empty").predict_frame(
            eligible, attach_stations=False
        )
        reference = pd.read_parquet(
            ROOT / "data/reference/june_all_eligible_decisions.parquet",
            columns=["datetime", "station_id", "p_lgbm_full", "y_same_30"],
        )
        compared = predicted.merge(
            reference,
            on=["datetime", "station_id"],
            validate="one_to_one",
            suffixes=("", "_stored"),
        )
        self.assertEqual(int(compared["y_same_30"].sum()), 11_921)
        self.assertEqual(
            float(
                np.max(
                    np.abs(
                        compared["p_lgbm_full"] - compared["p_lgbm_full_stored"]
                    )
                )
            ),
            0.0,
        )
        compared["duration_bin"] = evaluation._duration_bin(
            compared["red_duration_steps"]
        )
        model_ranked = evaluation._rank_rows(
            compared, ["datetime"], [("p_lgbm_full", False)], "p_lgbm_full"
        )
        duration_ranked = evaluation._rank_rows(
            compared,
            ["datetime"],
            [("red_duration_steps", False)],
            "red_duration_steps",
        )
        model_top10 = model_ranked.loc[model_ranked["_rank"].le(10)]
        duration_top10 = duration_ranked.loc[duration_ranked["_rank"].le(10)]
        self.assertEqual((len(model_top10), int(model_top10["y_same_30"].sum())), (4_152, 2_574))
        self.assertEqual((len(duration_top10), int(duration_top10["y_same_30"].sum())), (4_152, 2_327))

    def test_explicit_model_assets_override_environment_paths(self) -> None:
        explicit = {
            "model_path": ROOT / "model/lgbm_full.txt",
            "freeze_path": ROOT / "config/final_policy_freeze_before_may.json",
            "protocol_path": ROOT / "config/protocol_frozen_before_june.json",
            "stations_path": ROOT / "data/stations/dim_station.csv",
        }
        environment = {
            "UBIKE_MODEL_PATH": "does-not-exist/model.txt",
            "UBIKE_FREEZE_PATH": "does-not-exist/freeze.json",
            "UBIKE_PROTOCOL_PATH": "does-not-exist/protocol.json",
            "UBIKE_STATIONS_PATH": "does-not-exist/stations.csv",
        }
        with mock.patch.dict(os.environ, environment, clear=False):
            engine = evaluation.FrozenYouBikeModel(mode="empty", **explicit)
        self.assertEqual(engine.model_path.resolve(), explicit["model_path"].resolve())
        self.assertEqual(engine.freeze_path.resolve(), explicit["freeze_path"].resolve())
        self.assertEqual(engine.protocol_path.resolve(), explicit["protocol_path"].resolve())
        self.assertEqual(engine.stations_path.resolve(), explicit["stations_path"].resolve())


if __name__ == "__main__":
    unittest.main(verbosity=2)
