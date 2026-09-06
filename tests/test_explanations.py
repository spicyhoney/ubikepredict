from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from backend.explanations import (
    FEATURE_LABELS,
    SHAP_DISCLAIMER,
    display_feature_value,
    explain_frame,
    explain_row,
    split_factors,
)
from backend.inference import FrozenYouBikeModel, load_demo_source, select_eligible_june


class ExplanationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = FrozenYouBikeModel()
        cls.rows = select_eligible_june(load_demo_source()).head(3)

    def test_all_68_frozen_features_have_fixed_chinese_labels(self) -> None:
        self.assertEqual(len(self.engine.features), 68)
        self.assertEqual(set(self.engine.features), set(FEATURE_LABELS))
        self.assertTrue(all(label.strip() for label in FEATURE_LABELS.values()))

    def test_contributions_and_base_value_reconstruct_raw_score(self) -> None:
        explanations = explain_frame(
            self.engine,
            self.rows,
            top_positive=4,
            top_negative=3,
        )
        matrix = self.engine.model_matrix(self.rows)
        direct = np.asarray(
            self.engine.booster.predict(matrix, pred_contrib=True), dtype=np.float64
        )
        raw = np.asarray(
            self.engine.booster.predict(matrix, raw_score=True), dtype=np.float64
        )

        self.assertEqual(direct.shape, (3, 69))
        np.testing.assert_allclose(direct[:, :-1].sum(axis=1) + direct[:, -1], raw)
        for index, explanation in enumerate(explanations):
            self.assertAlmostEqual(explanation["base_value_raw"], direct[index, -1])
            self.assertAlmostEqual(explanation["raw_score"], raw[index])
            self.assertLess(abs(explanation["sum_error"]), 1e-10)
            self.assertEqual(explanation["disclaimer"], SHAP_DISCLAIMER)

    def test_factors_expose_raw_name_label_value_direction_and_contribution(self) -> None:
        explanation = explain_row(
            self.engine,
            self.rows.iloc[[0]],
            top_positive=3,
            top_negative=2,
        )
        self.assertEqual(len(explanation["factors"]), 5)
        for factor in explanation["factors"]:
            self.assertEqual(
                set(factor),
                {"feature", "label", "value_display", "contribution", "direction"},
            )
            self.assertIn(factor["feature"], self.engine.features)
            self.assertEqual(factor["label"], FEATURE_LABELS[factor["feature"]])
            self.assertIsInstance(factor["value_display"], str)
            self.assertIsInstance(factor["contribution"], float)
            if factor["direction"] == "increase":
                self.assertGreater(factor["contribution"], 0)
            else:
                self.assertEqual(factor["direction"], "decrease")
                self.assertLess(factor["contribution"], 0)

        split = split_factors(explanation)
        self.assertEqual(len(split["increase"]), 3)
        self.assertEqual(len(split["decrease"]), 2)

    def test_feature_values_are_business_readable_without_probability_claims(self) -> None:
        self.assertEqual(display_feature_value("bike_ratio", 0.125), "12.5%")
        self.assertEqual(display_feature_value("delta_bikes_ratio_30", -0.125), "-12.5個百分點")
        self.assertEqual(display_feature_value("target_hour", 7), "07:00")
        self.assertEqual(display_feature_value("target_weekday", 5), "星期六")
        self.assertEqual(display_feature_value("target_is_peak", 1), "是")
        self.assertEqual(display_feature_value("red_duration_capped_6", 6), "180分鐘以上")
        self.assertEqual(display_feature_value("bike_ratio", np.nan), "缺值")

    def test_explainer_rejects_a_broken_contribution_sum(self) -> None:
        class BrokenBooster:
            def predict(self, matrix: pd.DataFrame, **kwargs: object) -> np.ndarray:
                if kwargs.get("pred_contrib"):
                    return np.asarray([[0.25, 0.5]])
                return np.asarray([9.0])

        class BrokenEngine:
            features = ["available_bikes"]
            booster = BrokenBooster()

            @staticmethod
            def model_matrix(frame: pd.DataFrame) -> pd.DataFrame:
                return frame[["available_bikes"]]

        with self.assertRaisesRegex(RuntimeError, "do not reconstruct"):
            explain_frame(  # type: ignore[arg-type]
                BrokenEngine(), pd.DataFrame({"available_bikes": [0]})
            )

    def test_empty_frame_needs_no_model_call(self) -> None:
        self.assertEqual(explain_frame(self.engine, self.rows.iloc[0:0]), [])


if __name__ == "__main__":
    unittest.main()
