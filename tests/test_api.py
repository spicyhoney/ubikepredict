from __future__ import annotations

import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from backend.api import DemoRequestHandler, DemoService
from backend.inference import AUDIT_ONLY_COLUMNS


class DemoServiceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = DemoService()
        handler = type("TestDemoRequestHandler", (DemoRequestHandler,), {"service": cls.service})
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.server_thread.join(timeout=2)

    def test_representative_balanced_case_matches_frozen_evaluation(self) -> None:
        prediction = self.service.predict(
            {
                "decision_time": "2026-06-29T09:00:00+08:00",
                "district": "三重區",
                "policy": "balanced",
                "action_limit": 10,
            }
        )

        self.assertEqual(prediction["summary"]["current_empty"], 23)
        self.assertEqual(prediction["summary"]["scored_current_empty"], 21)
        self.assertEqual(prediction["summary"]["unscored_current_empty"], 2)
        self.assertEqual(prediction["summary"]["alerts_before_limit"], 12)
        self.assertEqual(prediction["summary"]["action_count"], 10)
        self.assertEqual(prediction["action_limit"], 10)
        self.assertEqual([row["route_order"] for row in prediction["actions"]], list(range(1, 11)))
        self.assertTrue(all(row["in_action_list"] for row in prediction["actions"]))

        outcome = self.service.reveal(
            {
                "decision_time": prediction["decision_time"],
                "station_ids": [row["station_id"] for row in prediction["actions"]],
            }
        )
        self.assertEqual(outcome["summary"]["evaluated_actions"], 10)
        self.assertEqual(outcome["summary"]["hits"], 7)
        self.assertEqual(outcome["summary"]["precision"], 0.7)

    def test_strict_policy_does_not_fill_action_limit(self) -> None:
        prediction = self.service.predict(
            {
                "decision_time": "2026-06-29T09:00:00+08:00",
                "district": "三重區",
                "policy": "strict",
                "action_limit": 10,
            }
        )

        self.assertEqual(prediction["summary"]["alerts_before_limit"], 7)
        self.assertEqual(prediction["summary"]["action_count"], 7)
        self.assertTrue(
            all(
                row["risk_probability"] >= prediction["policy"]["threshold"]
                for row in prediction["actions"]
            )
        )

    def test_unknown_history_is_visible_but_not_scored(self) -> None:
        prediction = self.service.predict(
            {
                "decision_time": "2026-06-29T09:00:00+08:00",
                "district": "三重區",
                "policy": "balanced",
                "action_limit": 10,
            }
        )
        unknown = [row for row in prediction["candidates"] if row["status"] == "insufficient_history"]

        self.assertEqual(len(unknown), 2)
        self.assertTrue(all(row["risk_probability"] is None for row in unknown))
        self.assertTrue(all(not row["is_alert"] for row in unknown))
        self.assertTrue(all(row["longitude"] is not None for row in unknown))
        self.assertTrue(all(row["latitude"] is not None for row in unknown))

    def test_prediction_input_is_physically_truth_free(self) -> None:
        self.assertTrue(AUDIT_ONLY_COLUMNS.isdisjoint(self.service.current_peak.columns))
        self.assertTrue(AUDIT_ONLY_COLUMNS.isdisjoint(self.service.eligible.columns))
        self.assertTrue(
            AUDIT_ONLY_COLUMNS.isdisjoint(self.service.current_peak_by_mode["full_dock"].columns)
        )
        self.assertTrue(
            AUDIT_ONLY_COLUMNS.isdisjoint(self.service.eligible_by_mode["full_dock"].columns)
        )

    def test_full_dock_mode_uses_independent_model_and_truth(self) -> None:
        options = self.service.options(mode="full_dock")
        self.assertEqual(options["mode"], "full_dock")
        self.assertEqual(options["default"]["mode"], "full_dock")
        self.assertEqual(options["default"]["district"], "板橋區")
        self.assertAlmostEqual(options["policies"][0]["threshold"], 0.42059604096412656)

        prediction = self.service.predict(
            {
                "mode": "full_dock",
                "decision_time": "2026-06-25T08:00:00+08:00",
                "district": "板橋區",
                "policy": "balanced",
                "action_limit": 10,
            }
        )
        self.assertEqual(prediction["mode"], "full_dock")
        self.assertEqual(prediction["summary"]["current_full_dock"], 6)
        self.assertEqual(prediction["summary"]["scored_current_full_dock"], 6)
        self.assertEqual(prediction["summary"]["alerts_before_limit"], 2)
        self.assertEqual(prediction["summary"]["action_count"], 2)

        outcome = self.service.reveal(
            {
                "mode": "full_dock",
                "decision_time": prediction["decision_time"],
                "station_ids": [row["station_id"] for row in prediction["actions"]],
            }
        )
        self.assertEqual(outcome["mode"], "full_dock")
        self.assertEqual(outcome["summary"], {"evaluated_actions": 2, "hits": 1, "precision": 0.5})
        self.assertTrue(all("still_full_dock" in row for row in outcome["results"]))

        explanation = self.service.explain(
            {
                "mode": "full_dock",
                "decision_time": prediction["decision_time"],
                "district": prediction["district"],
                "station_id": prediction["actions"][0]["station_id"],
                "policy": "balanced",
                "action_limit": 10,
            }
        )
        self.assertEqual(explanation["mode"], "full_dock")
        self.assertIn("滿柱", explanation["operational_summary"]["text"])
        self.assertNotIn("仍缺車風險", explanation["operational_summary"]["text"])

    def test_explain_uses_model_inputs_without_loading_truth(self) -> None:
        prediction = self.service.predict(
            {
                "decision_time": "2026-06-29T09:00:00+08:00",
                "district": "三重區",
                "policy": "balanced",
                "action_limit": 10,
            }
        )
        station = prediction["actions"][0]
        self.service._reference = None

        explanation = self.service.explain(
            {
                "decision_time": prediction["decision_time"],
                "district": prediction["district"],
                "station_id": station["station_id"],
                "policy": prediction["policy"]["id"],
                "action_limit": prediction["action_limit"],
            }
        )

        self.assertIsNone(self.service._reference)
        self.assertEqual(explanation["station"]["station_id"], station["station_id"])
        self.assertAlmostEqual(
            explanation["station"]["risk_probability"], station["risk_probability"], places=6
        )
        self.assertLess(abs(explanation["shap"]["sum_error"]), 1e-6)
        self.assertGreaterEqual(len(explanation["shap"]["factors"]), 1)
        self.assertEqual(explanation["operational_summary"]["provider"], "template")
        self.assertIn(
            f"第{station['route_order']}順位",
            explanation["operational_summary"]["text"],
        )

    def test_http_explain_rejects_unscored_station(self) -> None:
        prediction = self.service.predict(
            {
                "decision_time": "2026-06-29T09:00:00+08:00",
                "district": "三重區",
                "policy": "balanced",
                "action_limit": 10,
            }
        )
        station_id = next(
            row["station_id"]
            for row in prediction["candidates"]
            if row["status"] == "insufficient_history"
        )
        request = Request(
            f"{self.base_url}/api/explain",
            data=json.dumps(
                {
                    "decision_time": prediction["decision_time"],
                    "district": prediction["district"],
                    "station_id": station_id,
                    "policy": "balanced",
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(HTTPError) as raised:
            urlopen(request, timeout=5)
        self.assertEqual(raised.exception.code, 400)
        payload = json.loads(raised.exception.read().decode("utf-8"))
        self.assertIn("缺少完整歷史", payload["error"])

    def test_reveal_deduplicates_station_ids_before_scoring(self) -> None:
        prediction = self.service.predict(
            {
                "decision_time": "2026-06-29T09:00:00+08:00",
                "district": "三重區",
                "policy": "balanced",
                "action_limit": 10,
            }
        )
        station_id = prediction["actions"][0]["station_id"]
        outcome = self.service.reveal(
            {
                "decision_time": prediction["decision_time"],
                "station_ids": [station_id] * 10,
            }
        )
        self.assertEqual(outcome["summary"]["evaluated_actions"], 1)
        self.assertEqual(len(outcome["results"]), 1)

    def test_reference_path_does_not_follow_relocated_model(self) -> None:
        original_model_path = self.service.engine.model_path
        self.service.engine.model_path = Path("C:/temporary-model-cache/lgbm_full.txt")
        self.service._reference = None
        try:
            self.assertEqual(len(self.service.reference), 27_962)
        finally:
            self.service.engine.model_path = original_model_path

    def test_http_rejects_array_payload_and_unknown_time_cleanly(self) -> None:
        cases = [
            (b"[]", "JSON最外層必須是物件"),
            (
                json.dumps(
                    {
                        "decision_time": "2026-07-15T09:00:00+08:00",
                        "district": "三重區",
                        "policy": "balanced",
                        "action_limit": 10,
                    }
                ).encode("utf-8"),
                "decision_time 不在六月可評估的尖峰時點",
            ),
        ]
        for body, expected_message in cases:
            request = Request(
                f"{self.base_url}/api/predict",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(HTTPError) as raised:
                urlopen(request, timeout=5)
            self.assertEqual(raised.exception.code, 400)
            payload = json.loads(raised.exception.read().decode("utf-8"))
            self.assertEqual(payload["error"], expected_message)

    def test_cors_allows_local_ui_but_not_unknown_origin(self) -> None:
        allowed = Request(
            f"{self.base_url}/api/health",
            headers={"Origin": "http://localhost:3000"},
        )
        with urlopen(allowed, timeout=5) as response:
            self.assertEqual(response.headers.get("Access-Control-Allow-Origin"), "http://localhost:3000")

        unknown = Request(
            f"{self.base_url}/api/health",
            headers={"Origin": "https://unconfigured.example"},
        )
        with urlopen(unknown, timeout=5) as response:
            self.assertIsNone(response.headers.get("Access-Control-Allow-Origin"))


if __name__ == "__main__":
    unittest.main()
