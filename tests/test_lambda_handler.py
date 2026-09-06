from __future__ import annotations

import base64
import json
import os
import unittest
from unittest.mock import patch

from backend import lambda_handler as handler_module


def event(
    method: str,
    path: str,
    payload: object | None = None,
    *,
    origin: str | None = None,
    base64_encoded: bool = False,
    query: str = "",
) -> dict[str, object]:
    headers: dict[str, str] = {"content-type": "application/json"} if payload is not None else {}
    if origin:
        headers["origin"] = origin
    result: dict[str, object] = {
        "version": "2.0",
        "rawPath": path,
        "rawQueryString": query,
        "headers": headers,
        "requestContext": {"http": {"method": method, "path": path}},
        "isBase64Encoded": base64_encoded,
    }
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False)
        if base64_encoded:
            body = base64.b64encode(body.encode("utf-8")).decode("ascii")
        result["body"] = body
    return result


def decoded(response: dict[str, object]) -> dict[str, object]:
    return json.loads(str(response["body"]))


class FakePredictionService:
    eligible = range(123)

    def options(
        self, requested_time: str | None = None, mode: str = "empty"
    ) -> dict[str, object]:
        return {"requested_time": requested_time, "mode": mode}

    def predict(self, payload: dict[str, object]) -> dict[str, object]:
        return {"prediction": payload}

    def explain(self, payload: dict[str, object]) -> dict[str, object]:
        return {"explanation": payload}


class FakeRevealService:
    def reveal(self, payload: dict[str, object]) -> dict[str, object]:
        return {"reveal": payload}


class LambdaHandlerTest(unittest.TestCase):
    def tearDown(self) -> None:
        handler_module._reset_services_for_tests()

    def test_prediction_routes_http_api_v2(self) -> None:
        handler_module._reset_services_for_tests(
            prediction_factory=FakePredictionService
        )
        health = handler_module.prediction_handler(event("GET", "/api/health"), None)
        options = handler_module.prediction_handler(
            event(
                "GET",
                "/api/options",
                query="decision_time=2026-06-29T09%3A00&mode=full_dock",
            ),
            None,
        )
        prediction = handler_module.prediction_handler(
            event("POST", "/api/predict", {"station": 7}), None
        )
        explanation = handler_module.prediction_handler(
            event("POST", "/api/explain", {"station": 7}, base64_encoded=True), None
        )

        self.assertEqual(health["statusCode"], 200)
        self.assertEqual(decoded(health)["rows"], 123)
        self.assertEqual(decoded(options)["requested_time"], "2026-06-29T09:00")
        self.assertEqual(decoded(options)["mode"], "full_dock")
        self.assertEqual(decoded(prediction)["prediction"], {"station": 7})
        self.assertEqual(decoded(explanation)["explanation"], {"station": 7})

    def test_prediction_handler_cannot_dispatch_truth(self) -> None:
        reveal_created = 0

        def reveal_factory() -> FakeRevealService:
            nonlocal reveal_created
            reveal_created += 1
            return FakeRevealService()

        handler_module._reset_services_for_tests(
            prediction_factory=FakePredictionService,
            reveal_factory=reveal_factory,
        )
        response = handler_module.prediction_handler(
            event("POST", "/api/reveal", {"station_ids": [1]}), None
        )
        self.assertEqual(response["statusCode"], 404)
        self.assertEqual(reveal_created, 0)

    def test_reveal_handler_never_initialises_prediction_service(self) -> None:
        prediction_created = 0

        def prediction_factory() -> FakePredictionService:
            nonlocal prediction_created
            prediction_created += 1
            return FakePredictionService()

        handler_module._reset_services_for_tests(
            prediction_factory=prediction_factory,
            reveal_factory=FakeRevealService,
        )
        response = handler_module.reveal_handler(
            event(
                "POST",
                "/api/reveal",
                {"decision_time": "2026-06-29T09:00:00+08:00", "station_ids": [7]},
            ),
            None,
        )
        self.assertEqual(response["statusCode"], 200)
        self.assertEqual(decoded(response)["reveal"]["station_ids"], [7])
        self.assertEqual(prediction_created, 0)

    def test_options_and_payload_errors_share_json_shape(self) -> None:
        handler_module._reset_services_for_tests(
            prediction_factory=FakePredictionService
        )
        preflight = handler_module.prediction_handler(
            event("OPTIONS", "/api/predict", origin="http://localhost:3000"), None
        )
        wrong_shape = handler_module.prediction_handler(
            event("POST", "/api/predict", [1, 2]), None
        )
        too_large = event("POST", "/api/predict")
        too_large["body"] = "x" * (handler_module.MAX_PAYLOAD_BYTES + 1)
        too_large["headers"] = {"content-type": "application/json"}
        oversized = handler_module.prediction_handler(too_large, None)
        declared_large = event("POST", "/api/predict", {"small": True})
        declared_large["headers"] = {
            "content-type": "application/json",
            "content-length": str(handler_module.MAX_PAYLOAD_BYTES + 1)
        }
        declared_oversized = handler_module.prediction_handler(declared_large, None)
        wrong_content_type = event("POST", "/api/explain", {"station_id": 7})
        wrong_content_type["headers"] = {"content-type": "text/plain"}
        unsupported = handler_module.prediction_handler(wrong_content_type, None)

        self.assertEqual(preflight["statusCode"], 204)
        self.assertEqual(
            preflight["headers"]["Access-Control-Allow-Origin"],
            "http://localhost:3000",
        )
        self.assertEqual(wrong_shape["statusCode"], 400)
        self.assertEqual(set(decoded(wrong_shape)), {"error", "code"})
        self.assertEqual(oversized["statusCode"], 413)
        self.assertEqual(decoded(oversized)["code"], "PAYLOAD_TOO_LARGE")
        self.assertEqual(declared_oversized["statusCode"], 413)
        self.assertEqual(unsupported["statusCode"], 415)
        self.assertEqual(decoded(unsupported)["code"], "UNSUPPORTED_MEDIA_TYPE")

    def test_unlisted_origin_does_not_receive_cors_permission(self) -> None:
        handler_module._reset_services_for_tests(
            prediction_factory=FakePredictionService
        )
        with patch.dict(
            os.environ,
            {"UBIKE_ALLOWED_ORIGINS": "https://demo.example"},
            clear=False,
        ):
            denied = handler_module.prediction_handler(
                event("GET", "/api/health", origin="https://other.example"), None
            )
            allowed = handler_module.prediction_handler(
                event("GET", "/api/health", origin="https://demo.example"), None
            )
        self.assertNotIn("Access-Control-Allow-Origin", denied["headers"])
        self.assertEqual(
            allowed["headers"]["Access-Control-Allow-Origin"],
            "https://demo.example",
        )

    def test_explain_route_is_reserved_even_before_integration(self) -> None:
        class PredictionWithoutExplain:
            eligible: tuple[()] = ()

        handler_module._reset_services_for_tests(
            prediction_factory=PredictionWithoutExplain
        )
        response = handler_module.prediction_handler(
            event("POST", "/api/explain", {"station_id": 7}), None
        )
        self.assertEqual(response["statusCode"], 501)
        self.assertEqual(decoded(response)["code"], "EXPLAIN_NOT_CONFIGURED")

    def test_real_reveal_service_matches_frozen_representative_result(self) -> None:
        handler_module._reset_services_for_tests()
        response = handler_module.reveal_handler(
            event(
                "POST",
                "/api/reveal",
                {
                    "decision_time": "2026-06-29T09:00:00+08:00",
                    "station_ids": [673, 1526, 240, 544, 335, 469, 140, 472, 173, 604],
                },
            ),
            None,
        )
        self.assertEqual(response["statusCode"], 200)
        self.assertEqual(
            decoded(response)["summary"],
            {"evaluated_actions": 10, "hits": 7, "precision": 0.7},
        )

    def test_real_reveal_service_keeps_full_dock_truth_separate(self) -> None:
        handler_module._reset_services_for_tests()
        response = handler_module.reveal_handler(
            event(
                "POST",
                "/api/reveal",
                {
                    "mode": "full_dock",
                    "decision_time": "2026-06-25T08:00:00+08:00",
                    "station_ids": [910, 236],
                },
            ),
            None,
        )
        self.assertEqual(response["statusCode"], 200)
        payload = decoded(response)
        self.assertEqual(payload["mode"], "full_dock")
        self.assertEqual(
            payload["summary"],
            {"evaluated_actions": 2, "hits": 1, "precision": 0.5},
        )

    def test_real_prediction_and_explain_lambda_path_uses_no_truth(self) -> None:
        handler_module._reset_services_for_tests()
        with patch.dict(os.environ, {"BEDROCK_ENABLED": "false"}, clear=False):
            os.environ.pop("UBIKE_RUNTIME_BUCKET", None)
            prediction_response = handler_module.prediction_handler(
                event(
                    "POST",
                    "/api/predict",
                    {
                        "decision_time": "2026-06-29T09:00:00+08:00",
                        "district": "三重區",
                        "policy": "balanced",
                        "action_limit": 10,
                    },
                ),
                None,
            )
            self.assertEqual(prediction_response["statusCode"], 200)
            prediction = decoded(prediction_response)
            self.assertEqual(prediction["summary"]["alerts_before_limit"], 12)
            self.assertEqual(prediction["summary"]["action_count"], 10)

            selected = prediction["actions"][0]
            explanation_response = handler_module.prediction_handler(
                event(
                    "POST",
                    "/api/explain",
                    {
                        "decision_time": prediction["decision_time"],
                        "district": prediction["district"],
                        "station_id": selected["station_id"],
                        "policy": prediction["policy"]["id"],
                        "action_limit": prediction["action_limit"],
                    },
                ),
                None,
            )
        self.assertEqual(explanation_response["statusCode"], 200)
        explanation = decoded(explanation_response)
        self.assertEqual(explanation["station"]["station_id"], selected["station_id"])
        self.assertEqual(explanation["operational_summary"]["provider"], "template")
        self.assertLess(abs(float(explanation["shap"]["sum_error"])), 1e-6)
        self.assertIsNone(handler_module._prediction_service._reference)


if __name__ == "__main__":
    unittest.main()
