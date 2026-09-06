from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from backend.bedrock_summary import (
    BedrockConfig,
    BedrockSummaryService,
    SUMMARY_DISCLAIMER,
    compose_explain_response,
    controlled_prompt_payload,
    template_operational_summary,
)


STATION_FACTS = {
    "station_id": 673,
    "station_name": "示範站",
    "district": "三重區",
    "decision_time": "2026-06-29T09:00:00+08:00",
    "target_time": "2026-06-29T09:30:00+08:00",
    "risk_probability": 0.6584,
    "red_duration_display": "已持續30分鐘",
    "route_order": 2,
    "policy_label": "平衡模式",
    "threshold": 0.5668,
    "y_same_30": 1,
    "secret_note": "不可送出",
}
SHAP = {
    "base_value_raw": -0.2,
    "raw_score": 0.8,
    "sum_error": 0.0,
    "factors": [
        {
            "feature": "red_duration_capped_6",
            "label": "目前缺車已持續時間",
            "value_display": "30分鐘",
            "contribution": 0.42,
            "direction": "increase",
        },
        {
            "feature": "bike_ratio_lag_1d",
            "label": "前一天同時段可借車比例",
            "value_display": "8.0%",
            "contribution": -0.1,
            "direction": "decrease",
        },
    ],
    "disclaimer": "貢獻值是原始分數影響量，不是機率百分點。",
    "untrusted_extra": "不可送出",
}


class FakeClient:
    def __init__(self, response: object = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, object]] = []

    def converse(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


def converse_response(payload: object) -> dict[str, object]:
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return {"output": {"message": {"content": [{"text": text}]}}}


VALID_MODEL_OUTPUT = {
    "reason_features": ["red_duration_capped_6", "bike_ratio_lag_1d"],
    "emphasis": "duration",
    "disclaimer": SUMMARY_DISCLAIMER,
}


class BedrockSummaryTest(unittest.TestCase):
    def test_template_is_deterministic_and_never_claims_bedrock(self) -> None:
        first = template_operational_summary(STATION_FACTS, SHAP)
        second = template_operational_summary(STATION_FACTS, SHAP)
        self.assertEqual(first, second)
        self.assertEqual(first["provider"], "template")
        self.assertNotIn("fallback_reason", first)
        self.assertIn("65.8%", first["text"])
        self.assertIn("目前缺車已持續時間", first["text"])
        self.assertIn(SUMMARY_DISCLAIMER, first["text"])

    def test_disabled_service_does_not_call_client(self) -> None:
        client = FakeClient(error=AssertionError("must not be called"))
        service = BedrockSummaryService(BedrockConfig(enabled=False), client=client)
        summary = service.summarize(STATION_FACTS, SHAP)
        self.assertEqual(summary["provider"], "template")
        self.assertEqual(summary["fallback_reason"], "bedrock_disabled")
        self.assertEqual(client.calls, [])

    def test_valid_converse_output_is_the_only_way_to_claim_amazon_bedrock(self) -> None:
        client = FakeClient(response=converse_response(VALID_MODEL_OUTPUT))
        service = BedrockSummaryService(
            BedrockConfig(
                enabled=True,
                model_id="test.model-v1",
                region="ap-northeast-1",
                timeout_seconds=2.5,
            ),
            client=client,
        )
        summary = service.summarize(STATION_FACTS, SHAP)

        self.assertEqual(summary["provider"], "amazon_bedrock")
        self.assertNotIn("fallback_reason", summary)
        self.assertIn("65.8%", summary["text"])
        self.assertIn("目前缺車已持續時間", summary["text"])
        self.assertIn("Bedrock將重點歸納為", summary["text"])
        self.assertIn("依SHAP選出的主要訊號", summary["text"])
        self.assertEqual(len(client.calls), 1)
        call = client.calls[0]
        self.assertEqual(call["modelId"], "test.model-v1")
        self.assertEqual(call["inferenceConfig"], {"maxTokens": 300, "temperature": 0})

        user_text = call["messages"][0]["content"][0]["text"]  # type: ignore[index]
        sent = json.loads(user_text)
        self.assertEqual(set(sent), {"station_facts", "shap"})
        self.assertNotIn("y_same_30", sent["station_facts"])
        self.assertNotIn("secret_note", sent["station_facts"])
        self.assertNotIn("untrusted_extra", sent["shap"])
        self.assertNotIn("sum_error", sent["shap"])

    def test_prompt_payload_whitelists_only_facts_and_shap(self) -> None:
        payload = controlled_prompt_payload(STATION_FACTS, SHAP)
        self.assertEqual(
            set(payload["station_facts"]),
            {
                "station_id",
                "station_name",
                "district",
                "decision_time",
                "target_time",
                "risk_probability",
                "red_duration_display",
                "route_order",
                "policy_label",
                "threshold",
            },
        )
        self.assertEqual(
            set(payload["shap"]),
            {"base_value_raw", "raw_score", "factors", "disclaimer"},
        )
        self.assertEqual(
            set(payload["shap"]["factors"][0]),
            {"feature", "label", "value_display", "contribution", "direction"},
        )

    def test_prompt_payload_converts_non_finite_values_to_null(self) -> None:
        facts = {**STATION_FACTS, "risk_probability": float("nan")}
        shap = {**SHAP, "raw_score": float("inf")}
        payload = controlled_prompt_payload(facts, shap)
        self.assertIsNone(payload["station_facts"]["risk_probability"])
        self.assertIsNone(payload["shap"]["raw_score"])
        json.dumps(payload, allow_nan=False)

    def test_timeout_error_falls_back_and_is_not_labelled_bedrock(self) -> None:
        client = FakeClient(error=TimeoutError("offline timeout"))
        service = BedrockSummaryService(
            BedrockConfig(enabled=True, model_id="test.model-v1"), client=client
        )
        summary = service.summarize(STATION_FACTS, SHAP)
        self.assertEqual(summary["provider"], "template")
        self.assertEqual(summary["fallback_reason"], "bedrock_timeout")

    def test_client_error_falls_back_and_is_not_labelled_bedrock(self) -> None:
        client = FakeClient(error=RuntimeError("no AWS in unit tests"))
        service = BedrockSummaryService(
            BedrockConfig(enabled=True, model_id="test.model-v1"), client=client
        )
        summary = service.summarize(STATION_FACTS, SHAP)
        self.assertEqual(summary["provider"], "template")
        self.assertEqual(summary["fallback_reason"], "bedrock_error")

    def test_invalid_json_or_schema_always_uses_template_provider(self) -> None:
        invalid_outputs = [
            "not-json",
            f"```json\n{json.dumps(VALID_MODEL_OUTPUT, ensure_ascii=False)}\n```",
            {**VALID_MODEL_OUTPUT, "weather": "晴天"},
            {**VALID_MODEL_OUTPUT, "disclaimer": "這是派車指令"},
            {**VALID_MODEL_OUTPUT, "reason_features": []},
            {**VALID_MODEL_OUTPUT, "reason_features": ["weather"]},
            {**VALID_MODEL_OUTPUT, "emphasis": "dispatch_now"},
        ]
        for output in invalid_outputs:
            with self.subTest(output=output):
                client = FakeClient(response=converse_response(output))
                service = BedrockSummaryService(
                    BedrockConfig(enabled=True, model_id="test.model-v1"), client=client
                )
                summary = service.summarize(STATION_FACTS, SHAP)
                self.assertEqual(summary["provider"], "template")
                self.assertEqual(summary["fallback_reason"], "invalid_bedrock_output")

    def test_missing_model_id_uses_template_without_creating_aws_client(self) -> None:
        service = BedrockSummaryService(BedrockConfig(enabled=True, model_id=""))
        summary = service.summarize(STATION_FACTS, SHAP)
        self.assertEqual(summary["provider"], "template")
        self.assertEqual(summary["fallback_reason"], "missing_model_id")

    def test_composed_response_has_stable_shap_and_operational_summary_schema(self) -> None:
        service = BedrockSummaryService(BedrockConfig(enabled=False))
        response = compose_explain_response(STATION_FACTS, SHAP, service=service)
        self.assertEqual(set(response), {"shap", "operational_summary"})
        self.assertEqual(response["shap"], SHAP)
        self.assertEqual(response["operational_summary"]["provider"], "template")

    def test_environment_controls_enablement_model_region_and_short_timeout(self) -> None:
        environment = {
            "BEDROCK_ENABLED": "true",
            "BEDROCK_MODEL_ID": "profile.example",
            "BEDROCK_REGION": "us-west-2",
            "BEDROCK_TIMEOUT_SECONDS": "2.25",
            "BEDROCK_MAX_TOKENS": "256",
        }
        with patch.dict(os.environ, environment, clear=True):
            config = BedrockConfig.from_env()
        self.assertTrue(config.enabled)
        self.assertEqual(config.model_id, "profile.example")
        self.assertEqual(config.region, "us-west-2")
        self.assertEqual(config.timeout_seconds, 2.25)
        self.assertEqual(config.max_tokens, 256)


if __name__ == "__main__":
    unittest.main()
