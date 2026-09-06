from __future__ import annotations

"""Controlled Amazon Bedrock summaries with a deterministic local fallback.

Bedrock is intentionally downstream from LightGBM/TreeSHAP.  It can rewrite
facts into concise Chinese, but it cannot calculate risk, choose an alert, or
invent operational context.  Importing this module never requires AWS or
``boto3``; the SDK is loaded lazily only when an enabled service needs a client.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import json
import logging
import math
import os
from typing import Any, Protocol


LOGGER = logging.getLogger(__name__)


SUMMARY_DISCLAIMER = "此為決策輔助，不是實際派車指令"

_STATION_FACT_FIELDS = (
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
    "mode",
    "event_label",
    "outcome_label",
)
_FACTOR_FIELDS = ("feature", "label", "value_display", "contribution", "direction")

_SYSTEM_PROMPT = """你是YouBike營運摘要助理。你只能從使用者JSON內的SHAP因素選出最值得閱讀的項目，並遵守：
1. 不得創造天氣、活動、捷運人流、需求量、因果關係或未提供的事實。
2. 不得修改風險機率、門檻、排名、路線或模型因素；SHAP是原始分數貢獻，不是機率百分點。
3. 不得下達派車指令，只能說明為何值得關注；資料不足時直接說資料不足。
4. reason_features只能逐字複製輸入shap.factors中的feature，不可自行寫理由。
5. 只能輸出一個JSON物件，不得加Markdown或前後文字。schema必須恰為：
{"reason_features":["輸入中的feature"],"emphasis":"duration|recent_change|historical_pattern|capacity|other","disclaimer":"此為決策輔助，不是實際派車指令"}
"""


class ConverseClient(Protocol):
    """Small injectable surface shared by boto3 and offline test doubles."""

    def converse(self, **kwargs: Any) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class BedrockConfig:
    enabled: bool = False
    model_id: str = ""
    region: str | None = None
    timeout_seconds: float = 4.0
    max_tokens: int = 300

    def __post_init__(self) -> None:
        if not 0.1 <= float(self.timeout_seconds) <= 20.0:
            raise ValueError("Bedrock timeout_seconds must be between 0.1 and 20")
        if not 64 <= int(self.max_tokens) <= 1024:
            raise ValueError("Bedrock max_tokens must be between 64 and 1024")

    @classmethod
    def from_env(cls) -> "BedrockConfig":
        enabled = os.environ.get("BEDROCK_ENABLED", "false").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        return cls(
            enabled=enabled,
            model_id=os.environ.get("BEDROCK_MODEL_ID", "").strip(),
            region=(
                os.environ.get("BEDROCK_REGION", "").strip()
                or os.environ.get("AWS_REGION", "").strip()
                or None
            ),
            timeout_seconds=float(os.environ.get("BEDROCK_TIMEOUT_SECONDS", "4")),
            max_tokens=int(os.environ.get("BEDROCK_MAX_TOKENS", "300")),
        )


def _plain_scalar(value: Any) -> str | int | float | bool | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value[:240]
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        value = value.item()
        return _plain_scalar(value)
    return str(value)[:240]


def controlled_prompt_payload(
    station_facts: Mapping[str, Any], shap_explanation: Mapping[str, Any]
) -> dict[str, Any]:
    """Whitelist the only model-derived and station facts sent to Bedrock."""
    facts = {
        key: _plain_scalar(station_facts[key])
        for key in _STATION_FACT_FIELDS
        if key in station_facts
    }
    factors = []
    for raw_factor in list(shap_explanation.get("factors", []))[:8]:
        if not isinstance(raw_factor, Mapping):
            continue
        factor = {
            key: _plain_scalar(raw_factor[key])
            for key in _FACTOR_FIELDS
            if key in raw_factor
        }
        if {"feature", "label", "contribution", "direction"}.issubset(factor):
            factors.append(factor)
    shap = {
        "base_value_raw": _plain_scalar(shap_explanation.get("base_value_raw")),
        "raw_score": _plain_scalar(shap_explanation.get("raw_score")),
        "factors": factors,
        "disclaimer": str(shap_explanation.get("disclaimer", ""))[:240],
    }
    return {"station_facts": facts, "shap": shap}


def _safe_text(value: Any, fallback: str, maximum: int = 160) -> str:
    text = str(value or "").strip().replace("\x00", "")
    return text[:maximum] if text else fallback


def _probability_text(value: Any) -> str | None:
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return None
    if not 0 <= probability <= 1:
        return None
    return f"{probability:.1%}"


def template_operational_summary(
    station_facts: Mapping[str, Any],
    shap_explanation: Mapping[str, Any],
    *,
    fallback_reason: str | None = None,
) -> dict[str, str]:
    """Build a stable Chinese summary without network or generative AI."""
    station_name = _safe_text(station_facts.get("station_name"), "此站")
    risk = _probability_text(station_facts.get("risk_probability"))
    event_label = _safe_text(station_facts.get("event_label"), "缺車", 12)
    outcome_label = _safe_text(station_facts.get("outcome_label"), f"仍{event_label}", 16)
    duration = _safe_text(
        station_facts.get("red_duration_display"), f"目前為{event_label}狀態"
    )

    first_sentence = f"{station_name}{duration}"
    if risk is not None:
        first_sentence += f"，模型估計30分鐘後{outcome_label}風險為{risk}。"
    else:
        first_sentence += "，模型建議持續關注。"

    factors = [
        factor
        for factor in shap_explanation.get("factors", [])
        if isinstance(factor, Mapping)
    ]
    increases = [factor for factor in factors if factor.get("direction") == "increase"][:3]
    decreases = [factor for factor in factors if factor.get("direction") == "decrease"][:2]

    sections = [first_sentence]
    if increases:
        labels = "、".join(_safe_text(factor.get("label"), "模型訊號", 60) for factor in increases)
        sections.append(f"主要推升訊號為{labels}。")
    else:
        sections.append("目前沒有足夠的主要推升因素可供摘要。")
    if decreases:
        labels = "、".join(_safe_text(factor.get("label"), "模型訊號", 60) for factor in decreases)
        sections.append(f"同時，{labels}降低了模型風險分數。")

    route_order = station_facts.get("route_order")
    try:
        route_order = int(route_order) if route_order is not None else None
    except (TypeError, ValueError):
        route_order = None
    if route_order is not None and route_order > 0:
        sections.append(f"目前列為調度關注清單第{route_order}順位，仍需由營運人員綜合判斷。")
    else:
        sections.append("是否採取行動仍需由營運人員綜合現場條件判斷。")
    sections.append(f"{SUMMARY_DISCLAIMER}。")

    result = {"provider": "template", "text": "".join(sections)}
    if fallback_reason:
        result["fallback_reason"] = fallback_reason
    return result


def _create_bedrock_client(config: BedrockConfig) -> ConverseClient:
    # boto3 is supplied by the Lambda runtime.  The lazy import keeps local
    # inference/tests completely AWS-free.
    import boto3  # type: ignore[import-not-found]
    from botocore.config import Config  # type: ignore[import-not-found]

    timeout = float(config.timeout_seconds)
    network_config = Config(
        connect_timeout=min(timeout, 2.0),
        read_timeout=timeout,
        retries={"max_attempts": 1, "mode": "standard"},
    )
    return boto3.client(
        "bedrock-runtime",
        region_name=config.region,
        config=network_config,
    )


def _extract_converse_content(response: Mapping[str, Any]) -> Any:
    try:
        blocks = response["output"]["message"]["content"]  # type: ignore[index]
    except (KeyError, TypeError):
        raise ValueError("Bedrock response is missing Converse content") from None
    if not isinstance(blocks, list) or len(blocks) != 1 or not isinstance(blocks[0], Mapping):
        raise ValueError("Bedrock Converse content must contain exactly one block")
    block = blocks[0]
    if isinstance(block.get("text"), str):
        return block["text"]
    if isinstance(block.get("json"), Mapping):
        return dict(block["json"])
    raise ValueError("Bedrock Converse block does not contain text or json")


def _validated_model_summary(
    content: Any,
    *,
    allowed_features: set[str],
) -> dict[str, Any]:
    if isinstance(content, str):
        stripped = content.strip()
        if stripped.startswith("```"):
            raise ValueError("Markdown-wrapped output is not accepted")
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            raise ValueError("Bedrock output is not valid JSON") from None
    elif isinstance(content, Mapping):
        payload = dict(content)
    else:
        raise ValueError("Bedrock output must be a JSON object")

    expected_keys = {"reason_features", "emphasis", "disclaimer"}
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError("Bedrock output does not match the required schema")
    reasons = payload["reason_features"]
    if not isinstance(reasons, list) or not 1 <= len(reasons) <= 3:
        raise ValueError("Bedrock reason_features must contain one to three items")
    if any(not isinstance(feature, str) or feature not in allowed_features for feature in reasons):
        raise ValueError("Bedrock selected an unknown SHAP feature")
    if len(set(reasons)) != len(reasons):
        raise ValueError("Bedrock selected duplicate SHAP features")
    emphasis = payload["emphasis"]
    if emphasis not in {"duration", "recent_change", "historical_pattern", "capacity", "other"}:
        raise ValueError("Bedrock emphasis is not allowed")
    disclaimer = _strict_generated_text(payload["disclaimer"], "disclaimer", 40)
    if disclaimer.rstrip("。") != SUMMARY_DISCLAIMER:
        raise ValueError("Bedrock disclaimer was changed")
    return {
        "reason_features": reasons,
        "emphasis": emphasis,
        "disclaimer": SUMMARY_DISCLAIMER,
    }


def _strict_generated_text(value: Any, field: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Bedrock {field} must be text")
    text = value.strip().replace("\x00", "")
    if not text or len(text) > maximum:
        raise ValueError(f"Bedrock {field} is empty or too long")
    return text


def _render_model_summary(
    payload: Mapping[str, Any],
    station_facts: Mapping[str, Any],
    shap_explanation: Mapping[str, Any],
) -> str:
    """Render only server-owned facts after Bedrock selects allow-listed IDs."""
    station_name = _safe_text(station_facts.get("station_name"), "此站")
    risk = _probability_text(station_facts.get("risk_probability"))
    threshold = _probability_text(station_facts.get("threshold"))
    policy = _safe_text(station_facts.get("policy_label"), "目前政策")
    event_label = _safe_text(station_facts.get("event_label"), "缺車", 12)
    outcome_label = _safe_text(station_facts.get("outcome_label"), f"仍{event_label}", 16)
    factor_by_feature = {
        str(factor.get("feature")): factor
        for factor in shap_explanation.get("factors", [])
        if isinstance(factor, Mapping) and isinstance(factor.get("feature"), str)
    }
    selected = [
        factor_by_feature[feature]
        for feature in payload["reason_features"]
        if feature in factor_by_feature
    ]
    reasons = "、".join(
        f"{_safe_text(factor.get('label'), '模型訊號', 60)}"
        f"（{_safe_text(factor.get('value_display'), '目前值', 40)}）"
        for factor in selected
    )
    emphasis_text = {
        "duration": f"{event_label}持續狀態",
        "recent_change": "近期車輛變化",
        "historical_pattern": "歷史時段型態",
        "capacity": "場站容量與可用比例",
        "other": "其他模型訊號",
    }[str(payload["emphasis"])]

    if risk is not None and threshold is not None:
        comparison = "已達" if float(station_facts["risk_probability"]) >= float(station_facts["threshold"]) else "未達"
        opening = f"{station_name}的30分鐘{outcome_label}風險為{risk}，{comparison}{policy}門檻{threshold}。"
    elif risk is not None:
        opening = f"{station_name}的30分鐘{outcome_label}風險為{risk}。"
    else:
        opening = f"{station_name}目前需要持續觀察。"

    route_order = station_facts.get("route_order")
    try:
        route_order = int(route_order) if route_order is not None else None
    except (TypeError, ValueError):
        route_order = None
    action = (
        f"目前列為調度關注清單第{route_order}順位，仍需由營運人員綜合判斷。"
        if route_order is not None and route_order > 0
        else "是否採取行動仍需由營運人員綜合現場條件判斷。"
    )
    return (
        f"{opening}Bedrock將重點歸納為{emphasis_text}，"
        f"依SHAP選出的主要訊號：{reasons}。{action}{SUMMARY_DISCLAIMER}。"
    )


class BedrockSummaryService:
    """Generate a valid summary or return a clearly labelled template."""

    def __init__(
        self,
        config: BedrockConfig | None = None,
        *,
        client: ConverseClient | None = None,
    ) -> None:
        self.config = config or BedrockConfig.from_env()
        self._client = client

    def summarize(
        self,
        station_facts: Mapping[str, Any],
        shap_explanation: Mapping[str, Any],
    ) -> dict[str, str]:
        if not self.config.enabled:
            return template_operational_summary(
                station_facts, shap_explanation, fallback_reason="bedrock_disabled"
            )
        if not self.config.model_id:
            return template_operational_summary(
                station_facts, shap_explanation, fallback_reason="missing_model_id"
            )

        payload = controlled_prompt_payload(station_facts, shap_explanation)
        try:
            if self._client is None:
                self._client = _create_bedrock_client(self.config)
            client = self._client
            response = client.converse(
                modelId=self.config.model_id,
                system=[{"text": _SYSTEM_PROMPT}],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "text": json.dumps(
                                    payload,
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                    allow_nan=False,
                                )
                            }
                        ],
                    }
                ],
                inferenceConfig={
                    "maxTokens": int(self.config.max_tokens),
                    "temperature": 0,
                },
            )
            allowed_features = {
                str(factor["feature"])
                for factor in payload["shap"]["factors"]
                if isinstance(factor, Mapping) and isinstance(factor.get("feature"), str)
            }
            structured = _validated_model_summary(
                _extract_converse_content(response),
                allowed_features=allowed_features,
            )
        except (TimeoutError, ConnectionError):
            LOGGER.warning("Bedrock summary timed out; using deterministic template")
            return template_operational_summary(
                station_facts, shap_explanation, fallback_reason="bedrock_timeout"
            )
        except Exception as error:
            LOGGER.warning(
                "Bedrock summary failed validation or invocation (%s); using template",
                type(error).__name__,
            )
            if isinstance(error, (ValueError, TypeError, KeyError, json.JSONDecodeError)):
                reason = "invalid_bedrock_output"
            elif "timeout" in type(error).__name__.lower():
                reason = "bedrock_timeout"
            else:
                reason = "bedrock_error"
            return template_operational_summary(
                station_facts, shap_explanation, fallback_reason=reason
            )

        return {
            "provider": "amazon_bedrock",
            "text": _render_model_summary(structured, station_facts, shap_explanation),
        }


def summarize_operationally(
    station_facts: Mapping[str, Any],
    shap_explanation: Mapping[str, Any],
    *,
    config: BedrockConfig | None = None,
    client: ConverseClient | None = None,
) -> dict[str, str]:
    """Functional wrapper convenient for handlers and local callers."""
    return BedrockSummaryService(config=config, client=client).summarize(
        station_facts, shap_explanation
    )


def compose_explain_response(
    station_facts: Mapping[str, Any],
    shap_explanation: Mapping[str, Any],
    *,
    service: BedrockSummaryService | None = None,
) -> dict[str, Any]:
    """Compose the stable API fragment used by local and Lambda handlers."""
    summary_service = service or BedrockSummaryService()
    return {
        "shap": dict(shap_explanation),
        "operational_summary": summary_service.summarize(
            station_facts, shap_explanation
        ),
    }
