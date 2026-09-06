from __future__ import annotations

"""Deterministic explanations for the frozen LightGBM model.

LightGBM can calculate TreeSHAP values itself with ``pred_contrib=True``.
This module deliberately does not depend on the separate ``shap`` package and
never turns raw-score contributions into percentage-point claims.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from backend.inference import FrozenYouBikeModel


SHAP_DISCLAIMER = (
    "貢獻值是LightGBM原始分數的影響量，不是機率百分點，也不代表因果關係。"
)


# Fixed labels keep field semantics out of the language-model prompt.  Every
# frozen feature is covered; the fallback is only for forward-compatible use.
FEATURE_LABELS: dict[str, str] = {
    "available_bikes": "目前可借車數",
    "available_docks": "目前可還車位數",
    "capacity": "場站總容量",
    "unavailable_count": "目前不可用車柱數",
    "bike_ratio": "目前可借車比例",
    "dock_ratio": "目前可還車位比例",
    "unavailable_ratio": "目前不可用車柱比例",
    "bikes_lag_30": "30分鐘前可借車數",
    "bikes_lag_60": "60分鐘前可借車數",
    "bikes_lag_90": "90分鐘前可借車數",
    "docks_lag_30": "30分鐘前可還車位數",
    "docks_lag_60": "60分鐘前可還車位數",
    "docks_lag_90": "90分鐘前可還車位數",
    "bike_ratio_lag_30": "30分鐘前可借車比例",
    "bike_ratio_lag_60": "60分鐘前可借車比例",
    "bike_ratio_lag_90": "90分鐘前可借車比例",
    "dock_ratio_lag_30": "30分鐘前可還車位比例",
    "dock_ratio_lag_60": "60分鐘前可還車位比例",
    "dock_ratio_lag_90": "90分鐘前可還車位比例",
    "delta_bikes_30": "近30分鐘車輛數變化",
    "delta_bikes_60": "近60分鐘車輛數變化",
    "delta_bikes_90": "近90分鐘車輛數變化",
    "delta_docks_30": "近30分鐘空位數變化",
    "delta_docks_60": "近60分鐘空位數變化",
    "delta_docks_90": "近90分鐘空位數變化",
    "abs_delta_bikes_30": "近30分鐘車輛變動幅度",
    "delta_bikes_ratio_30": "近30分鐘車輛比例變化",
    "delta_docks_ratio_30": "近30分鐘空位比例變化",
    "bike_acceleration_30": "車輛數變化速度",
    "dock_acceleration_30": "空位數變化速度",
    "bike_ratio_mean_90": "近90分鐘平均可借車比例",
    "bike_ratio_min_90": "近90分鐘最低可借車比例",
    "bike_ratio_max_90": "近90分鐘最高可借車比例",
    "dock_ratio_mean_90": "近90分鐘平均可還車位比例",
    "dock_ratio_min_90": "近90分鐘最低可還車位比例",
    "dock_ratio_max_90": "近90分鐘最高可還車位比例",
    "bike_ratio_lag_1d": "前一天同時段可借車比例",
    "dock_ratio_lag_1d": "前一天同時段可還車位比例",
    "bike_ratio_lag_7d": "前一週同時段可借車比例",
    "dock_ratio_lag_7d": "前一週同時段可還車位比例",
    "target_hour_sin": "預測時段週期位置（正弦）",
    "target_hour_cos": "預測時段週期位置（餘弦）",
    "target_weekday_sin": "預測星期週期位置（正弦）",
    "target_weekday_cos": "預測星期週期位置（餘弦）",
    "target_is_weekend": "預測時點是否為週末",
    "target_is_peak": "預測時點是否為尖峰",
    "lag_90_missing": "近90分鐘歷史是否不足",
    "lag_1d_missing": "前一天歷史是否不足",
    "lag_7d_missing": "前一週歷史是否不足",
    "suspected_large_jump_past": "近期是否出現疑似大量運補跳動",
    "station_id": "此站的歷史風險型態",
    "district_code": "行政區歷史風險型態",
    "target_hour": "預測目標時段",
    "target_weekday": "預測目標星期",
    "longitude": "場站經度位置",
    "latitude": "場站緯度位置",
    "same_red_lag_30": "30分鐘前是否同為缺車",
    "same_red_lag_60": "60分鐘前是否同為缺車",
    "same_red_lag_90": "90分鐘前是否同為缺車",
    "same_red_lag_1d": "前一天同時段是否缺車",
    "same_red_lag_7d": "前一週同時段是否缺車",
    "opposite_red_lag_30": "30分鐘前是否滿柱",
    "opposite_red_lag_60": "60分鐘前是否滿柱",
    "opposite_red_lag_90": "90分鐘前是否滿柱",
    "same_red_count_past_90": "近90分鐘缺車快照數",
    "red_duration_capped_6": "目前缺車已持續時間",
    "red_duration_log1p_capped": "目前缺車持續時間（模型轉換值）",
    "duration_90plus": "目前是否已缺車至少90分鐘",
}


_RATIO_FEATURES = {
    name
    for name in FEATURE_LABELS
    if "ratio" in name and name not in {"delta_bikes_ratio_30", "delta_docks_ratio_30"}
}
_RATIO_DELTA_FEATURES = {"delta_bikes_ratio_30", "delta_docks_ratio_30"}
_BOOLEAN_FEATURES = {
    "target_is_weekend",
    "target_is_peak",
    "lag_90_missing",
    "lag_1d_missing",
    "lag_7d_missing",
    "suspected_large_jump_past",
    "same_red_lag_30",
    "same_red_lag_60",
    "same_red_lag_90",
    "same_red_lag_1d",
    "same_red_lag_7d",
    "opposite_red_lag_30",
    "opposite_red_lag_60",
    "opposite_red_lag_90",
    "duration_90plus",
}
_INTEGER_FEATURES = {
    "available_bikes",
    "available_docks",
    "capacity",
    "unavailable_count",
    "bikes_lag_30",
    "bikes_lag_60",
    "bikes_lag_90",
    "docks_lag_30",
    "docks_lag_60",
    "docks_lag_90",
    "delta_bikes_30",
    "delta_bikes_60",
    "delta_bikes_90",
    "delta_docks_30",
    "delta_docks_60",
    "delta_docks_90",
    "abs_delta_bikes_30",
    "same_red_count_past_90",
    "station_id",
    "district_code",
}
_WEEKDAYS = ("星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日")


def _python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def display_feature_value(feature: str, value: Any) -> str:
    """Format one model input without implying unsupported business meaning."""
    value = _python_scalar(value)
    try:
        missing = value is None or bool(pd.isna(value))
    except (TypeError, ValueError):
        missing = False
    if missing:
        return "缺值"

    numeric: float | None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = None

    if feature in _BOOLEAN_FEATURES and numeric is not None:
        return "是" if numeric >= 0.5 else "否"
    if feature == "target_hour" and numeric is not None:
        return f"{int(numeric):02d}:00"
    if feature == "target_weekday" and numeric is not None:
        weekday = int(numeric)
        return _WEEKDAYS[weekday] if 0 <= weekday < len(_WEEKDAYS) else str(weekday)
    if feature == "station_id" and numeric is not None:
        return f"站點代碼 {int(round(numeric))}"
    if feature == "district_code" and numeric is not None:
        return f"行政區代碼 {int(round(numeric))}"
    if feature == "red_duration_capped_6" and numeric is not None:
        minutes = max(0, int(round(numeric))) * 30
        return "180分鐘以上" if minutes >= 180 else f"{minutes}分鐘"
    if feature in _RATIO_FEATURES and numeric is not None:
        return f"{numeric * 100:.1f}%"
    if feature in _RATIO_DELTA_FEATURES and numeric is not None:
        return f"{numeric * 100:+.1f}個百分點"
    if feature in {"longitude", "latitude"} and numeric is not None:
        return f"{numeric:.5f}"
    if feature in _INTEGER_FEATURES and numeric is not None:
        return f"{int(round(numeric))}"
    if numeric is not None:
        return f"{numeric:.3f}"
    return str(value)


def _factor(feature: str, value: Any, contribution: float) -> dict[str, Any]:
    contribution = float(contribution)
    return {
        "feature": feature,
        "label": FEATURE_LABELS.get(feature, "其他模型訊號"),
        "value_display": display_feature_value(feature, value),
        "contribution": contribution,
        "direction": "increase" if contribution >= 0 else "decrease",
    }


def _select_top_factors(
    feature_names: list[str],
    row: pd.Series,
    contributions: np.ndarray,
    top_positive: int,
    top_negative: int,
) -> list[dict[str, Any]]:
    positive = [index for index, value in enumerate(contributions) if value > 0]
    negative = [index for index, value in enumerate(contributions) if value < 0]
    positive.sort(key=lambda index: (-float(contributions[index]), feature_names[index]))
    negative.sort(key=lambda index: (float(contributions[index]), feature_names[index]))

    selected = positive[:top_positive] + negative[:top_negative]
    factors = [
        _factor(feature_names[index], row.iloc[index], float(contributions[index]))
        for index in selected
    ]
    return sorted(
        factors,
        key=lambda item: (-abs(float(item["contribution"])), item["feature"]),
    )


def explain_frame(
    engine: FrozenYouBikeModel,
    frame: pd.DataFrame,
    *,
    top_positive: int = 3,
    top_negative: int = 2,
    tolerance: float = 1e-6,
) -> list[dict[str, Any]]:
    """Return TreeSHAP explanations for every row in ``frame``.

    The returned ``factors`` contain the strongest positive and negative
    raw-score contributions.  A mismatch between contributions and the model's
    raw score is treated as a hard error rather than silently shown to users.
    """
    if top_positive < 0 or top_negative < 0:
        raise ValueError("top_positive and top_negative must not be negative")
    if tolerance <= 0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be a finite positive number")
    if frame.empty:
        return []

    matrix = engine.model_matrix(frame)
    feature_names = list(engine.features)
    if list(matrix.columns) != feature_names:
        raise RuntimeError("Explanation matrix does not match frozen feature order")

    contribution_matrix = np.asarray(
        engine.booster.predict(matrix, pred_contrib=True), dtype=np.float64
    )
    if contribution_matrix.ndim == 1:
        contribution_matrix = contribution_matrix.reshape(1, -1)
    expected_shape = (len(matrix), len(feature_names) + 1)
    if contribution_matrix.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected LightGBM contribution shape {contribution_matrix.shape}; "
            f"expected {expected_shape}"
        )

    raw_scores = np.asarray(
        engine.booster.predict(matrix, raw_score=True), dtype=np.float64
    ).reshape(-1)
    reconstructed = contribution_matrix.sum(axis=1)
    errors = reconstructed - raw_scores
    if not np.allclose(reconstructed, raw_scores, rtol=tolerance, atol=tolerance):
        worst = float(np.max(np.abs(errors)))
        raise RuntimeError(
            "LightGBM contributions do not reconstruct the raw score "
            f"(maximum absolute error {worst:.3g})"
        )

    explanations: list[dict[str, Any]] = []
    for position in range(len(matrix)):
        contributions = contribution_matrix[position, :-1]
        explanations.append(
            {
                "base_value_raw": float(contribution_matrix[position, -1]),
                "raw_score": float(raw_scores[position]),
                "sum_error": float(errors[position]),
                "factors": _select_top_factors(
                    feature_names,
                    matrix.iloc[position],
                    contributions,
                    top_positive,
                    top_negative,
                ),
                "disclaimer": SHAP_DISCLAIMER,
            }
        )
    return explanations


def explain_row(
    engine: FrozenYouBikeModel,
    row: pd.Series | Mapping[str, Any] | pd.DataFrame,
    *,
    top_positive: int = 3,
    top_negative: int = 2,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Convenience wrapper for a single input row."""
    if isinstance(row, pd.DataFrame):
        if len(row) != 1:
            raise ValueError("explain_row requires exactly one row")
        frame = row
    elif isinstance(row, pd.Series):
        frame = row.to_frame().T
    else:
        frame = pd.DataFrame([dict(row)])
    return explain_frame(
        engine,
        frame,
        top_positive=top_positive,
        top_negative=top_negative,
        tolerance=tolerance,
    )[0]


def split_factors(shap_explanation: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Split the compact factor list for front-end positive/negative sections."""
    factors = list(shap_explanation.get("factors", []))
    return {
        "increase": [factor for factor in factors if factor.get("direction") == "increase"],
        "decrease": [factor for factor in factors if factor.get("direction") == "decrease"],
    }
