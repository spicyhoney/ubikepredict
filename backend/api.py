from __future__ import annotations

"""Local JSON API for the historical YouBike dashboard.

The prediction endpoint reads only the sealed June feature cache and the frozen
model. Ground truth is loaded lazily and is available only through /api/reveal.
This mirrors the later S3 input/truth separation without requiring AWS locally.
"""

import argparse
import json
import math
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

from backend.inference import (
    AUDIT_ONLY_COLUMNS,
    JULY_START,
    JUNE_START,
    FrozenYouBikeModel,
    load_demo_source,
    select_eligible_june,
)


TAIPEI_TIMEZONE = "Asia/Taipei"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _configured_path(variable: str, default: str) -> Path:
    path = Path(os.environ.get(variable, default))
    return path if path.is_absolute() else REPO_ROOT / path


ALLOWED_ORIGINS = {
    origin.strip().rstrip("/")
    for origin in os.environ.get(
        "UBIKE_ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
}

SCENARIOS = [
    {
        "id": "representative",
        "label": "代表案例 · 三重早尖峰",
        "decision_time": "2026-06-29T09:00:00+08:00",
        "district": "三重區",
        "note": "同時包含命中與誤報，適合展示完整的歷史回放流程。",
    },
    {
        "id": "high_load",
        "label": "高負載 · 板橋早尖峰",
        "decision_time": "2026-06-22T08:00:00+08:00",
        "district": "板橋區",
        "note": "可評估紅燈較多，適合展示如何縮成有限行動清單。",
    },
    {
        "id": "realistic_evening",
        "label": "寫實案例 · 板橋晚尖峰",
        "decision_time": "2026-06-10T19:30:00+08:00",
        "district": "板橋區",
        "note": "同時包含命中與誤報，呈現模型的真實限制。",
    },
    {
        "id": "weekend",
        "label": "週末觀光 · 淡水傍晚",
        "decision_time": "2026-06-20T17:00:00+08:00",
        "district": "淡水區",
        "note": "非通勤時段案例，用來觀察週末泛化。",
    },
    {
        "id": "showcase",
        "label": "補充案例 · 板橋晚尖峰",
        "decision_time": "2026-06-23T19:30:00+08:00",
        "district": "板橋區",
        "note": "另一個板橋晚尖峰時點，可用來比較不同日期的結果。",
    },
]


def _local_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(TAIPEI_TIMEZONE).tz_localize(None)
    return timestamp


def _iso_taipei(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(TAIPEI_TIMEZONE)
    else:
        timestamp = timestamp.tz_convert(TAIPEI_TIMEZONE)
    return timestamp.isoformat()


def _safe_number(value: Any, digits: int = 6) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def _safe_text(value: Any, fallback: str) -> str:
    if value is None or pd.isna(value):
        return fallback
    text = str(value).strip()
    return text or fallback


def _duration_text(steps: Any) -> str:
    if steps is None or pd.isna(steps):
        return "歷史長度不足"
    value = int(steps)
    if value <= 0:
        return "首次觀察缺車"
    if value >= 3:
        return "已持續90分鐘以上"
    return f"已持續{value * 30}分鐘"


def _haversine_km(left: dict[str, Any], right: dict[str, Any]) -> float:
    if any(left.get(key) is None or right.get(key) is None for key in ("latitude", "longitude")):
        return math.inf
    lat1, lon1 = math.radians(left["latitude"]), math.radians(left["longitude"])
    lat2, lon2 = math.radians(right["latitude"]), math.radians(right["longitude"])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    hav = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * math.asin(math.sqrt(hav))


def _risk_first_nearest_route(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Start at highest risk, then visit the nearest remaining alert."""
    if not rows:
        return []
    remaining = sorted(rows, key=lambda item: (-item["risk_probability"], item["station_id"]))
    ordered = [remaining.pop(0)]
    ordered[0]["distance_from_previous_km"] = 0.0
    while remaining:
        previous = ordered[-1]
        next_row = min(
            remaining,
            key=lambda item: (_haversine_km(previous, item), -item["risk_probability"]),
        )
        remaining.remove(next_row)
        distance = _haversine_km(previous, next_row)
        next_row["distance_from_previous_km"] = None if math.isinf(distance) else round(distance, 3)
        ordered.append(next_row)
    for index, row in enumerate(ordered, start=1):
        row["route_order"] = index
        row["in_action_list"] = True
    return ordered


class DemoService:
    def __init__(self) -> None:
        self.engine = FrozenYouBikeModel()
        self.api_input_path = _configured_path(
            "UBIKE_API_INPUT_PATH",
            "data/source/dynamic_red_empty_2026_06_input.parquet",
        )
        self.reference_path = _configured_path(
            "UBIKE_REFERENCE_PATH",
            "data/reference/june_all_eligible_decisions.parquet",
        )
        source = load_demo_source(self.api_input_path)
        leaked_columns = AUDIT_ONLY_COLUMNS.intersection(source.columns)
        if leaked_columns:
            raise RuntimeError(f"Demo input contains audit-only columns: {sorted(leaked_columns)}")
        station_fields = self.engine.stations[
            ["station_id", "station_name", "district"]
        ]
        peak_mask = (
            source["decision_is_peak"].eq(1)
            & source["datetime"].ge(JUNE_START)
            & source["datetime"].lt(JULY_START)
            & source["target_datetime"].lt(JULY_START)
        )
        self.current_peak = source.loc[peak_mask].merge(
            station_fields, on="station_id", how="left", validate="many_to_one"
        )
        self.eligible = select_eligible_june(source)
        self._reference: pd.DataFrame | None = None

        self.decision_times = sorted(self.eligible["datetime"].drop_duplicates().tolist())
        self.districts_by_time: dict[pd.Timestamp, list[dict[str, Any]]] = {}
        for decision_time, frame in self.current_peak.groupby("datetime", sort=True):
            counts = (
                frame.dropna(subset=["district"])
                .groupby("district", sort=True)
                .size()
                .rename("current_empty_count")
                .reset_index()
            )
            self.districts_by_time[pd.Timestamp(decision_time)] = [
                {"district": str(row.district), "current_empty_count": int(row.current_empty_count)}
                for row in counts.itertuples(index=False)
            ]

    @property
    def reference(self) -> pd.DataFrame:
        if self._reference is None:
            self._reference = pd.read_parquet(
                self.reference_path,
                columns=["datetime", "target_datetime", "station_id", "y_same_30"],
            )
            self._reference["datetime"] = pd.to_datetime(self._reference["datetime"])
            self._reference["target_datetime"] = pd.to_datetime(self._reference["target_datetime"])
        return self._reference

    def options(self, requested_time: Any | None = None) -> dict[str, Any]:
        default = SCENARIOS[0]
        decision_time = _local_timestamp(requested_time or default["decision_time"])
        return {
            "timezone": TAIPEI_TIMEZONE,
            "decision_times": [_iso_taipei(value) for value in self.decision_times],
            "districts": self.districts_by_time.get(decision_time, []),
            "policies": [
                {
                    "id": "balanced",
                    "label": "平衡模式",
                    "threshold": self.engine.balanced_threshold,
                    "description": "保留較多值得關注的持續缺車候選。",
                },
                {
                    "id": "strict",
                    "label": "嚴格模式",
                    "threshold": self.engine.strict_threshold,
                    "description": "減少警示數量，只保留更高機率站點。",
                },
            ],
            "scenarios": SCENARIOS,
            "default": {
                "decision_time": default["decision_time"],
                "district": default["district"],
                "policy": "balanced",
                "action_limit": 10,
            },
        }

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        requested_time = payload.get("decision_time")
        if not requested_time:
            raise ValueError("decision_time 不可為空")
        decision_time = _local_timestamp(requested_time)
        if decision_time not in self.decision_times:
            raise ValueError("decision_time 不在六月可評估的尖峰時點")
        district = str(payload.get("district", "")).strip()
        policy = str(payload.get("policy", "balanced"))
        action_limit = int(payload.get("action_limit", 10))
        if policy not in {"balanced", "strict"}:
            raise ValueError("policy 必須是 balanced 或 strict")
        if not 1 <= action_limit <= 25:
            raise ValueError("action_limit 必須介於1到25")
        if not district:
            raise ValueError("district 不可為空")

        current = self.current_peak.loc[
            self.current_peak["datetime"].eq(decision_time)
            & self.current_peak["district"].eq(district)
        ].drop_duplicates("station_id", keep="last")
        eligible = self.eligible.loc[self.eligible["datetime"].eq(decision_time)].copy()
        scored = self.engine.predict_frame(eligible, attach_stations=True)
        scored = scored.loc[scored["district"].eq(district)].copy()

        threshold = (
            self.engine.balanced_threshold if policy == "balanced" else self.engine.strict_threshold
        )
        scored = scored.sort_values(["p_lgbm_full", "station_id"], ascending=[False, True])
        scored["risk_rank"] = np.arange(1, len(scored) + 1)
        scored["is_alert"] = scored["p_lgbm_full"].ge(threshold)
        alerts = scored.loc[scored["is_alert"]].head(action_limit)

        action_rows = [self._station_record(row, is_scored=True) for _, row in alerts.iterrows()]
        route = _risk_first_nearest_route(action_rows)
        route_by_station = {row["station_id"]: row for row in route}

        scored_by_station = {int(row.station_id): row for row in scored.itertuples(index=False)}
        candidates: list[dict[str, Any]] = []
        for row in current.sort_values("station_id").itertuples(index=False):
            station_id = int(row.station_id)
            scored_row = scored_by_station.get(station_id)
            if scored_row is None:
                candidates.append(self._station_record(row, is_scored=False))
                continue
            item = self._station_record(scored_row, is_scored=True)
            if station_id in route_by_station:
                route_item = route_by_station[station_id]
                item["route_order"] = route_item["route_order"]
                item["distance_from_previous_km"] = route_item["distance_from_previous_km"]
                item["in_action_list"] = True
            candidates.append(item)
        candidates.sort(
            key=lambda item: (
                item["risk_probability"] is None,
                -(item["risk_probability"] or -1),
                item["station_id"],
            )
        )

        target_time = decision_time + pd.Timedelta(minutes=30)
        max_probability = scored["p_lgbm_full"].max() if not scored.empty else None
        known_durations = pd.to_numeric(scored["red_duration_steps"], errors="coerce")
        longest_steps = int(known_durations.max()) if known_durations.notna().any() else None
        return {
            "decision_time": _iso_taipei(decision_time),
            "target_time": _iso_taipei(target_time),
            "timezone": TAIPEI_TIMEZONE,
            "district": district,
            "action_limit": action_limit,
            "policy": {
                "id": policy,
                "label": "平衡模式" if policy == "balanced" else "嚴格模式",
                "threshold": threshold,
            },
            "summary": {
                "current_empty": int(len(current)),
                "scored_current_empty": int(len(scored)),
                "unscored_current_empty": int(len(current) - len(scored)),
                "alerts_before_limit": int(scored["is_alert"].sum()),
                "action_count": int(len(route)),
                "max_risk_probability": _safe_number(max_probability),
                "longest_red_duration": _duration_text(longest_steps),
            },
            "candidates": candidates,
            "actions": route,
            "route_method": "最高風險站起點，再依相鄰距離串接；僅為巡補順序示意。",
            "truth_revealed": False,
        }

    def reveal(self, payload: dict[str, Any]) -> dict[str, Any]:
        requested_time = payload.get("decision_time")
        if not requested_time:
            raise ValueError("decision_time 不可為空")
        decision_time = _local_timestamp(requested_time)
        if decision_time not in self.decision_times:
            raise ValueError("decision_time 不在六月可評估的尖峰時點")
        station_ids = list(dict.fromkeys(int(value) for value in payload.get("station_ids", [])))
        if not station_ids:
            raise ValueError("station_ids 不可為空")
        selected = self.reference.loc[
            self.reference["datetime"].eq(decision_time)
            & self.reference["station_id"].isin(station_ids)
        ].drop_duplicates("station_id", keep="last")
        result_by_station = {
            int(row.station_id): bool(row.y_same_30)
            for row in selected.itertuples(index=False)
        }
        results = [
            {"station_id": station_id, "still_empty": result_by_station.get(station_id)}
            for station_id in station_ids
        ]
        evaluated = [item for item in results if item["still_empty"] is not None]
        hits = sum(bool(item["still_empty"]) for item in evaluated)
        return {
            "decision_time": _iso_taipei(decision_time),
            "target_time": _iso_taipei(decision_time + pd.Timedelta(minutes=30)),
            "results": results,
            "summary": {
                "evaluated_actions": len(evaluated),
                "hits": hits,
                "precision": round(hits / len(evaluated), 6) if evaluated else None,
            },
            "scoring_note": "逐快照比較t+30是否仍為0車，不使用Episode命中。",
        }

    @staticmethod
    def _station_record(row: Any, is_scored: bool) -> dict[str, Any]:
        def get(name: str, default: Any = None) -> Any:
            if isinstance(row, pd.Series):
                return row.get(name, default)
            return getattr(row, name, default)

        station_id = int(get("station_id"))
        probability = _safe_number(get("p_lgbm_full")) if is_scored else None
        record = {
            "station_id": station_id,
            "station_name": _safe_text(get("station_name"), f"站點 {station_id}"),
            "district": _safe_text(get("district"), "未知行政區"),
            "latitude": _safe_number(get("latitude")),
            "longitude": _safe_number(get("longitude")),
            "current_bikes": int(get("available_bikes", 0)),
            "current_docks": int(get("available_docks", 0)),
            "capacity": int(get("capacity", 0)),
            "red_duration_steps": int(get("red_duration_steps")) if is_scored else None,
            "red_duration_display": _duration_text(get("red_duration_steps") if is_scored else None),
            "status": "scored" if is_scored else "insufficient_history",
            "risk_probability": probability,
            "risk_rank": int(get("risk_rank")) if is_scored else None,
            "is_alert": bool(get("is_alert", False)) if is_scored else False,
            "in_action_list": False,
            "route_order": None,
            "distance_from_previous_km": None,
        }
        return record


class DemoRequestHandler(BaseHTTPRequestHandler):
    service: DemoService
    server_version = "YouBikeDemo/1.0"

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self._cors_headers()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/health":
                self._write_json(
                    {
                        "status": "ok",
                        "model": "lgbm_full",
                        "rows": len(self.service.eligible),
                        "process_id": os.getpid(),
                    }
                )
                return
            if parsed.path == "/api/options":
                query = parse_qs(parsed.query)
                requested = query.get("decision_time", [None])[0]
                self._write_json(self.service.options(requested))
                return
            self._write_json({"error": "找不到此端點"}, HTTPStatus.NOT_FOUND)
        except (ValueError, TypeError) as error:
            self._write_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:  # pragma: no cover - last-resort local server guard
            self._write_json({"error": f"服務暫時無法處理：{error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
            if parsed.path == "/api/predict":
                self._write_json(self.service.predict(payload))
                return
            if parsed.path == "/api/reveal":
                self._write_json(self.service.reveal(payload))
                return
            self._write_json({"error": "找不到此端點"}, HTTPStatus.NOT_FOUND)
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self._write_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:  # pragma: no cover - last-resort local server guard
            self._write_json({"error": f"服務暫時無法處理：{error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[api] {self.address_string()} - {format % args}")

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0 or content_length > 64 * 1024:
            raise ValueError("請提供有效的JSON請求")
        payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON最外層必須是物件")
        return payload

    def _cors_headers(self) -> None:
        origin = self.headers.get("Origin")
        if origin in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("X-Content-Type-Options", "nosniff")

    def _write_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self._cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def create_server(host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    service = DemoService()
    handler = type("BoundDemoRequestHandler", (DemoRequestHandler,), {"service": service})
    return ThreadingHTTPServer((host, port), handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the local YouBike demo API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = create_server(args.host, args.port)
    print(f"YouBike demo API: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
