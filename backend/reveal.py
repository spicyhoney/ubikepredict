from __future__ import annotations

"""Truth-only replay service used by the isolated Reveal Lambda.

This module intentionally does not import ``backend.inference`` or LightGBM.
The prediction container therefore does not need permission to read truth, and
the reveal container does not initialise a model merely to score a replay.
"""

import os
from pathlib import Path
from typing import Any

import pandas as pd


TAIPEI_TIMEZONE = "Asia/Taipei"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_PATH = "data/reference/june_all_eligible_decisions.parquet"
FULL_DOCK_REFERENCE_PATH = "data/reference/june_full_dock_all_eligible_decisions.parquet"
REFERENCE_CONFIG = {
    "empty": ("UBIKE_REFERENCE_PATH", DEFAULT_REFERENCE_PATH),
    "full_dock": ("UBIKE_FULL_DOCK_REFERENCE_PATH", FULL_DOCK_REFERENCE_PATH),
}
OUTCOME_LABELS = {"empty": "仍為0車", "full_dock": "仍為0空位"}


def _configured_path(variable: str, default: str) -> Path:
    path = Path(os.environ.get(variable, default))
    return path if path.is_absolute() else REPO_ROOT / path


def _local_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("decision_time 格式錯誤")
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


class RevealService:
    """Read frozen outcomes only after an explicit reveal request."""

    def __init__(self, reference_path: Path | None = None) -> None:
        self.reference_paths = {
            mode: _configured_path(variable, default)
            for mode, (variable, default) in REFERENCE_CONFIG.items()
        }
        if reference_path is not None:
            self.reference_paths["empty"] = reference_path
        self.reference_path = self.reference_paths["empty"]
        self._references: dict[str, pd.DataFrame] = {}

    @property
    def reference(self) -> pd.DataFrame:
        return self.reference_for("empty")

    def reference_for(self, mode: str) -> pd.DataFrame:
        if mode not in REFERENCE_CONFIG:
            raise ValueError("mode 必須是 empty 或 full_dock")
        if mode not in self._references:
            frame = pd.read_parquet(
                self.reference_paths[mode],
                columns=["datetime", "target_datetime", "station_id", "y_same_30"],
            )
            frame["datetime"] = pd.to_datetime(frame["datetime"])
            frame["target_datetime"] = pd.to_datetime(frame["target_datetime"])
            self._references[mode] = frame
        return self._references[mode]

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "reveal",
            "rows": int(len(self.reference)),
            "rows_by_mode": {
                mode: int(len(self.reference_for(mode))) for mode in REFERENCE_CONFIG
            },
        }

    def reveal(self, payload: dict[str, Any]) -> dict[str, Any]:
        mode = str(payload.get("mode") or "empty").strip()
        if mode not in REFERENCE_CONFIG:
            raise ValueError("mode 必須是 empty 或 full_dock")
        requested_time = payload.get("decision_time")
        if not requested_time:
            raise ValueError("decision_time 不可為空")
        decision_time = _local_timestamp(requested_time)

        raw_station_ids = payload.get("station_ids")
        if not isinstance(raw_station_ids, list) or not raw_station_ids:
            raise ValueError("station_ids 不可為空")
        if len(raw_station_ids) > 25:
            raise ValueError("station_ids 一次最多25站")

        station_ids: list[int] = []
        try:
            for value in raw_station_ids:
                # A boolean is an ``int`` subclass in Python but is not a valid id.
                if isinstance(value, bool):
                    raise ValueError
                station_id = int(value)
                if isinstance(value, float) and not value.is_integer():
                    raise ValueError
                if station_id not in station_ids:
                    station_ids.append(station_id)
        except (TypeError, ValueError) as error:
            raise ValueError("station_ids 必須是整數陣列") from error
        if not station_ids:
            raise ValueError("station_ids 不可為空")

        reference = self.reference_for(mode)
        if not reference["datetime"].eq(decision_time).any():
            raise ValueError("decision_time 不在六月可評估的尖峰時點")
        selected = reference.loc[
            reference["datetime"].eq(decision_time)
            & reference["station_id"].isin(station_ids)
        ].drop_duplicates("station_id", keep="last")
        result_by_station = {
            int(row.station_id): bool(row.y_same_30)
            for row in selected.itertuples(index=False)
        }
        results = []
        for station_id in station_ids:
            outcome = result_by_station.get(station_id)
            item = {"station_id": station_id, "still_red": outcome, "still_imbalanced": outcome}
            item["still_empty" if mode == "empty" else "still_full_dock"] = outcome
            results.append(item)
        evaluated = [item for item in results if item["still_red"] is not None]
        hits = sum(bool(item["still_red"]) for item in evaluated)
        return {
            "mode": mode,
            "decision_time": _iso_taipei(decision_time),
            "target_time": _iso_taipei(decision_time + pd.Timedelta(minutes=30)),
            "results": results,
            "summary": {
                "evaluated_actions": len(evaluated),
                "hits": hits,
                "precision": round(hits / len(evaluated), 6) if evaluated else None,
            },
            "scoring_note": f"逐快照比較t+30是否{OUTCOME_LABELS[mode]}，不使用Episode命中。",
        }


__all__ = ["RevealService"]
