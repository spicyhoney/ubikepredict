from __future__ import annotations

"""Standalone inference for the frozen YouBike persistence LightGBM models.

Only the 68 feature names frozen before May are selected for the model matrix.
Outcome columns such as ``y_same_30`` and ``future_30_*`` are never passed to
LightGBM, even though the sealed June cache retains them for offline auditing.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
JUNE_START = pd.Timestamp("2026-06-01 00:00:00")
JULY_START = pd.Timestamp("2026-07-01 00:00:00")
DERIVED_FEATURES = {
    "red_duration_capped_6",
    "red_duration_log1p_capped",
    "duration_90plus",
}
AUDIT_ONLY_COLUMNS = {
    "y_same_30",
    "future_30_balanced",
    "future_30_opposite_red",
    "primary_prediction_correct",
}
SUPPORTED_MODES = ("empty", "full_dock")
MODE_ASSETS: dict[str, dict[str, str]] = {
    "empty": {
        "model_env": "UBIKE_MODEL_PATH",
        "model_default": "model/lgbm_full.txt",
        "freeze_env": "UBIKE_FREEZE_PATH",
        "freeze_default": "config/final_policy_freeze_before_may.json",
        "protocol_env": "UBIKE_PROTOCOL_PATH",
        "protocol_default": "config/protocol_frozen_before_june.json",
        "input_env": "UBIKE_INPUT_PATH",
        "input_default": "data/source/dynamic_red_empty_2026_06.parquet",
    },
    "full_dock": {
        "model_env": "UBIKE_FULL_DOCK_MODEL_PATH",
        "model_default": "model/lgbm_full_dock.txt",
        "freeze_env": "UBIKE_FULL_DOCK_FREEZE_PATH",
        "freeze_default": "config/final_policy_freeze_full_dock_before_may.json",
        "protocol_env": "UBIKE_FULL_DOCK_PROTOCOL_PATH",
        "protocol_default": "config/protocol_full_dock_frozen_before_june.json",
        "input_env": "UBIKE_FULL_DOCK_INPUT_PATH",
        "input_default": "data/source/dynamic_red_full_2026_06.parquet",
    },
}


def _repo_path(env_name: str, default: str) -> Path:
    value = Path(os.environ.get(env_name, default))
    return value if value.is_absolute() else REPO_ROOT / value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class FrozenYouBikeModel:
    """Load, validate, and run the immutable full LightGBM policy."""

    def __init__(
        self,
        model_path: Path | None = None,
        freeze_path: Path | None = None,
        protocol_path: Path | None = None,
        stations_path: Path | None = None,
        mode: str = "empty",
    ) -> None:
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"mode must be one of {SUPPORTED_MODES}")
        self.mode = mode
        assets = MODE_ASSETS[mode]
        self.model_path = model_path or _repo_path(assets["model_env"], assets["model_default"])
        self.freeze_path = freeze_path or _repo_path(
            assets["freeze_env"], assets["freeze_default"]
        )
        self.protocol_path = protocol_path or _repo_path(
            assets["protocol_env"], assets["protocol_default"]
        )
        self.stations_path = stations_path or _repo_path(
            "UBIKE_STATIONS_PATH", "data/stations/dim_station.csv"
        )

        self.freeze = _read_json(self.freeze_path)
        self.protocol = _read_json(self.protocol_path)
        self.features = list(self.freeze["features"]["lgbm_full"])
        if len(self.features) != 68 or len(set(self.features)) != 68:
            raise RuntimeError("Frozen lgbm_full feature list must contain 68 unique names")
        leaked = AUDIT_ONLY_COLUMNS.intersection(self.features)
        if leaked:
            raise RuntimeError(f"Outcome leakage in frozen features: {sorted(leaked)}")

        self.category_schema = self.protocol["category_schemas"]["lgbm_full"]
        expected_categories = {"station_id", "district_code", "target_hour", "target_weekday"}
        if set(self.category_schema) != expected_categories:
            raise RuntimeError("Frozen categorical schema is incomplete or unexpected")

        # model_str avoids a LightGBM Windows issue with non-ASCII parent paths.
        self.booster = lgb.Booster(model_str=self.model_path.read_text(encoding="utf-8"))
        if self.booster.feature_name() != self.features:
            raise RuntimeError("Model feature order does not match the frozen feature order")

        calibration = self.freeze["calibrators"]["lgbm_full"]
        self.platt_slope = float(calibration["slope"])
        self.platt_intercept = float(calibration["intercept"])
        thresholds = self.freeze["thresholds"]["lgbm_full"]
        self.balanced_threshold = float(thresholds["precision_first"]["threshold"])
        self.strict_threshold = float(thresholds["p70_recall"]["threshold"])
        constants = np.asarray(
            [
                self.platt_slope,
                self.platt_intercept,
                self.balanced_threshold,
                self.strict_threshold,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(constants).all():
            raise RuntimeError("Frozen calibration or threshold contains a non-finite value")

        station_columns = [
            "station_id",
            "station_name",
            "district",
            "longitude",
            "latitude",
            "capacity_min",
            "capacity_max",
        ]
        self.stations = pd.read_csv(self.stations_path, usecols=station_columns)
        self.stations = self.stations.drop_duplicates("station_id", keep="last")
        if self.stations["station_id"].duplicated().any():
            raise RuntimeError("Station lookup must be unique by station_id")

    def model_matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Build the exact frozen matrix; unknown categories become missing."""
        working = frame.copy()
        if "red_duration_steps" not in working:
            raise ValueError("Input is missing red_duration_steps required for duration features")
        working["red_duration_capped_6"] = (
            pd.to_numeric(working["red_duration_steps"], errors="coerce")
            .clip(upper=6)
            .astype("float32")
        )
        working["red_duration_log1p_capped"] = np.log1p(
            working["red_duration_capped_6"]
        ).astype("float32")
        working["duration_90plus"] = (
            pd.to_numeric(working["red_duration_steps"], errors="coerce").ge(3).astype("int8")
        )

        missing = [name for name in self.features if name not in working.columns]
        if missing:
            raise ValueError(f"Input is missing frozen features: {missing}")
        matrix = working.loc[:, self.features].copy()
        for column, categories in self.category_schema.items():
            numeric = pd.to_numeric(matrix[column], errors="coerce")
            numeric = numeric.where(numeric.isin(categories))
            matrix[column] = pd.Categorical(numeric, categories=categories)
        for column in set(self.features) - set(self.category_schema):
            matrix[column] = pd.to_numeric(matrix[column], errors="coerce").astype("float32")
        return matrix

    def predict_frame(self, frame: pd.DataFrame, attach_stations: bool = True) -> pd.DataFrame:
        """Return calibrated risk and both frozen policy decisions."""
        matrix = self.model_matrix(frame)
        raw_score = np.asarray(
            self.booster.predict(matrix, raw_score=True), dtype=np.float32
        )
        linear = np.clip(
            self.platt_slope * raw_score + self.platt_intercept,
            -35.0,
            35.0,
        )
        probability = (1.0 / (1.0 + np.exp(-linear))).astype(np.float32)

        display_candidates = [
            "datetime",
            "target_datetime",
            "station_id",
            "red_duration_steps",
            "available_bikes",
            "available_docks",
            "capacity",
        ]
        result = frame.loc[:, [c for c in display_candidates if c in frame.columns]].copy()
        result["raw_lgbm_full"] = raw_score
        result["p_lgbm_full"] = probability
        result["alert_secondary_balanced"] = (
            result["p_lgbm_full"].ge(self.balanced_threshold).astype("int8")
        )
        result["alert_primary_strict"] = (
            result["p_lgbm_full"].ge(self.strict_threshold).astype("int8")
        )
        if attach_stations and "station_id" in result:
            result = result.merge(
                self.stations,
                on="station_id",
                how="left",
                validate="many_to_one",
            )
        return result


def load_demo_source(path: Path | None = None, mode: str = "empty") -> pd.DataFrame:
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"mode must be one of {SUPPORTED_MODES}")
    assets = MODE_ASSETS[mode]
    source_path = path or _repo_path(assets["input_env"], assets["input_default"])
    frame = pd.read_parquet(source_path)
    for column in ("datetime", "target_datetime", "episode_start_datetime"):
        frame[column] = pd.to_datetime(frame[column])
    return frame


def select_eligible_june(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the population rule frozen before June was opened."""
    required = {
        "decision_is_peak",
        "duration_unknown",
        "episode_start_datetime",
        "datetime",
        "target_datetime",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Input is missing population columns: {sorted(missing)}")
    mask = (
        frame["decision_is_peak"].eq(1)
        & frame["duration_unknown"].eq(0)
        & frame["episode_start_datetime"].ge(JUNE_START)
        & frame["episode_start_datetime"].lt(JULY_START)
        & frame["datetime"].ge(JUNE_START)
        & frame["target_datetime"].lt(JULY_START)
    )
    return frame.loc[mask].copy()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen YouBike t+30 empty-station risk model."
    )
    parser.add_argument("--datetime", help="Exact decision time, e.g. 2026-06-23 19:30")
    parser.add_argument("--district", help="Exact district name, e.g. 板橋區")
    parser.add_argument(
        "--mode",
        choices=SUPPORTED_MODES,
        default="empty",
        help="Predict persistent empty stations or persistent full docks",
    )
    parser.add_argument("--top", type=int, default=10, help="Highest-risk rows to return")
    parser.add_argument(
        "--policy",
        choices=("balanced", "strict", "all"),
        default="balanced",
        help="Optionally keep only rows alerted by one frozen policy",
    )
    parser.add_argument("--format", choices=("table", "json", "csv"), default="table")
    parser.add_argument("--output", type=Path, help="Optional UTF-8 output file")
    parser.add_argument("--input", type=Path, help="Override the sealed June parquet")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.top <= 0:
        raise SystemExit("--top must be greater than zero")
    engine = FrozenYouBikeModel(mode=args.mode)
    source = select_eligible_june(load_demo_source(args.input, mode=args.mode))
    if args.datetime:
        requested_time = pd.Timestamp(args.datetime)
        source = source.loc[source["datetime"].eq(requested_time)].copy()
    predictions = engine.predict_frame(source, attach_stations=True)
    if args.district:
        predictions = predictions.loc[predictions["district"].eq(args.district)].copy()
    if args.policy == "balanced":
        predictions = predictions.loc[predictions["alert_secondary_balanced"].eq(1)].copy()
    elif args.policy == "strict":
        predictions = predictions.loc[predictions["alert_primary_strict"].eq(1)].copy()
    predictions = predictions.sort_values(
        ["p_lgbm_full", "station_id"], ascending=[False, True]
    ).head(args.top)
    if predictions.empty:
        raise SystemExit("No rows matched the requested time, district, and policy")

    ordered = [
        "datetime",
        "target_datetime",
        "station_id",
        "station_name",
        "district",
        "longitude",
        "latitude",
        "available_bikes",
        "available_docks",
        "capacity",
        "red_duration_steps",
        "p_lgbm_full",
        "alert_secondary_balanced",
        "alert_primary_strict",
    ]
    predictions = predictions.loc[:, [c for c in ordered if c in predictions.columns]]
    if args.format == "json":
        rendered = predictions.to_json(orient="records", date_format="iso", force_ascii=False)
    elif args.format == "csv":
        rendered = predictions.to_csv(index=False)
    else:
        rendered = predictions.to_string(index=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote {len(predictions):,} rows to {args.output}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
