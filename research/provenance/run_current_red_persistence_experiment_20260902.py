from __future__ import annotations

"""Predict whether a newly observed exact-zero YouBike incident persists.

Primary population: the first observed red snapshot of an episode.  A row is
an onset only when the immediately previous exact 30-minute snapshot is valid,
operational, same-capacity, and not the same incident type.  This prevents a
long incident from being counted as many easy independent predictions.

May is not opened until models, calibrators, methods, and thresholds have all
been frozen using January through April.  June is never opened.
"""

import argparse
import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from train_youbike_model_v1 import (
    CATEGORICAL_FEATURES,
    LIGHTGBM_FEATURES,
    MONTH_FILES,
    PEAK_HOURS,
    READ_COLUMNS,
    PlattCalibrator,
    _row_stat,
    _valid_lag,
    binary_counts,
    build_metadata_maps,
)


SEED = 42
ROOT = Path(__file__).resolve().parent
WORK_DIR = ROOT / "tmp" / "youbike_current_red_persistence_20260902"
OUTPUT_DIR = ROOT / "outputs" / "youbike_current_red_persistence_20260902"
CALIBRATION_END = pd.Timestamp("2026-04-15")
SELECTION_END = pd.Timestamp("2026-04-21")
APRIL_END = pd.Timestamp("2026-05-01")
RECALL_FLOOR = 0.20
HISTORY_SMOOTHING = 20.0
EVENTS = ("empty", "full")
TARGETS = ("same_30", "same_both")
FOLDS = (
    ("fold_1", (1,), 2),
    ("fold_2", (1, 2), 3),
)

PERSISTENCE_FEATURES = [
    "same_red_lag_30",
    "same_red_lag_60",
    "same_red_lag_90",
    "same_red_lag_1d",
    "same_red_lag_7d",
    "opposite_red_lag_30",
    "opposite_red_lag_60",
    "opposite_red_lag_90",
    "same_red_count_past_90",
]
MODEL_FEATURES = list(dict.fromkeys(LIGHTGBM_FEATURES + PERSISTENCE_FEATURES))
META_COLUMNS = [
    "datetime",
    "target_datetime",
    "t60_datetime",
    "station_id",
    "source_month",
    "decision_is_peak",
    "target_is_peak",
    "is_first_observed_red",
    "is_ongoing_red",
    "future_30_balanced",
    "future_60_balanced",
    "future_30_opposite_red",
    "future_60_opposite_red",
    "y_same_30",
    "y_same_60",
    "y_same_both",
]

# One predeclared, conservative table-model structure.  Tree count is selected
# only from Jan-Mar rolling folds via early stopping; April/May/June never choose it.
LGB_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.04,
    "n_estimators": 1000,
    "num_leaves": 31,
    "max_depth": 8,
    "min_child_samples": 100,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.80,
    "reg_alpha": 0.5,
    "reg_lambda": 5.0,
    "max_bin": 127,
    "deterministic": True,
    "force_col_wise": True,
    "random_state": SEED,
    "verbosity": -1,
}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    raise TypeError(type(value))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_layout() -> None:
    for directory in (
        WORK_DIR,
        OUTPUT_DIR / "config",
        OUTPUT_DIR / "data",
        OUTPUT_DIR / "metrics",
        OUTPUT_DIR / "models",
    ):
        directory.mkdir(parents=True, exist_ok=True)


def cache_path(month: int, event: str) -> Path:
    return WORK_DIR / f"current_red_{event}_2026_{month:02d}.parquet"


def valid_inventory(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    capacity = frame["capacity"].astype("float32")
    bikes = frame["available_bikes"].astype("float32")
    docks = frame["available_docks"].astype("float32")
    unavailable = frame["unavailable_count"].astype("float32")
    positive_capacity = capacity.gt(0)
    row_valid = (
        positive_capacity
        & bikes.notna()
        & docks.notna()
        & unavailable.notna()
        & bikes.ge(0)
        & docks.ge(0)
        & unavailable.ge(0)
        & bikes.le(capacity)
        & docks.le(capacity)
        & (bikes + docks).le(capacity)
    )
    return capacity, bikes, docks, unavailable, row_valid


def build_month(
    month: int,
    district_map: dict[str, int],
    segment_map: dict[int, int],
    force: bool,
    prefuture_output_dir: Path | None = None,
    dynamic_output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    destinations = {event: cache_path(month, event) for event in EVENTS}
    prefuture_destinations = (
        {
            event: prefuture_output_dir / f"prefuture_red_onset_{event}_2026_{month:02d}.parquet"
            for event in EVENTS
        }
        if prefuture_output_dir is not None
        else {}
    )
    dynamic_destinations = (
        {
            event: dynamic_output_dir / f"dynamic_red_{event}_2026_{month:02d}.parquet"
            for event in EVENTS
        }
        if dynamic_output_dir is not None
        else {}
    )
    quality_path = WORK_DIR / f"quality_2026_{month:02d}.json"
    requested_prefuture_exists = not prefuture_destinations or all(
        path.is_file() for path in prefuture_destinations.values()
    )
    requested_dynamic_exists = not dynamic_destinations or all(
        path.is_file() for path in dynamic_destinations.values()
    )
    if (
        all(path.is_file() for path in destinations.values())
        and quality_path.is_file()
        and requested_prefuture_exists
        and requested_dynamic_exists
        and not force
    ):
        log(f"Month {month:02d}: red-onset caches already exist")
        return json.loads(quality_path.read_text(encoding="utf-8"))

    source = MONTH_FILES[month]
    log(f"Month {month:02d}: reading raw snapshots and building past-only features")
    frame = pd.read_csv(
        source,
        usecols=READ_COLUMNS,
        parse_dates=["datetime"],
        dtype={
            "station_id": "int32",
            "station_name": "string",
            "district": "string",
            "capacity": "Int16",
            "available_bikes": "Int16",
            "available_docks": "Int16",
            "unavailable_count": "Int16",
            "longitude": "float32",
            "latitude": "float32",
            "empty_now": "int8",
            "full_now": "int8",
            "inactive_or_ambiguous_zero": "int8",
            "operational_observation": "int8",
        },
    )
    raw_rows = len(frame)
    frame.sort_values(["station_id", "datetime"], inplace=True, kind="mergesort")
    frame.reset_index(drop=True, inplace=True)
    capacity, bikes, docks, unavailable, row_valid = valid_inventory(frame)
    operational = row_valid & frame["operational_observation"].eq(1) & frame["inactive_or_ambiguous_zero"].eq(0)
    frame["row_valid_feature"] = row_valid.astype("int8")
    frame["valid_operational_feature"] = operational.astype("int8")
    frame["bike_ratio"] = (bikes / capacity).where(capacity.gt(0)).astype("float32")
    frame["dock_ratio"] = (docks / capacity).where(capacity.gt(0)).astype("float32")
    frame["unavailable_ratio"] = (unavailable / capacity).where(capacity.gt(0)).astype("float32")

    grouped = frame.groupby("station_id", sort=False, observed=True)
    lag_values: dict[int, tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]] = {}
    for steps in (1, 2, 3, 48, 336):
        lag_values[steps] = _valid_lag(grouped, frame, steps)
    for steps, minutes in ((1, 30), (2, 60), (3, 90)):
        _, lag_bikes, lag_docks, lag_bike_ratio, lag_dock_ratio = lag_values[steps]
        frame[f"bikes_lag_{minutes}"] = lag_bikes
        frame[f"docks_lag_{minutes}"] = lag_docks
        frame[f"bike_ratio_lag_{minutes}"] = lag_bike_ratio
        frame[f"dock_ratio_lag_{minutes}"] = lag_dock_ratio
        frame[f"delta_bikes_{minutes}"] = (bikes - lag_bikes).astype("float32")
        frame[f"delta_docks_{minutes}"] = (docks - lag_docks).astype("float32")
    frame["bike_ratio_lag_1d"] = lag_values[48][3]
    frame["dock_ratio_lag_1d"] = lag_values[48][4]
    frame["bike_ratio_lag_7d"] = lag_values[336][3]
    frame["dock_ratio_lag_7d"] = lag_values[336][4]
    frame["abs_delta_bikes_30"] = frame["delta_bikes_30"].abs().astype("float32")
    frame["delta_bikes_ratio_30"] = (frame["delta_bikes_30"] / capacity).astype("float32")
    frame["delta_docks_ratio_30"] = (frame["delta_docks_30"] / capacity).astype("float32")
    previous_delta_bikes = frame["bikes_lag_30"] - frame["bikes_lag_60"]
    previous_delta_docks = frame["docks_lag_30"] - frame["docks_lag_60"]
    frame["bike_acceleration_30"] = (frame["delta_bikes_30"] - previous_delta_bikes).astype("float32")
    frame["dock_acceleration_30"] = (frame["delta_docks_30"] - previous_delta_docks).astype("float32")
    bike_window = [frame["bike_ratio"], frame["bike_ratio_lag_30"], frame["bike_ratio_lag_60"], frame["bike_ratio_lag_90"]]
    dock_window = [frame["dock_ratio"], frame["dock_ratio_lag_30"], frame["dock_ratio_lag_60"], frame["dock_ratio_lag_90"]]
    frame["bike_ratio_mean_90"] = _row_stat(bike_window, "mean")
    frame["bike_ratio_min_90"] = _row_stat(bike_window, "min")
    frame["bike_ratio_max_90"] = _row_stat(bike_window, "max")
    frame["dock_ratio_mean_90"] = _row_stat(dock_window, "mean")
    frame["dock_ratio_min_90"] = _row_stat(dock_window, "min")
    frame["dock_ratio_max_90"] = _row_stat(dock_window, "max")

    target_datetime = frame["datetime"] + pd.Timedelta(minutes=30)
    target_hour = target_datetime.dt.hour.astype("int8")
    weekday_zero = target_datetime.dt.weekday.astype("int8")
    frame["target_datetime"] = target_datetime
    frame["t60_datetime"] = frame["datetime"] + pd.Timedelta(minutes=60)
    frame["target_hour"] = target_hour
    frame["target_weekday"] = (weekday_zero + 1).astype("int8")
    frame["target_hour_sin"] = np.sin(2 * np.pi * target_hour / 24).astype("float32")
    frame["target_hour_cos"] = np.cos(2 * np.pi * target_hour / 24).astype("float32")
    frame["target_weekday_sin"] = np.sin(2 * np.pi * weekday_zero / 7).astype("float32")
    frame["target_weekday_cos"] = np.cos(2 * np.pi * weekday_zero / 7).astype("float32")
    frame["target_is_weekend"] = weekday_zero.ge(5).astype("int8")
    frame["target_is_peak"] = target_hour.isin(PEAK_HOURS).astype("int8")
    frame["decision_is_peak"] = frame["datetime"].dt.hour.isin(PEAK_HOURS).astype("int8")
    frame["lag_90_missing"] = frame["bike_ratio_lag_90"].isna().astype("int8")
    frame["lag_1d_missing"] = frame["bike_ratio_lag_1d"].isna().astype("int8")
    frame["lag_7d_missing"] = frame["bike_ratio_lag_7d"].isna().astype("int8")
    jump_threshold = np.maximum(5.0, 0.25 * capacity)
    frame["suspected_large_jump_past"] = (
        frame["abs_delta_bikes_30"].notna() & frame["abs_delta_bikes_30"].ge(jump_threshold)
    ).astype("int8")
    frame["district_code"] = frame["district"].map(district_map).fillna(-1).astype("int16")
    frame["segment_code"] = frame["station_id"].map(segment_map).fillna(0).astype("int8")
    frame["source_month"] = np.int8(month)

    exact_current = {
        "empty": operational & bikes.eq(0) & docks.gt(0),
        "full": operational & docks.eq(0) & bikes.gt(0),
    }
    event_flags = {event: exact_current[event].astype("int8") for event in EVENTS}
    lag_flag_cache: dict[str, dict[int, pd.Series]] = {event: {} for event in EVENTS}
    for event in EVENTS:
        grouped_flag = event_flags[event].groupby(frame["station_id"], sort=False)
        for steps in (1, 2, 3, 48, 336):
            lag_flag_cache[event][steps] = grouped_flag.shift(steps).where(lag_values[steps][0]).astype("float32")

    future_lookup = frame[
        [
            "station_id",
            "datetime",
            "capacity",
            "available_bikes",
            "available_docks",
            "operational_observation",
            "inactive_or_ambiguous_zero",
            "row_valid_feature",
        ]
    ].copy()
    quality_rows: list[dict[str, Any]] = []
    base_columns = list(dict.fromkeys(["datetime", "target_datetime", "t60_datetime", "station_id", "source_month", "decision_is_peak", "target_is_peak"] + MODEL_FEATURES))

    for event in EVENTS:
        opposite = "full" if event == "empty" else "empty"
        for steps, suffix in ((1, "30"), (2, "60"), (3, "90"), (48, "1d"), (336, "7d")):
            frame[f"same_red_lag_{suffix}"] = lag_flag_cache[event][steps]
        for steps, suffix in ((1, "30"), (2, "60"), (3, "90")):
            frame[f"opposite_red_lag_{suffix}"] = lag_flag_cache[opposite][steps]
        frame["same_red_count_past_90"] = frame[["same_red_lag_30", "same_red_lag_60", "same_red_lag_90"]].sum(axis=1, min_count=1).astype("float32")

        current_mask = exact_current[event]
        current = frame.loc[current_mask, base_columns].copy()
        current["is_first_observed_red"] = current["same_red_lag_30"].eq(0).astype("int8")
        current["is_ongoing_red"] = current["same_red_lag_30"].eq(1).astype("int8")
        new_episode = current["same_red_lag_30"].ne(1)
        current["dynamic_episode_seq"] = (
            new_episode.groupby(current["station_id"], sort=False).cumsum().astype("int32")
        )
        episode_groups = current.groupby(
            ["station_id", "dynamic_episode_seq"], sort=False, observed=True
        )
        current["red_duration_steps"] = episode_groups.cumcount().astype("int16")
        current["red_duration_log1p"] = np.log1p(current["red_duration_steps"]).astype("float32")
        current["episode_start_datetime"] = episode_groups["datetime"].transform("min")
        # ``GroupBy.first`` skips missing values, so it cannot be used here:
        # a red episode already active at the first snapshot of a month has a
        # missing lag on its first row, followed by lag=1 rows.  ``first``
        # would incorrectly return 1 and mark that cross-boundary episode as
        # having a known duration.  Carry the first-row missing flag across
        # the episode explicitly instead.
        unknown_at_episode_start = (
            current["red_duration_steps"].eq(0)
            & current["same_red_lag_30"].isna()
        )
        current["duration_unknown"] = unknown_at_episode_start.groupby(
            [current["station_id"], current["dynamic_episode_seq"]],
            sort=False,
        ).transform("max").astype("int8")
        current_rows_before_future = len(current)

        # Optional deployment-valid onset cache for duration/survival work.
        # It is intentionally written *before* any t+30/t+60 availability
        # filter, so future observability cannot decide whether an episode is
        # admitted to the cohort.  The default pipeline is unchanged when the
        # optional directory is omitted.
        if prefuture_destinations:
            prefuture_output_dir.mkdir(parents=True, exist_ok=True)
            prefuture_columns = list(
                dict.fromkeys(
                    base_columns
                    + ["is_first_observed_red", "is_ongoing_red"]
                )
            )
            current.loc[current["is_first_observed_red"].eq(1), prefuture_columns].to_parquet(
                prefuture_destinations[event],
                engine="pyarrow",
                compression="zstd",
                index=False,
            )

        lookup30 = future_lookup.rename(
            columns={
                "datetime": "target_datetime",
                "capacity": "f30_capacity",
                "available_bikes": "f30_bikes",
                "available_docks": "f30_docks",
                "operational_observation": "f30_operational",
                "inactive_or_ambiguous_zero": "f30_inactive",
                "row_valid_feature": "f30_row_valid",
            }
        )
        lookup60 = future_lookup.rename(
            columns={
                "datetime": "t60_datetime",
                "capacity": "f60_capacity",
                "available_bikes": "f60_bikes",
                "available_docks": "f60_docks",
                "operational_observation": "f60_operational",
                "inactive_or_ambiguous_zero": "f60_inactive",
                "row_valid_feature": "f60_row_valid",
            }
        )
        current = current.merge(lookup30, on=["station_id", "target_datetime"], how="left", validate="one_to_one", sort=False)
        current = current.merge(lookup60, on=["station_id", "t60_datetime"], how="left", validate="one_to_one", sort=False)
        valid30 = (
            current["f30_capacity"].notna()
            & current["capacity"].astype("float32").eq(current["f30_capacity"].astype("float32"))
            & current["f30_operational"].eq(1)
            & current["f30_inactive"].eq(0)
            & current["f30_row_valid"].eq(1)
        )
        valid60 = (
            current["f60_capacity"].notna()
            & current["capacity"].astype("float32").eq(current["f60_capacity"].astype("float32"))
            & current["f60_operational"].eq(1)
            & current["f60_inactive"].eq(0)
            & current["f60_row_valid"].eq(1)
        )
        valid = valid30 & valid60

        if dynamic_destinations:
            dynamic_output_dir.mkdir(parents=True, exist_ok=True)
            dynamic_current = current.loc[valid30].copy()
            if event == "empty":
                dynamic_same30 = dynamic_current["f30_bikes"].eq(0) & dynamic_current["f30_docks"].gt(0)
                dynamic_opposite30 = dynamic_current["f30_docks"].eq(0) & dynamic_current["f30_bikes"].gt(0)
            else:
                dynamic_same30 = dynamic_current["f30_docks"].eq(0) & dynamic_current["f30_bikes"].gt(0)
                dynamic_opposite30 = dynamic_current["f30_bikes"].eq(0) & dynamic_current["f30_docks"].gt(0)
            dynamic_current["y_same_30"] = dynamic_same30.astype("int8")
            dynamic_current["future_30_balanced"] = (
                dynamic_current["f30_bikes"].gt(0) & dynamic_current["f30_docks"].gt(0)
            ).astype("int8")
            dynamic_current["future_30_opposite_red"] = dynamic_opposite30.astype("int8")
            dynamic_columns = list(
                dict.fromkeys(
                    [
                        "datetime",
                        "target_datetime",
                        "station_id",
                        "source_month",
                        "decision_is_peak",
                        "target_is_peak",
                        "is_first_observed_red",
                        "is_ongoing_red",
                        "dynamic_episode_seq",
                        "episode_start_datetime",
                        "red_duration_steps",
                        "red_duration_log1p",
                        "duration_unknown",
                        "future_30_balanced",
                        "future_30_opposite_red",
                        "y_same_30",
                    ]
                    + MODEL_FEATURES
                )
            )
            dynamic_current[dynamic_columns].to_parquet(
                dynamic_destinations[event],
                engine="pyarrow",
                compression="zstd",
                index=False,
            )
            dynamic_valid30_rows = len(dynamic_current)
            dynamic_valid30_positives = int(dynamic_current["y_same_30"].sum())
            del dynamic_current
        else:
            dynamic_valid30_rows = None
            dynamic_valid30_positives = None

        current = current.loc[valid].copy()
        if event == "empty":
            same30 = current["f30_bikes"].eq(0) & current["f30_docks"].gt(0)
            same60 = current["f60_bikes"].eq(0) & current["f60_docks"].gt(0)
            opposite30 = current["f30_docks"].eq(0) & current["f30_bikes"].gt(0)
            opposite60 = current["f60_docks"].eq(0) & current["f60_bikes"].gt(0)
        else:
            same30 = current["f30_docks"].eq(0) & current["f30_bikes"].gt(0)
            same60 = current["f60_docks"].eq(0) & current["f60_bikes"].gt(0)
            opposite30 = current["f30_bikes"].eq(0) & current["f30_docks"].gt(0)
            opposite60 = current["f60_bikes"].eq(0) & current["f60_docks"].gt(0)
        current["y_same_30"] = same30.astype("int8")
        current["y_same_60"] = same60.astype("int8")
        current["y_same_both"] = (same30 & same60).astype("int8")
        current["future_30_balanced"] = (current["f30_bikes"].gt(0) & current["f30_docks"].gt(0)).astype("int8")
        current["future_60_balanced"] = (current["f60_bikes"].gt(0) & current["f60_docks"].gt(0)).astype("int8")
        current["future_30_opposite_red"] = opposite30.astype("int8")
        current["future_60_opposite_red"] = opposite60.astype("int8")
        output_columns = list(dict.fromkeys(META_COLUMNS + MODEL_FEATURES))
        current[output_columns].to_parquet(destinations[event], engine="pyarrow", compression="zstd", index=False)

        onset = current["is_first_observed_red"].eq(1)
        ongoing = current["is_ongoing_red"].eq(1)
        peak = current["decision_is_peak"].eq(1)
        quality_rows.append(
            {
                "month": month,
                "event": event,
                "raw_rows": raw_rows,
                "current_red_rows_before_future_validation": current_rows_before_future,
                "valid_t30_t60_current_red_rows": len(current),
                "dynamic_valid_t30_current_red_rows": dynamic_valid30_rows,
                "dynamic_valid_t30_same30_positives": dynamic_valid30_positives,
                "first_observed_red_rows": int(onset.sum()),
                "ongoing_red_rows": int(ongoing.sum()),
                "peak_first_observed_red_rows": int((onset & peak).sum()),
                "peak_first_observed_same30_positives": int(current.loc[onset & peak, "y_same_30"].sum()),
                "peak_first_observed_sameboth_positives": int(current.loc[onset & peak, "y_same_both"].sum()),
                "opposite_red_at_30_rows": int(current.loc[onset, "future_30_opposite_red"].sum()),
                "opposite_red_at_60_rows": int(current.loc[onset, "future_60_opposite_red"].sum()),
                "cache_file": str(destinations[event]),
            }
        )
        del current, lookup30, lookup60
        gc.collect()

    write_json(quality_path, quality_rows)
    del frame, future_lookup, lag_values, lag_flag_cache, event_flags
    gc.collect()
    return quality_rows


def load_event_month(event: str, month: int, onset_only: bool = True) -> pd.DataFrame:
    frame = pd.read_parquet(cache_path(month, event), engine="pyarrow")
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame["target_datetime"] = pd.to_datetime(frame["target_datetime"])
    frame["t60_datetime"] = pd.to_datetime(frame["t60_datetime"])
    if onset_only:
        frame = frame.loc[frame["is_first_observed_red"].eq(1)].copy()
    return frame.reset_index(drop=True)


def category_schema(frames: list[pd.DataFrame]) -> dict[str, list[int]]:
    schema: dict[str, list[int]] = {}
    for column in CATEGORICAL_FEATURES:
        values = pd.concat([frame[column] for frame in frames], ignore_index=True)
        schema[column] = sorted(int(value) for value in values.dropna().unique())
    return schema


def model_matrix(frame: pd.DataFrame, schema: dict[str, list[int]]) -> pd.DataFrame:
    matrix = frame[MODEL_FEATURES].copy()
    for column in CATEGORICAL_FEATURES:
        numeric = pd.to_numeric(matrix[column], errors="coerce")
        numeric = numeric.where(numeric.isin(schema[column]))
        matrix[column] = pd.Categorical(numeric, categories=schema[column])
    for column in set(MODEL_FEATURES) - set(CATEGORICAL_FEATURES):
        matrix[column] = pd.to_numeric(matrix[column], errors="coerce").astype("float32")
    return matrix


def safe_ap(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=float)
    finite = np.isfinite(score)
    if finite.sum() == 0 or np.unique(y[finite]).size < 2:
        return math.nan
    return float(average_precision_score(y[finite], score[finite]))


def fit_history_table(train: pd.DataFrame, target: str) -> dict[str, Any]:
    global_rate = float(train[target].mean())
    grouped = train.groupby(["station_id", "target_hour"], observed=True)[target].agg(["sum", "count"]).reset_index()
    grouped["probability"] = (grouped["sum"] + HISTORY_SMOOTHING * global_rate) / (grouped["count"] + HISTORY_SMOOTHING)
    return {
        "global_rate": global_rate,
        "table": grouped[["station_id", "target_hour", "probability"]],
    }


def history_score(frame: pd.DataFrame, history: dict[str, Any]) -> np.ndarray:
    keyed = frame[["station_id", "target_hour"]].copy()
    keyed["_row"] = np.arange(len(keyed), dtype=np.int64)
    merged = keyed.merge(history["table"], on=["station_id", "target_hour"], how="left", sort=False)
    merged.sort_values("_row", inplace=True)
    return merged["probability"].fillna(history["global_rate"]).to_numpy(dtype=np.float32)


def model_parameters(n_jobs: int, n_estimators: int | None = None) -> dict[str, Any]:
    params = dict(LGB_PARAMS)
    params["n_jobs"] = n_jobs
    if n_estimators is not None:
        params["n_estimators"] = int(n_estimators)
    return params


def rolling_cv(n_jobs: int) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    rows: list[dict[str, Any]] = []
    selected_trees: dict[str, dict[str, list[int]]] = {event: {target: [] for target in TARGETS} for event in EVENTS}
    for event in EVENTS:
        monthly = {month: load_event_month(event, month) for month in (1, 2, 3)}
        for fold_id, train_months, validation_month in FOLDS:
            train = pd.concat([monthly[month] for month in train_months], ignore_index=True)
            validation = monthly[validation_month].loc[monthly[validation_month]["decision_is_peak"].eq(1)].reset_index(drop=True)
            schema = category_schema([train])
            train_x = model_matrix(train, schema)
            validation_x = model_matrix(validation, schema)
            for target in TARGETS:
                label = f"y_{target}"
                y_train = train[label].to_numpy(dtype=np.int8)
                y_val = validation[label].to_numpy(dtype=np.int8)
                model = lgb.LGBMClassifier(**model_parameters(n_jobs))
                model.fit(
                    train_x,
                    y_train,
                    categorical_feature=CATEGORICAL_FEATURES,
                    eval_set=[(validation_x, y_val)],
                    eval_metric="average_precision",
                    callbacks=[lgb.early_stopping(80, verbose=False)],
                )
                raw = model.predict_proba(validation_x, num_iteration=model.best_iteration_)[:, 1]
                best_iteration = int(model.best_iteration_ or LGB_PARAMS["n_estimators"])
                selected_trees[event][target].append(best_iteration)
                history = fit_history_table(train, label)
                history_prob = history_score(validation, history)
                prevalence = float(y_val.mean())
                rows.extend(
                    [
                        {
                            "event": event,
                            "target": target,
                            "fold": fold_id,
                            "train_months": ",".join(map(str, train_months)),
                            "validation_month": validation_month,
                            "method": "lightgbm",
                            "rows": len(validation),
                            "positives": int(y_val.sum()),
                            "prevalence": prevalence,
                            "average_precision": safe_ap(y_val, raw),
                            "best_iteration": best_iteration,
                        },
                        {
                            "event": event,
                            "target": target,
                            "fold": fold_id,
                            "train_months": ",".join(map(str, train_months)),
                            "validation_month": validation_month,
                            "method": "station_hour_history",
                            "rows": len(validation),
                            "positives": int(y_val.sum()),
                            "prevalence": prevalence,
                            "average_precision": safe_ap(y_val, history_prob),
                            "best_iteration": math.nan,
                        },
                    ]
                )
                log(f"CV {event}/{target}/{fold_id}: LGB AP={rows[-2]['average_precision']:.3f}, history AP={rows[-1]['average_precision']:.3f}")
                del model, raw, history, history_prob
                gc.collect()
            del train, validation, train_x, validation_x, schema
            gc.collect()
        del monthly
        gc.collect()
    trees = {
        event: {target: max(10, int(round(float(np.median(values))))) for target, values in targets.items()}
        for event, targets in selected_trees.items()
    }
    cv = pd.DataFrame(rows)
    cv.to_csv(OUTPUT_DIR / "metrics" / "rolling_cv.csv", index=False)
    write_json(OUTPUT_DIR / "config" / "cv_selected_tree_counts.json", trees)
    return cv, trees


def train_final_models(
    trees: dict[str, dict[str, int]],
    n_jobs: int,
) -> tuple[dict[str, dict[str, lgb.Booster]], dict[str, dict[str, list[int]]], dict[str, dict[str, dict[str, Any]]]]:
    boosters: dict[str, dict[str, lgb.Booster]] = {event: {} for event in EVENTS}
    schemas: dict[str, dict[str, list[int]]] = {}
    histories: dict[str, dict[str, dict[str, Any]]] = {event: {} for event in EVENTS}
    model_records: list[dict[str, Any]] = []
    for event in EVENTS:
        train = pd.concat([load_event_month(event, month) for month in (1, 2, 3)], ignore_index=True)
        schema = category_schema([train])
        schemas[event] = schema
        train_x = model_matrix(train, schema)
        for target in TARGETS:
            label = f"y_{target}"
            model = lgb.LGBMClassifier(**model_parameters(n_jobs, trees[event][target]))
            started = time.perf_counter()
            model.fit(train_x, train[label].to_numpy(dtype=np.int8), categorical_feature=CATEGORICAL_FEATURES)
            elapsed = time.perf_counter() - started
            path = OUTPUT_DIR / "models" / f"lightgbm_{event}_{target}.txt"
            path.write_text(model.booster_.model_to_string(), encoding="utf-8")
            boosters[event][target] = lgb.Booster(model_str=path.read_text(encoding="utf-8"))
            histories[event][target] = fit_history_table(train, label)
            histories[event][target]["table"].to_csv(
                OUTPUT_DIR / "models" / f"history_{event}_{target}.csv", index=False
            )
            model_records.append(
                {
                    "event": event,
                    "target": target,
                    "training_months": "1,2,3",
                    "training_rows": len(train),
                    "training_positives": int(train[label].sum()),
                    "n_estimators": trees[event][target],
                    "fit_seconds": elapsed,
                    "model_sha256": sha256(path),
                }
            )
            del model
            gc.collect()
        del train, train_x
        gc.collect()
    write_json(OUTPUT_DIR / "config" / "final_model_records.json", model_records)
    write_json(OUTPUT_DIR / "config" / "category_schemas.json", schemas)
    return boosters, schemas, histories


def score_month(
    month: int,
    boosters: dict[str, dict[str, lgb.Booster]],
    schemas: dict[str, dict[str, list[int]]],
    histories: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, pd.DataFrame]:
    scored: dict[str, pd.DataFrame] = {}
    for event in EVENTS:
        frame = load_event_month(event, month)
        frame = frame.loc[frame["decision_is_peak"].eq(1)].reset_index(drop=True)
        matrix = model_matrix(frame, schemas[event])
        for target in TARGETS:
            frame[f"raw_lgb_{target}"] = boosters[event][target].predict(matrix, raw_score=True).astype(np.float32)
            frame[f"p_history_{target}"] = history_score(frame, histories[event][target])
        scored[event] = frame
        del matrix
        gc.collect()
    return scored


def choose_threshold(y: np.ndarray, score: np.ndarray) -> tuple[float, dict[str, Any]]:
    candidates = np.unique(np.quantile(score[np.isfinite(score)], np.linspace(0.0, 1.0, 501)))
    viable: list[tuple[float, float, int, float, dict[str, Any]]] = []
    fallback: list[tuple[float, float, int, float, dict[str, Any]]] = []
    for threshold in candidates:
        counts = binary_counts(y, score >= threshold)
        row = (float(counts["precision"]), float(counts["recall"]), -int(counts["alerts"]), float(threshold), counts)
        fallback.append(row)
        if counts["recall"] >= RECALL_FLOOR and counts["tp"] >= 20:
            viable.append(row)
    selected = max(viable or fallback, key=lambda item: (item[0], item[1], item[2], item[3]))
    return selected[3], selected[4]


def wilson_interval(tp: int, alerts: int) -> tuple[float, float]:
    if alerts == 0:
        return math.nan, math.nan
    z = 1.96
    p = tp / alerts
    denominator = 1 + z * z / alerts
    centre = (p + z * z / (2 * alerts)) / denominator
    margin = z * math.sqrt(p * (1 - p) / alerts + z * z / (4 * alerts * alerts)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def fit_calibrators_and_freeze(
    april: dict[str, pd.DataFrame],
    cv: pd.DataFrame,
    trees: dict[str, dict[str, int]],
) -> tuple[dict[str, PlattCalibrator], dict[str, Any], pd.DataFrame]:
    calibrators: dict[str, PlattCalibrator] = {}
    rows: list[dict[str, Any]] = []
    event_freeze: dict[str, Any] = {}
    for event in EVENTS:
        frame = april[event]
        calibration = frame["target_datetime"].lt(CALIBRATION_END)
        selection = frame["target_datetime"].ge(CALIBRATION_END) & frame["target_datetime"].lt(SELECTION_END)
        event_freeze[event] = {}
        for target in TARGETS:
            label = f"y_{target}"
            key = f"{event}_{target}"
            calibrator = PlattCalibrator.fit(
                frame.loc[calibration, f"raw_lgb_{target}"].to_numpy(dtype=np.float32),
                frame.loc[calibration, label].to_numpy(dtype=np.int8),
            )
            calibrators[key] = calibrator
            frame[f"p_lgb_{target}"] = calibrator.predict(frame[f"raw_lgb_{target}"].to_numpy(dtype=np.float32))
            y = frame.loc[selection, label].to_numpy(dtype=np.int8)
            candidates = {
                "lightgbm": frame.loc[selection, f"p_lgb_{target}"].to_numpy(dtype=np.float32),
                "station_hour_history": frame.loc[selection, f"p_history_{target}"].to_numpy(dtype=np.float32),
            }
            methods: dict[str, Any] = {}
            for method, score in candidates.items():
                threshold, counts = choose_threshold(y, score)
                ap = safe_ap(y, score)
                methods[method] = {"threshold": threshold, "average_precision": ap, **counts}
                rows.append(
                    {
                        "split": "april_15_20_selection",
                        "event": event,
                        "target": target,
                        "method": method,
                        "rows": len(y),
                        "positives": int(y.sum()),
                        "prevalence": float(y.mean()),
                        "average_precision": ap,
                        "threshold": threshold,
                        **counts,
                    }
                )
            champion = max(
                methods,
                key=lambda method: (
                    methods[method]["precision"],
                    methods[method]["average_precision"],
                    methods[method]["recall"],
                    method,
                ),
            )
            event_freeze[event][target] = {
                "champion": champion,
                "threshold": methods[champion]["threshold"],
                "candidate_selection": methods,
            }
    selection_frame = pd.DataFrame(rows)
    selection_frame.to_csv(OUTPUT_DIR / "metrics" / "april_15_20_selection.csv", index=False)
    freeze = {
        "created_at": pd.Timestamp.now(),
        "primary_population": "first observed exact-zero incident only; previous exact t-30 snapshot must be valid and not the same incident",
        "events": "empty and full are separate",
        "targets": {
            "same_30": "same exact-zero incident at exact t+30",
            "same_both": "same exact-zero incident at both exact t+30 and exact t+60; no claim about every minute between snapshots",
        },
        "opposite_red_rule": "an opposite imbalance is negative for same-event persistence but is not described as recovery",
        "training_months": [1, 2, 3],
        "rolling_cv": "Jan->Feb and Jan-Feb->Mar; tree counts frozen from median early-stopping iterations",
        "april_calibration": "April 1-14 decision-time peak rows",
        "april_method_threshold_selection": "April 15-20 decision-time peak rows",
        "april_locked_gate": "April 21-30 decision-time peak rows",
        "may_rule": "not opened before this freeze; used once for final out-of-time evaluation only",
        "june_rule": "completely unopened by this experiment",
        "threshold_objective": "maximize precision subject to recall >= 20% and TP >= 20; no Top-K or Episode any-hit",
        "history_baseline": f"Jan-Mar station x target-hour rate with alpha={HISTORY_SMOOTHING} smoothing",
        "calibrators": {name: calibrator.as_dict() for name, calibrator in calibrators.items()},
        "tree_counts": trees,
        "event_targets": event_freeze,
        "cv_summary": cv.groupby(["event", "target", "method"])["average_precision"].mean().reset_index().to_dict("records"),
        "code_sha256": sha256(Path(__file__).resolve()),
    }
    write_json(OUTPUT_DIR / "config" / "pre_april_gate_and_may_freeze.json", freeze)
    log("Methods and thresholds frozen before the April gate and before opening May; June stays sealed")
    return calibrators, freeze, selection_frame


def apply_calibrators(scored: dict[str, pd.DataFrame], calibrators: dict[str, PlattCalibrator]) -> None:
    for event in EVENTS:
        for target in TARGETS:
            scored[event][f"p_lgb_{target}"] = calibrators[f"{event}_{target}"].predict(
                scored[event][f"raw_lgb_{target}"].to_numpy(dtype=np.float32)
            )


def method_score(frame: pd.DataFrame, target: str, method: str) -> np.ndarray:
    column = f"p_lgb_{target}" if method == "lightgbm" else f"p_history_{target}"
    return frame[column].to_numpy(dtype=np.float32)


def metric_row(
    split: str,
    event: str,
    target: str,
    method: str,
    frame: pd.DataFrame,
    threshold: float | None,
) -> dict[str, Any]:
    y = frame[f"y_{target}"].to_numpy(dtype=np.int8)
    if method == "all_persist_baseline":
        score = np.ones(len(frame), dtype=np.float32)
        alerts = np.ones(len(frame), dtype=bool)
        threshold = 1.0
    else:
        score = method_score(frame, target, method)
        alerts = score >= float(threshold)
    counts = binary_counts(y, alerts)
    low, high = wilson_interval(int(counts["tp"]), int(counts["alerts"]))
    prevalence = float(y.mean())
    days = max(1, frame["target_datetime"].dt.normalize().nunique())
    return {
        "split": split,
        "event": event,
        "target": target,
        "method": method,
        "rows": len(frame),
        "positives": int(y.sum()),
        "prevalence": prevalence,
        "average_precision": prevalence if method == "all_persist_baseline" else safe_ap(y, score),
        "threshold": threshold,
        **counts,
        "precision_ci95_low": low,
        "precision_ci95_high": high,
        "alerts_per_day": counts["alerts"] / days,
        "precision_lift_over_all_persist": counts["precision"] / prevalence if prevalence else math.nan,
        "passes_70_precision_and_20_recall": bool(counts["precision"] >= 0.70 and counts["recall"] >= 0.20),
    }


def evaluate(
    scored: dict[str, pd.DataFrame],
    split: str,
    freeze: dict[str, Any],
    date_mask: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in EVENTS:
        frame = scored[event]
        if date_mask is not None:
            start, stop = date_mask
            frame = frame.loc[frame["target_datetime"].ge(start) & frame["target_datetime"].lt(stop)].reset_index(drop=True)
        for target in TARGETS:
            for method in ("all_persist_baseline", "station_hour_history", "lightgbm"):
                threshold = None if method == "all_persist_baseline" else freeze["event_targets"][event][target]["candidate_selection"][method]["threshold"]
                rows.append(metric_row(split, event, target, method, frame, threshold))
    return pd.DataFrame(rows)


def cohort_descriptives(months: list[int], split: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in EVENTS:
        frame = pd.concat([load_event_month(event, month, onset_only=False) for month in months], ignore_index=True)
        frame = frame.loc[frame["decision_is_peak"].eq(1)].copy()
        for cohort, mask in {
            "first_observed_red_primary": frame["is_first_observed_red"].eq(1),
            "ongoing_red_secondary": frame["is_ongoing_red"].eq(1),
            "all_valid_red_snapshots": np.ones(len(frame), dtype=bool),
        }.items():
            subset = frame.loc[mask]
            for target in TARGETS:
                rows.append(
                    {
                        "split": split,
                        "event": event,
                        "cohort": cohort,
                        "target": target,
                        "rows": len(subset),
                        "positives": int(subset[f"y_{target}"].sum()),
                        "persistence_rate": float(subset[f"y_{target}"].mean()) if len(subset) else math.nan,
                        "opposite_red_at_30": int(subset["future_30_opposite_red"].sum()),
                        "opposite_red_at_60": int(subset["future_60_opposite_red"].sum()),
                    }
                )
        del frame
    return pd.DataFrame(rows)


def prediction_extract(scored: dict[str, pd.DataFrame], split: str) -> None:
    pieces = []
    for event in EVENTS:
        frame = scored[event].copy()
        frame["event"] = event
        pieces.append(
            frame[
                [
                    "event",
                    "datetime",
                    "target_datetime",
                    "t60_datetime",
                    "station_id",
                    "y_same_30",
                    "y_same_60",
                    "y_same_both",
                    "future_30_balanced",
                    "future_60_balanced",
                    "future_30_opposite_red",
                    "future_60_opposite_red",
                    "p_lgb_same_30",
                    "p_history_same_30",
                    "p_lgb_same_both",
                    "p_history_same_both",
                ]
            ]
        )
    pd.concat(pieces, ignore_index=True).to_parquet(
        OUTPUT_DIR / "data" / f"{split}_peak_first_red_predictions.parquet",
        engine="pyarrow",
        compression="zstd",
        index=False,
    )


def summary_json(
    freeze: dict[str, Any],
    april_gate: pd.DataFrame,
    may: pd.DataFrame,
    cohort: pd.DataFrame,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "primary_population": "first observed exact-zero incident, one prediction per episode onset",
        "may_training_or_tuning": False,
        "june_opened": False,
        "events": {},
    }
    for event in EVENTS:
        result["events"][event] = {}
        for target in TARGETS:
            champion = freeze["event_targets"][event][target]["champion"]
            april_row = april_gate.loc[(april_gate.event.eq(event)) & (april_gate.target.eq(target)) & (april_gate.method.eq(champion))].iloc[0]
            may_row = may.loc[(may.event.eq(event)) & (may.target.eq(target)) & (may.method.eq(champion))].iloc[0]
            may_base = may.loc[(may.event.eq(event)) & (may.target.eq(target)) & (may.method.eq("all_persist_baseline"))].iloc[0]
            result["events"][event][target] = {
                "may_selected_method": champion,
                "frozen_threshold": freeze["event_targets"][event][target]["threshold"],
                "april_gate_precision": april_row.precision,
                "april_gate_recall": april_row.recall,
                "may_base_persistence_rate": may_base.precision,
                "may_precision": may_row.precision,
                "may_recall": may_row.recall,
                "may_average_precision": may_row.average_precision,
                "may_alerts_per_day": may_row.alerts_per_day,
                "may_precision_ci95": [may_row.precision_ci95_low, may_row.precision_ci95_high],
                "passes_70_precision_and_20_recall": bool(may_row.passes_70_precision_and_20_recall),
            }
    return result


def write_markdown(summary: dict[str, Any]) -> None:
    event_names = {"empty": "缺車", "full": "滿柱"}
    target_names = {"same_30": "30 分鐘後仍為同類失衡", "same_both": "30、60 分鐘兩個快照皆為同類失衡"}
    lines = [
        "# 已發生失衡的持續性預測",
        "",
        "主評分只取每段事件第一次觀測到 0 車／0 空位的快照，一個事件只產生一個預測。長事件後續重複紅燈不納入主分數。",
        "",
        "- 1–3 月：訓練與滾動交叉驗證。",
        "- 4/1–4/14：機率校正。",
        "- 4/15–4/20：方法與門檻選擇。",
        "- 4/21–4/30：凍結檢查。",
        "- 5 月：上述全部鎖定後才讀取，只作一次跨月測試。",
        "- 6 月：本實驗完全不讀取。",
        "- 沒有使用 Top-K 或 Episode any-hit。",
        "",
        "## 五月跨月結果",
        "",
    ]
    for event in EVENTS:
        lines.extend([f"### {event_names[event]}", ""])
        for target in TARGETS:
            item = summary["events"][event][target]
            lines.extend(
                [
                    f"- {target_names[target]}：四月選定 `{item['may_selected_method']}`。",
                    f"  - 全猜持續的基準率：{item['may_base_persistence_rate']:.2%}。",
                    f"  - 凍結 Precision / Recall：{item['may_precision']:.2%} / {item['may_recall']:.2%}。",
                    f"  - AP：{item['may_average_precision']:.2%}；每天平均警示 {item['may_alerts_per_day']:.1f} 筆。",
                    f"  - 是否同時達到 Precision≥70%、Recall≥20%：{'是' if item['passes_70_precision_and_20_recall'] else '否'}。",
                ]
            )
        lines.append("")
    lines.extend(
        [
            "## 限制",
            "",
            "資料沒有運補紀錄，因此結果只能解讀為『在歷史營運機制下是否仍失衡』。同類失衡消失不一定是自然恢復，也可能已有人介入；若變成相反類型的失衡，也不稱為恢復。",
        ]
    )
    (OUTPUT_DIR / "RESULT_SUMMARY_ZH.md").write_text("\n".join(lines), encoding="utf-8")


def run(force_cache: bool, n_jobs: int) -> None:
    ensure_layout()
    district_map, segment_map, _ = build_metadata_maps()
    log("Phase 1: build Jan-Apr current-red episode-onset data; May and June remain unopened")
    quality: list[dict[str, Any]] = []
    for month in (1, 2, 3, 4):
        quality.extend(build_month(month, district_map, segment_map, force_cache))
    pd.DataFrame(quality).to_csv(OUTPUT_DIR / "metrics" / "data_quality_pre_may.csv", index=False)

    log("Phase 2: Jan-Mar rolling CV and final fitting")
    cv, trees = rolling_cv(n_jobs)
    boosters, schemas, histories = train_final_models(trees, n_jobs)

    log("Phase 3: April calibration and method/threshold freeze")
    april = score_month(4, boosters, schemas, histories)
    calibrators, freeze, _ = fit_calibrators_and_freeze(april, cv, trees)
    april_gate = evaluate(april, "april_21_30_frozen_gate", freeze, (SELECTION_END, APRIL_END))
    april_gate.to_csv(OUTPUT_DIR / "metrics" / "april_21_30_frozen_metrics.csv", index=False)
    april_cohorts = cohort_descriptives([4], "april_decision_peak")
    april_cohorts.to_csv(OUTPUT_DIR / "metrics" / "april_cohort_persistence_rates.csv", index=False)
    prediction_extract(april, "april")

    # Gate results are recorded without changing the frozen choices.
    write_json(
        OUTPUT_DIR / "config" / "pre_may_final_manifest.json",
        {
            "may_opened": False,
            "june_opened": False,
            "selection_unchanged_after_april_gate": True,
            "champions": {
                event: {target: freeze["event_targets"][event][target]["champion"] for target in TARGETS}
                for event in EVENTS
            },
            "thresholds": {
                event: {target: freeze["event_targets"][event][target]["threshold"] for target in TARGETS}
                for event in EVENTS
            },
            "code_sha256": sha256(Path(__file__).resolve()),
        },
    )

    log("Phase 4: all rules frozen; now open May once for final confirmation; June stays unopened")
    quality.extend(build_month(5, district_map, segment_map, force_cache))
    pd.DataFrame(quality).to_csv(OUTPUT_DIR / "metrics" / "data_quality_jan_may.csv", index=False)
    may = score_month(5, boosters, schemas, histories)
    apply_calibrators(may, calibrators)
    may_metrics = evaluate(may, "may_frozen_confirmation", freeze)
    may_metrics.to_csv(OUTPUT_DIR / "metrics" / "may_frozen_metrics.csv", index=False)
    may_cohorts = cohort_descriptives([5], "may_decision_peak")
    may_cohorts.to_csv(OUTPUT_DIR / "metrics" / "may_cohort_persistence_rates.csv", index=False)
    prediction_extract(may, "may")

    summary = summary_json(freeze, april_gate, may_metrics, may_cohorts)
    write_json(OUTPUT_DIR / "result_summary.json", summary)
    write_markdown(summary)
    write_json(
        OUTPUT_DIR / "completion.json",
        {
            "status": "complete",
            "completed_at": pd.Timestamp.now(),
            "may_used_for_training_or_tuning": False,
            "june_opened": False,
            "june_used_for_training": False,
            "june_used_for_cross_validation": False,
            "june_used_for_calibration": False,
            "june_used_for_method_selection": False,
            "june_used_for_threshold_selection": False,
            "code_sha256": sha256(Path(__file__).resolve()),
        },
    )
    log(f"Complete: {OUTPUT_DIR / 'RESULT_SUMMARY_ZH.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.force_cache, arguments.n_jobs)
