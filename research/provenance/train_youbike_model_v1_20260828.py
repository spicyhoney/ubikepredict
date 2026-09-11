from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)


SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = PROJECT_ROOT / "outputs" / "youbike_trend_analysis_20260816" / "processed_monthly"
SEGMENT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "youbike_risk_core_analysis_20260817"
    / "analysis_data"
    / "all_station_segments.csv"
)
STATION_PROFILE_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "youbike_trend_analysis_20260816"
    / "powerbi_data"
    / "station_profile.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "youbike_model_v1_20260828"
DEFAULT_CACHE_DIR = Path(os.environ.get("TEMP", PROJECT_ROOT / "tmp")) / "youbike_model_v1_20260828_features"

PEAK_HOURS = {7, 8, 9, 16, 17, 18, 19}
MONTH_FILES = {month: SOURCE_DIR / f"youbike_clean_2026-{month:02d}.csv" for month in range(1, 7)}

READ_COLUMNS = [
    "datetime",
    "station_id",
    "station_name",
    "district",
    "capacity",
    "available_bikes",
    "available_docks",
    "unavailable_count",
    "longitude",
    "latitude",
    "empty_now",
    "full_now",
    "inactive_or_ambiguous_zero",
    "operational_observation",
]

LOGISTIC_FEATURES = [
    "available_bikes",
    "available_docks",
    "capacity",
    "unavailable_count",
    "bike_ratio",
    "dock_ratio",
    "unavailable_ratio",
    "bikes_lag_30",
    "bikes_lag_60",
    "bikes_lag_90",
    "docks_lag_30",
    "docks_lag_60",
    "docks_lag_90",
    "bike_ratio_lag_30",
    "bike_ratio_lag_60",
    "bike_ratio_lag_90",
    "dock_ratio_lag_30",
    "dock_ratio_lag_60",
    "dock_ratio_lag_90",
    "delta_bikes_30",
    "delta_bikes_60",
    "delta_bikes_90",
    "delta_docks_30",
    "delta_docks_60",
    "delta_docks_90",
    "abs_delta_bikes_30",
    "delta_bikes_ratio_30",
    "delta_docks_ratio_30",
    "bike_acceleration_30",
    "dock_acceleration_30",
    "bike_ratio_mean_90",
    "bike_ratio_min_90",
    "bike_ratio_max_90",
    "dock_ratio_mean_90",
    "dock_ratio_min_90",
    "dock_ratio_max_90",
    "bike_ratio_lag_1d",
    "dock_ratio_lag_1d",
    "bike_ratio_lag_7d",
    "dock_ratio_lag_7d",
    "target_hour_sin",
    "target_hour_cos",
    "target_weekday_sin",
    "target_weekday_cos",
    "target_is_weekend",
    "target_is_peak",
    "lag_90_missing",
    "lag_1d_missing",
    "lag_7d_missing",
    "suspected_large_jump_past",
]

LIGHTGBM_FEATURES = LOGISTIC_FEATURES + [
    "station_id",
    "district_code",
    "target_hour",
    "target_weekday",
    "longitude",
    "latitude",
]

CATEGORICAL_FEATURES = ["station_id", "district_code", "target_hour", "target_weekday"]
META_COLUMNS = [
    "datetime",
    "target_datetime",
    "station_id",
    "segment_code",
    "source_month",
    "target_is_peak",
    "y_empty_30",
    "y_full_30",
]

SEGMENT_LABELS = {
    0: "其他站",
    1: "高價值核心站",
    2: "僅高失衡站",
    3: "僅高活動站",
    4: "資料注意站",
}


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{now_text()}] {message}", flush=True)


def memory_gb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024**3)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_metadata_maps() -> tuple[dict[str, int], dict[int, int], pd.DataFrame]:
    station_profile = pd.read_csv(
        STATION_PROFILE_FILE,
        usecols=["data_split", "station_id", "station_name", "district"],
        dtype={"data_split": "string", "station_id": "int32", "station_name": "string", "district": "string"},
    )
    dev_profile = station_profile.loc[station_profile["data_split"].eq("development")].copy()
    dev_profile = dev_profile.drop_duplicates("station_id", keep="last")
    districts = sorted(str(value) for value in dev_profile["district"].dropna().unique())
    district_map = {name: index for index, name in enumerate(districts)}

    segment_map: dict[int, int] = {}
    if SEGMENT_FILE.exists():
        segments = pd.read_csv(
            SEGMENT_FILE,
            usecols=["station_id", "segment", "attention_flag"],
            dtype={"station_id": "int32", "segment": "string", "attention_flag": "string"},
        )
        for row in segments.itertuples(index=False):
            text = "" if pd.isna(row.segment) else str(row.segment)
            attention = "" if pd.isna(row.attention_flag) else str(row.attention_flag).strip()
            if attention:
                code = 4
            elif "核心" in text or ("高活動" in text and "高失衡" in text):
                code = 1
            elif "僅高失衡" in text:
                code = 2
            elif "僅高活動" in text:
                code = 3
            else:
                code = 0
            segment_map[int(row.station_id)] = code

    lookup = station_profile.sort_values(["station_id", "data_split"]).drop_duplicates("station_id", keep="last")
    lookup = lookup[["station_id", "station_name", "district"]].copy()
    lookup["district_code"] = lookup["district"].map(district_map).fillna(-1).astype("int16")
    lookup["segment_code"] = lookup["station_id"].map(segment_map).fillna(0).astype("int8")
    lookup["segment"] = lookup["segment_code"].map(SEGMENT_LABELS)
    return district_map, segment_map, lookup


def _valid_lag(grouped: Any, frame: pd.DataFrame, steps: int) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    lag_time = grouped["datetime"].shift(steps)
    lag_capacity = grouped["capacity"].shift(steps)
    lag_operational_valid = grouped["valid_operational_feature"].shift(steps)
    valid = (
        (frame["datetime"] - lag_time).dt.total_seconds().eq(steps * 1800)
        & frame["valid_operational_feature"].eq(1)
        & lag_operational_valid.eq(1)
        & frame["capacity"].eq(lag_capacity)
    )
    if steps <= 3:
        for intermediate_step in range(1, steps):
            valid &= grouped["valid_operational_feature"].shift(intermediate_step).eq(1)
            valid &= frame["capacity"].eq(grouped["capacity"].shift(intermediate_step))
    bikes = grouped["available_bikes"].shift(steps).where(valid).astype("float32")
    docks = grouped["available_docks"].shift(steps).where(valid).astype("float32")
    bike_ratio = grouped["bike_ratio"].shift(steps).where(valid).astype("float32")
    dock_ratio = grouped["dock_ratio"].shift(steps).where(valid).astype("float32")
    return valid, bikes, docks, bike_ratio, dock_ratio


def _row_stat(columns: Iterable[pd.Series], operation: str) -> pd.Series:
    matrix = pd.concat(list(columns), axis=1)
    if operation == "mean":
        result = matrix.mean(axis=1, skipna=False)
    elif operation == "min":
        result = matrix.min(axis=1, skipna=False)
    elif operation == "max":
        result = matrix.max(axis=1, skipna=False)
    else:
        raise ValueError(operation)
    return result.astype("float32")


def prepare_month(
    month: int,
    cache_dir: Path,
    district_map: dict[str, int],
    segment_map: dict[int, int],
    smoke: bool = False,
) -> dict[str, Any]:
    source = MONTH_FILES[month]
    log(f"Month {month:02d}: reading {source.name}")
    frame = pd.read_csv(
        source,
        usecols=READ_COLUMNS,
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
        parse_dates=["datetime"],
    )
    raw_rows = len(frame)
    if smoke:
        days = frame["datetime"].dt.day
        if month == 5:
            frame = frame.loc[days.isin([1, 21])].copy()
        else:
            frame = frame.loc[days.eq(1)].copy()
    frame.sort_values(["station_id", "datetime"], inplace=True, kind="mergesort")
    frame.reset_index(drop=True, inplace=True)

    capacity = frame["capacity"].astype("float32")
    bikes = frame["available_bikes"].astype("float32")
    docks = frame["available_docks"].astype("float32")
    unavailable = frame["unavailable_count"].astype("float32")
    positive_capacity = capacity.gt(0)
    row_valid = (
        positive_capacity
        & bikes.notna()
        & docks.notna()
        & bikes.ge(0)
        & docks.ge(0)
        & bikes.le(capacity)
        & docks.le(capacity)
        & (bikes + docks).le(capacity)
        & unavailable.ge(0)
    )
    valid_operational = (
        row_valid
        & frame["operational_observation"].eq(1)
        & frame["inactive_or_ambiguous_zero"].eq(0)
    )
    frame["row_valid_feature"] = row_valid.astype("int8")
    frame["valid_operational_feature"] = valid_operational.astype("int8")
    frame["bike_ratio"] = (bikes / capacity).where(positive_capacity).astype("float32")
    frame["dock_ratio"] = (docks / capacity).where(positive_capacity).astype("float32")
    frame["unavailable_ratio"] = (unavailable / capacity).where(positive_capacity).astype("float32")

    grouped = frame.groupby("station_id", sort=False, observed=True)
    lag_values: dict[int, tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]] = {}
    for steps in [1, 2, 3, 48, 336]:
        lag_values[steps] = _valid_lag(grouped, frame, steps)

    for steps, minutes in [(1, 30), (2, 60), (3, 90)]:
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
    target_weekday_zero = target_datetime.dt.weekday.astype("int8")
    frame["target_datetime"] = target_datetime
    frame["target_hour"] = target_hour
    frame["target_weekday"] = (target_weekday_zero + 1).astype("int8")
    frame["target_hour_sin"] = np.sin(2 * np.pi * target_hour / 24).astype("float32")
    frame["target_hour_cos"] = np.cos(2 * np.pi * target_hour / 24).astype("float32")
    frame["target_weekday_sin"] = np.sin(2 * np.pi * target_weekday_zero / 7).astype("float32")
    frame["target_weekday_cos"] = np.cos(2 * np.pi * target_weekday_zero / 7).astype("float32")
    frame["target_is_weekend"] = target_weekday_zero.ge(5).astype("int8")
    frame["target_is_peak"] = target_hour.isin(PEAK_HOURS).astype("int8")
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

    next_time = grouped["datetime"].shift(-1)
    valid_next = (next_time - frame["datetime"]).dt.total_seconds().eq(1800)
    next_operational = grouped["operational_observation"].shift(-1)
    next_inactive = grouped["inactive_or_ambiguous_zero"].shift(-1)
    next_empty = grouped["empty_now"].shift(-1)
    next_full = grouped["full_now"].shift(-1)
    next_row_valid = row_valid.groupby(frame["station_id"], sort=False).shift(-1)
    next_capacity = grouped["capacity"].shift(-1)
    eligible = (
        valid_next
        & row_valid
        & frame["operational_observation"].eq(1)
        & frame["inactive_or_ambiguous_zero"].eq(0)
        & frame["empty_now"].eq(0)
        & frame["full_now"].eq(0)
        & next_operational.eq(1)
        & next_inactive.eq(0)
        & next_row_valid.eq(True)
        & frame["capacity"].eq(next_capacity)
        & next_empty.notna()
        & next_full.notna()
    )
    frame["y_empty_30"] = next_empty.fillna(0).astype("int8")
    frame["y_full_30"] = next_full.fillna(0).astype("int8")
    selected_columns = list(dict.fromkeys(META_COLUMNS + LIGHTGBM_FEATURES))
    eligible_frame = frame.loc[eligible, selected_columns].copy()

    for column in LIGHTGBM_FEATURES:
        if column in CATEGORICAL_FEATURES:
            eligible_frame[column] = eligible_frame[column].astype("int32")
        elif eligible_frame[column].dtype.kind in "f":
            eligible_frame[column] = eligible_frame[column].astype("float32")
    output = cache_dir / f"features_2026_{month:02d}.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    eligible_frame.to_parquet(output, engine="pyarrow", compression="zstd", index=False)
    result = {
        "month": month,
        "source_rows": raw_rows,
        "rows_after_smoke_filter": len(frame),
        "eligible_rows": len(eligible_frame),
        "empty_positives": int(eligible_frame["y_empty_30"].sum()),
        "full_positives": int(eligible_frame["y_full_30"].sum()),
        "peak_rows": int(eligible_frame["target_is_peak"].sum()),
        "invalid_inventory_rows": int((~row_valid).sum()),
        "cache_file": str(output),
        "cache_size_mb": round(output.stat().st_size / (1024**2), 2),
    }
    log(
        f"Month {month:02d}: {len(eligible_frame):,} eligible rows, "
        f"empty={result['empty_positives']:,}, full={result['full_positives']:,}, "
        f"cache={result['cache_size_mb']:.1f} MB, RAM={memory_gb():.1f} GB"
    )
    del frame, eligible_frame
    gc.collect()
    return result


def prepare_feature_cache(
    cache_dir: Path,
    smoke: bool,
    force: bool,
    months: Iterable[int],
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    district_map, segment_map, lookup = build_metadata_maps()
    cache_dir.mkdir(parents=True, exist_ok=True)
    write_json(cache_dir / "feature_definition.json", {
        "created_at": now_text(),
        "smoke": smoke,
        "district_map": district_map,
        "segment_labels": SEGMENT_LABELS,
        "logistic_features": LOGISTIC_FEATURES,
        "lightgbm_features": LIGHTGBM_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "label_definition": {
            "eligible": "current snapshot is operational and neither empty nor full; exact next 30-minute snapshot is operational",
            "y_empty_30": "next exact 30-minute snapshot has zero bikes and positive docks",
            "y_full_30": "next exact 30-minute snapshot has zero docks and positive bikes",
        },
        "suspected_jump_definition": "past observed 30-minute absolute bike change >= max(5 bikes, 25% of current capacity); candidate signal only",
    })
    summaries: list[dict[str, Any]] = []
    for month in months:
        target = cache_dir / f"features_2026_{month:02d}.parquet"
        if target.exists() and not force:
            cached = pd.read_parquet(target, columns=["y_empty_30", "y_full_30", "target_is_peak"])
            summaries.append({
                "month": month,
                "source_rows": None,
                "rows_after_smoke_filter": None,
                "eligible_rows": len(cached),
                "empty_positives": int(cached["y_empty_30"].sum()),
                "full_positives": int(cached["y_full_30"].sum()),
                "peak_rows": int(cached["target_is_peak"].sum()),
                "invalid_inventory_rows": None,
                "cache_file": str(target),
                "cache_size_mb": round(target.stat().st_size / (1024**2), 2),
                "reused": True,
            })
            del cached
        else:
            summaries.append(prepare_month(month, cache_dir, district_map, segment_map, smoke=smoke))
    return summaries, lookup


@dataclass
class NumericTransformer:
    features: list[str]
    medians: np.ndarray | None = None
    means: np.ndarray | None = None
    scales: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame) -> "NumericTransformer":
        medians: list[float] = []
        means: list[float] = []
        scales: list[float] = []
        for feature in self.features:
            values = frame[feature].to_numpy(dtype=np.float32, copy=False)
            finite = np.isfinite(values)
            if finite.any():
                median = float(np.median(values[finite]))
                mean = float(values[finite].mean(dtype=np.float64))
                scale = float(values[finite].std(dtype=np.float64))
            else:
                median, mean, scale = 0.0, 0.0, 1.0
            if not np.isfinite(scale) or scale < 1e-8:
                scale = 1.0
            medians.append(median)
            means.append(mean)
            scales.append(scale)
        self.medians = np.asarray(medians, dtype=np.float32)
        self.means = np.asarray(means, dtype=np.float32)
        self.scales = np.asarray(scales, dtype=np.float32)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.medians is None or self.means is None or self.scales is None:
            raise RuntimeError("Transformer is not fitted")
        matrix = frame[self.features].to_numpy(dtype=np.float32, copy=True)
        for index in range(matrix.shape[1]):
            missing = ~np.isfinite(matrix[:, index])
            if missing.any():
                matrix[missing, index] = self.medians[index]
            matrix[:, index] -= self.means[index]
            matrix[:, index] /= self.scales[index]
        return matrix

    def as_dict(self) -> dict[str, Any]:
        return {
            "features": self.features,
            "medians": self.medians.tolist() if self.medians is not None else None,
            "means": self.means.tolist() if self.means is not None else None,
            "scales": self.scales.tolist() if self.scales is not None else None,
        }


@dataclass
class PlattCalibrator:
    slope: float
    intercept: float

    @classmethod
    def fit(cls, raw_score: np.ndarray, y_true: np.ndarray) -> "PlattCalibrator":
        finite = np.isfinite(raw_score)
        usable_y = np.asarray(y_true[finite], dtype=np.int8)
        if usable_y.size == 0:
            return cls(slope=0.0, intercept=0.0)
        if np.unique(usable_y).size < 2:
            # A tiny smoke-test slice can contain only one class.  Use a
            # smoothed constant probability instead of failing calibration.
            probability = (float(usable_y.sum()) + 0.5) / (float(usable_y.size) + 1.0)
            probability = float(np.clip(probability, 1e-6, 1.0 - 1e-6))
            return cls(slope=0.0, intercept=float(np.log(probability / (1.0 - probability))))
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000, random_state=SEED)
        model.fit(raw_score[finite].reshape(-1, 1), usable_y)
        return cls(slope=float(model.coef_[0, 0]), intercept=float(model.intercept_[0]))

    def predict(self, raw_score: np.ndarray) -> np.ndarray:
        linear = np.clip(self.slope * raw_score + self.intercept, -35.0, 35.0)
        return (1.0 / (1.0 + np.exp(-linear))).astype(np.float32)

    def as_dict(self) -> dict[str, float]:
        return {"slope": self.slope, "intercept": self.intercept}


def load_months(cache_dir: Path, months: Iterable[int], columns: list[str] | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for month in months:
        path = cache_dir / f"features_2026_{month:02d}.parquet"
        frames.append(pd.read_parquet(path, columns=columns))
    result = pd.concat(frames, ignore_index=True)
    del frames
    return result


def augment_station_lookup_after_holdout(station_lookup: pd.DataFrame, month: int) -> pd.DataFrame:
    """Add display metadata for stations first appearing in the sealed holdout.

    This is intentionally called only after model/calibration/threshold freezing.
    Station names and districts are used for reporting only, never as model input.
    """
    monthly = pd.read_csv(
        MONTH_FILES[month],
        usecols=["station_id", "station_name", "district"],
        dtype={"station_id": "int32", "station_name": "string", "district": "string"},
    ).drop_duplicates("station_id", keep="last")
    metadata = station_lookup[["station_id", "district_code", "segment_code", "segment"]]
    monthly = monthly.merge(metadata, on="station_id", how="left")
    district_codes = (
        station_lookup.dropna(subset=["district"])
        .drop_duplicates("district")
        .set_index("district")["district_code"]
        .to_dict()
    )
    monthly["district_code"] = (
        monthly["district_code"].fillna(monthly["district"].map(district_codes)).fillna(-1).astype("int16")
    )
    monthly["segment_code"] = monthly["segment_code"].fillna(0).astype("int8")
    monthly["segment"] = monthly["segment"].fillna(monthly["segment_code"].map(SEGMENT_LABELS))
    existing_not_seen = station_lookup.loc[~station_lookup["station_id"].isin(monthly["station_id"])]
    combined = pd.concat([existing_not_seen, monthly], ignore_index=True)
    return combined.sort_values("station_id").reset_index(drop=True)


def binary_counts(y_true: np.ndarray, alerts: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=np.int8)
    predicted = np.asarray(alerts, dtype=bool)
    tp = int(np.sum((y == 1) & predicted))
    fp = int(np.sum((y == 0) & predicted))
    fn = int(np.sum((y == 1) & ~predicted))
    tn = int(np.sum((y == 0) & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "alerts": int(predicted.sum()),
        "precision": precision,
        "recall": recall,
        "f2": f2,
    }


def choose_best_from_candidates(
    y_true: np.ndarray,
    values: np.ndarray,
    candidates: Iterable[float],
    direction: str,
    method: str,
    event: str,
) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for candidate in sorted(set(float(item) for item in candidates)):
        if direction == "le":
            alerts = values <= candidate
        elif direction == "ge":
            alerts = values >= candidate
        else:
            raise ValueError(direction)
        counts = binary_counts(y_true, alerts)
        rows.append({"event": event, "method": method, "candidate": candidate, **counts})
    best = max(rows, key=lambda row: (float(row["f2"]), -int(row["alerts"]), float(row["candidate"])))
    return float(best["candidate"]), rows


def choose_probability_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    method: str,
    event: str,
    quantile_count: int = 1001,
) -> tuple[float, list[dict[str, Any]]]:
    finite = np.isfinite(probabilities)
    y = y_true[finite]
    scores = probabilities[finite]
    candidates = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, quantile_count)))
    return choose_best_from_candidates(y, scores, candidates, "ge", method, event)


def safe_ap(y_true: np.ndarray, score: np.ndarray) -> float:
    finite = np.isfinite(score)
    if not finite.any() or np.unique(y_true[finite]).size < 2:
        return math.nan
    return float(average_precision_score(y_true[finite], score[finite]))


def safe_roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    finite = np.isfinite(score)
    if not finite.any() or np.unique(y_true[finite]).size < 2:
        return math.nan
    return float(roc_auc_score(y_true[finite], score[finite]))


def evaluate_method(
    meta: pd.DataFrame,
    y_true: np.ndarray,
    score: np.ndarray,
    alerts: np.ndarray,
    event: str,
    method: str,
    subset: str,
    probability: np.ndarray | None = None,
) -> dict[str, Any]:
    counts = binary_counts(y_true, alerts)
    station_days = max(1, meta["station_id"].nunique() * meta["target_datetime"].dt.normalize().nunique())
    result: dict[str, Any] = {
        "event": event,
        "method": method,
        "subset": subset,
        "observations": int(len(y_true)),
        "positives": int(y_true.sum()),
        "prevalence": float(y_true.mean()) if len(y_true) else math.nan,
        "average_precision": safe_ap(y_true, score),
        "roc_auc": safe_roc_auc(y_true, score),
        "false_alerts_per_100_station_days": 100.0 * int(counts["fp"]) / station_days,
        **counts,
    }
    if probability is not None:
        finite = np.isfinite(probability)
        clipped = np.clip(probability[finite], 1e-7, 1 - 1e-7)
        result["brier"] = float(brier_score_loss(y_true[finite], clipped))
        result["log_loss"] = float(log_loss(y_true[finite], clipped, labels=[0, 1]))
    else:
        result["brier"] = math.nan
        result["log_loss"] = math.nan
    return result


def topk_metrics(
    meta: pd.DataFrame,
    y_true: np.ndarray,
    score: np.ndarray,
    event: str,
    method: str,
    subset: str,
    ks: tuple[int, ...] = (5, 10, 20),
) -> list[dict[str, Any]]:
    time_values = meta["target_datetime"].to_numpy(dtype="datetime64[ns]").astype("int64")
    finite_scores = np.where(np.isfinite(score), score, -np.inf)
    order = np.lexsort((-finite_scores, time_values))
    sorted_times = time_values[order]
    new_group = np.ones(len(order), dtype=bool)
    if len(order) > 1:
        new_group[1:] = sorted_times[1:] != sorted_times[:-1]
    starts = np.maximum.accumulate(np.where(new_group, np.arange(len(order)), 0))
    ranks = np.arange(len(order)) - starts
    decision_times = int(new_group.sum())
    positives = int(y_true.sum())
    days = max(1, meta["target_datetime"].dt.normalize().nunique())
    stations = max(1, meta["station_id"].nunique())
    rows: list[dict[str, Any]] = []
    for k in ks:
        selected_order = order[ranks < k]
        tp = int(y_true[selected_order].sum())
        alerts = len(selected_order)
        fp = alerts - tp
        rows.append({
            "event": event,
            "method": method,
            "subset": subset,
            "k": k,
            "decision_times": decision_times,
            "alerts": alerts,
            "tp": tp,
            "fp": fp,
            "recall": tp / positives if positives else 0.0,
            "precision": tp / alerts if alerts else 0.0,
            "alerts_per_day": alerts / days,
            "false_alerts_per_100_station_days": 100.0 * fp / (stations * days),
        })
    return rows


def reliability_rows(
    y_true: np.ndarray,
    probability: np.ndarray,
    event: str,
    method: str,
    subset: str,
    bins: int = 10,
) -> list[dict[str, Any]]:
    temporary = pd.DataFrame({"y": y_true, "p": probability}).dropna()
    if temporary.empty:
        return []
    try:
        temporary["bin"] = pd.qcut(temporary["p"], q=bins, duplicates="drop")
    except ValueError:
        temporary["bin"] = "all"
    if temporary["bin"].isna().all():
        temporary["bin"] = "all"
    grouped = temporary.groupby("bin", observed=True).agg(
        observations=("y", "size"),
        positives=("y", "sum"),
        mean_probability=("p", "mean"),
        observed_rate=("y", "mean"),
        min_probability=("p", "min"),
        max_probability=("p", "max"),
    ).reset_index(drop=True)
    grouped["event"] = event
    grouped["method"] = method
    grouped["subset"] = subset
    grouped["bin_number"] = np.arange(1, len(grouped) + 1)
    total = max(1, grouped["observations"].sum())
    grouped["ece_component"] = (
        grouped["observations"] / total * (grouped["observed_rate"] - grouped["mean_probability"]).abs()
    )
    return grouped.to_dict("records")


def pr_curve_rows(
    y_true: np.ndarray,
    score: np.ndarray,
    event: str,
    method: str,
    subset: str,
    points: int = 150,
) -> list[dict[str, Any]]:
    finite = np.isfinite(score)
    precision, recall, thresholds = precision_recall_curve(y_true[finite], score[finite])
    threshold_values = np.append(thresholds, np.nan)
    if len(precision) > points:
        indices = np.unique(np.linspace(0, len(precision) - 1, points).astype(int))
    else:
        indices = np.arange(len(precision))
    return [
        {
            "event": event,
            "method": method,
            "subset": subset,
            "point": int(point),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "threshold": None if not np.isfinite(threshold_values[index]) else float(threshold_values[index]),
        }
        for point, index in enumerate(indices, start=1)
    ]


def lgb_parameters(n_estimators: int, n_jobs: int) -> dict[str, Any]:
    return {
        "objective": "binary",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "n_estimators": n_estimators,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 1000,
        "max_bin": 127,
        "subsample": 0.80,
        "subsample_freq": 1,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.10,
        "reg_lambda": 2.00,
        "max_cat_threshold": 32,
        "cat_smooth": 20,
        "cat_l2": 10,
        "scale_pos_weight": 1.0,
        "deterministic": True,
        "force_col_wise": True,
        "random_state": SEED,
        "n_jobs": n_jobs,
        "verbosity": -1,
    }


def train_logistic_models(train: pd.DataFrame, model_dir: Path, smoke: bool) -> tuple[NumericTransformer, dict[str, SGDClassifier]]:
    log("Logistic: fitting train-only imputation and scaling")
    transformer = NumericTransformer(LOGISTIC_FEATURES).fit(train)
    matrix = transformer.transform(train)
    models: dict[str, SGDClassifier] = {}
    for event, target in [("empty", "y_empty_30"), ("full", "y_full_30")]:
        log(f"Logistic {event}: fitting on {len(train):,} rows")
        model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-5,
            learning_rate="optimal",
            average=True,
            class_weight=None,
            max_iter=3 if smoke else 5,
            tol=None,
            shuffle=True,
            random_state=SEED,
        )
        model.fit(matrix, train[target].to_numpy(dtype=np.int8))
        models[event] = model
        joblib.dump(model, model_dir / f"logistic_{event}_30.joblib", compress=3)
    write_json(model_dir / "logistic_transformer.json", transformer.as_dict())
    del matrix
    gc.collect()
    return transformer, models


def train_lightgbm_models(
    train: pd.DataFrame,
    model_dir: Path,
    smoke: bool,
    n_jobs: int,
) -> tuple[dict[str, lgb.LGBMClassifier], dict[str, int], list[dict[str, Any]]]:
    models: dict[str, lgb.LGBMClassifier] = {}
    best_iterations: dict[str, int] = {}
    importance_rows: list[dict[str, Any]] = []
    development_mask = train["source_month"].le(3).to_numpy()
    april_mask = train["source_month"].eq(4).to_numpy()
    maximum_trees = 60 if smoke else 600
    for event, target in [("empty", "y_empty_30"), ("full", "y_full_30")]:
        log(f"LightGBM {event}: time validation Jan-Mar -> Apr")
        probe = lgb.LGBMClassifier(**lgb_parameters(maximum_trees, n_jobs))
        probe.fit(
            train.loc[development_mask, LIGHTGBM_FEATURES],
            train.loc[development_mask, target],
            categorical_feature=CATEGORICAL_FEATURES,
            eval_set=[(train.loc[april_mask, LIGHTGBM_FEATURES], train.loc[april_mask, target])],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(20 if smoke else 50, verbose=True), lgb.log_evaluation(25)],
        )
        best_iteration = int(probe.best_iteration_ or maximum_trees)
        best_iteration = max(20 if smoke else 50, best_iteration)
        best_iterations[event] = best_iteration
        del probe
        gc.collect()

        log(f"LightGBM {event}: refitting Jan-Apr with {best_iteration} trees")
        model = lgb.LGBMClassifier(**lgb_parameters(best_iteration, n_jobs))
        model.fit(
            train[LIGHTGBM_FEATURES],
            train[target],
            categorical_feature=CATEGORICAL_FEATURES,
            callbacks=[lgb.log_evaluation(50)],
        )
        models[event] = model
        # LightGBM's native file writer can fail on Windows paths containing
        # non-ASCII characters.  Let Python write the serialized model string.
        (model_dir / f"lightgbm_{event}_30.txt").write_text(
            model.booster_.model_to_string(), encoding="utf-8"
        )
        gains = model.booster_.feature_importance(importance_type="gain")
        splits = model.booster_.feature_importance(importance_type="split")
        for feature, gain, split_count in zip(model.booster_.feature_name(), gains, splits, strict=True):
            importance_rows.append({
                "event": event,
                "feature": feature,
                "gain": float(gain),
                "split_count": int(split_count),
            })
        gc.collect()
    return models, best_iterations, importance_rows


def predict_model_scores(
    frame: pd.DataFrame,
    transformer: NumericTransformer,
    logistic_models: dict[str, SGDClassifier],
    lightgbm_models: dict[str, lgb.LGBMClassifier],
) -> dict[str, dict[str, np.ndarray]]:
    logistic_matrix = transformer.transform(frame)
    scores: dict[str, dict[str, np.ndarray]] = {"empty": {}, "full": {}}
    for event in ["empty", "full"]:
        scores[event]["logistic_raw"] = logistic_models[event].decision_function(logistic_matrix).astype(np.float32)
    del logistic_matrix
    gc.collect()
    for event in ["empty", "full"]:
        scores[event]["lightgbm_raw"] = lightgbm_models[event].booster_.predict(
            frame[LIGHTGBM_FEATURES], raw_score=True
        ).astype(np.float32)
    capacity = frame["capacity"].to_numpy(dtype=np.float32)
    for event, ratio_column, value_column, delta_column in [
        ("empty", "bike_ratio", "available_bikes", "delta_bikes_30"),
        ("full", "dock_ratio", "available_docks", "delta_docks_30"),
    ]:
        ratio = frame[ratio_column].to_numpy(dtype=np.float32)
        value = frame[value_column].to_numpy(dtype=np.float32)
        delta = frame[delta_column].to_numpy(dtype=np.float32)
        projected_ratio = (value + np.where(np.isfinite(delta), delta, 0.0)) / capacity
        scores[event]["ratio_value"] = ratio
        scores[event]["ratio_score"] = -ratio
        scores[event]["linear_value"] = projected_ratio.astype(np.float32)
        scores[event]["linear_score"] = (-projected_ratio).astype(np.float32)
    return scores


def calibrate_and_select_thresholds(
    may: pd.DataFrame,
    transformer: NumericTransformer,
    logistic_models: dict[str, SGDClassifier],
    lightgbm_models: dict[str, lgb.LGBMClassifier],
    output_dir: Path,
) -> tuple[dict[str, dict[str, PlattCalibrator]], dict[str, dict[str, float]], pd.DataFrame]:
    log("May: generating scores for calibration and threshold selection")
    scores = predict_model_scores(may, transformer, logistic_models, lightgbm_models)
    calibration_mask = may["target_datetime"].lt(pd.Timestamp("2026-05-21")).to_numpy()
    threshold_mask = may["target_datetime"].ge(pd.Timestamp("2026-05-21")).to_numpy()
    threshold_peak_mask = threshold_mask & may["target_is_peak"].eq(1).to_numpy()
    if calibration_mask.sum() == 0 or threshold_peak_mask.sum() == 0:
        raise RuntimeError("May calibration or threshold split is empty")

    calibrators: dict[str, dict[str, PlattCalibrator]] = {"empty": {}, "full": {}}
    thresholds: dict[str, dict[str, float]] = {"empty": {}, "full": {}}
    search_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    reliability: list[dict[str, Any]] = []

    for event, target in [("empty", "y_empty_30"), ("full", "y_full_30")]:
        y = may[target].to_numpy(dtype=np.int8)
        for method in ["logistic", "lightgbm"]:
            calibrator = PlattCalibrator.fit(scores[event][f"{method}_raw"][calibration_mask], y[calibration_mask])
            calibrators[event][method] = calibrator
            scores[event][f"{method}_probability"] = calibrator.predict(scores[event][f"{method}_raw"])
            reliability.extend(
                reliability_rows(
                    y[calibration_mask],
                    scores[event][f"{method}_probability"][calibration_mask],
                    event,
                    method,
                    "may_calibration_01_20",
                )
            )

        ratio_threshold, rows = choose_best_from_candidates(
            y[threshold_peak_mask],
            scores[event]["ratio_value"][threshold_peak_mask],
            [0.02, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25],
            "le",
            "ratio_rule",
            event,
        )
        thresholds[event]["ratio_rule"] = ratio_threshold
        search_rows.extend(rows)

        linear_threshold, rows = choose_best_from_candidates(
            y[threshold_peak_mask],
            scores[event]["linear_value"][threshold_peak_mask],
            [0.0, 0.02, 0.05],
            "le",
            "linear_extrapolation",
            event,
        )
        thresholds[event]["linear_extrapolation"] = linear_threshold
        search_rows.extend(rows)

        for method in ["logistic", "lightgbm"]:
            threshold, rows = choose_probability_threshold(
                y[threshold_peak_mask],
                scores[event][f"{method}_probability"][threshold_peak_mask],
                method,
                event,
            )
            thresholds[event][method] = threshold
            search_rows.extend(rows)

        peak_meta = may.loc[threshold_peak_mask, META_COLUMNS].copy()
        for method in ["ratio_rule", "linear_extrapolation", "logistic", "lightgbm"]:
            if method == "ratio_rule":
                method_score = scores[event]["ratio_score"][threshold_peak_mask]
                alerts = scores[event]["ratio_value"][threshold_peak_mask] <= thresholds[event][method]
                probability = None
            elif method == "linear_extrapolation":
                method_score = scores[event]["linear_score"][threshold_peak_mask]
                alerts = scores[event]["linear_value"][threshold_peak_mask] <= thresholds[event][method]
                probability = None
            else:
                probability = scores[event][f"{method}_probability"][threshold_peak_mask]
                method_score = probability
                alerts = probability >= thresholds[event][method]
            validation_rows.append(
                evaluate_method(
                    peak_meta,
                    y[threshold_peak_mask],
                    method_score,
                    alerts,
                    event,
                    method,
                    "may_threshold_peak_21_31",
                    probability,
                )
            )

    pd.DataFrame(search_rows).to_csv(output_dir / "data" / "threshold_search_may.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(validation_rows).to_csv(output_dir / "data" / "may_threshold_performance.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(reliability).to_csv(output_dir / "data" / "may_calibration_table.csv", index=False, encoding="utf-8-sig")
    calibrator_json = {
        event: {method: calibrator.as_dict() for method, calibrator in methods.items()}
        for event, methods in calibrators.items()
    }
    write_json(output_dir / "models" / "platt_calibrators.json", calibrator_json)
    write_json(output_dir / "models" / "selected_thresholds.json", thresholds)
    log(f"May thresholds frozen: {thresholds}")
    del scores
    gc.collect()
    return calibrators, thresholds, pd.DataFrame(validation_rows)


def method_arrays(
    frame: pd.DataFrame,
    event_scores: dict[str, np.ndarray],
    event_thresholds: dict[str, float],
    calibrators: dict[str, PlattCalibrator],
) -> dict[str, dict[str, np.ndarray | None]]:
    logistic_probability = calibrators["logistic"].predict(event_scores["logistic_raw"])
    lightgbm_probability = calibrators["lightgbm"].predict(event_scores["lightgbm_raw"])
    return {
        "ratio_rule": {
            "score": event_scores["ratio_score"],
            "alerts": event_scores["ratio_value"] <= event_thresholds["ratio_rule"],
            "probability": None,
        },
        "linear_extrapolation": {
            "score": event_scores["linear_score"],
            "alerts": event_scores["linear_value"] <= event_thresholds["linear_extrapolation"],
            "probability": None,
        },
        "logistic": {
            "score": logistic_probability,
            "alerts": logistic_probability >= event_thresholds["logistic"],
            "probability": logistic_probability,
        },
        "lightgbm": {
            "score": lightgbm_probability,
            "alerts": lightgbm_probability >= event_thresholds["lightgbm"],
            "probability": lightgbm_probability,
        },
    }


def evaluate_holdout(
    test: pd.DataFrame,
    transformer: NumericTransformer,
    logistic_models: dict[str, SGDClassifier],
    lightgbm_models: dict[str, lgb.LGBMClassifier],
    calibrators: dict[str, dict[str, PlattCalibrator]],
    thresholds: dict[str, dict[str, float]],
    station_lookup: pd.DataFrame,
    output_dir: Path,
    holdout_label: str,
) -> dict[str, pd.DataFrame]:
    log(f"{holdout_label}: generating frozen-model predictions")
    raw_scores = predict_model_scores(test, transformer, logistic_models, lightgbm_models)
    performance_rows: list[dict[str, Any]] = []
    topk_rows: list[dict[str, Any]] = []
    reliability: list[dict[str, Any]] = []
    pr_rows: list[dict[str, Any]] = []
    subgroup_rows: list[dict[str, Any]] = []
    case_frames: list[pd.DataFrame] = []
    prediction_output = test[META_COLUMNS + [
        "available_bikes", "available_docks", "capacity", "bike_ratio", "dock_ratio",
        "delta_bikes_30", "delta_docks_30", "suspected_large_jump_past",
    ]].copy()

    for event, target in [("empty", "y_empty_30"), ("full", "y_full_30")]:
        y_all = test[target].to_numpy(dtype=np.int8)
        methods = method_arrays(test, raw_scores[event], thresholds[event], calibrators[event])
        for method, arrays in methods.items():
            prediction_output[f"{event}_{method}_score"] = np.asarray(arrays["score"], dtype=np.float32)
            prediction_output[f"{event}_{method}_alert"] = np.asarray(arrays["alerts"], dtype=np.int8)

        for subset, subset_mask in [
            ("all_day", np.ones(len(test), dtype=bool)),
            ("peak", test["target_is_peak"].eq(1).to_numpy()),
        ]:
            meta = test.loc[subset_mask, META_COLUMNS].copy()
            y = y_all[subset_mask]
            for method, arrays in methods.items():
                score = np.asarray(arrays["score"])[subset_mask]
                alerts = np.asarray(arrays["alerts"])[subset_mask]
                probability_array = arrays["probability"]
                probability = None if probability_array is None else np.asarray(probability_array)[subset_mask]
                performance_rows.append(
                    evaluate_method(meta, y, score, alerts, event, method, subset, probability)
                )
                topk_rows.extend(topk_metrics(meta, y, score, event, method, subset))
                if subset == "peak":
                    pr_rows.extend(pr_curve_rows(y, score, event, method, subset))
                if probability is not None:
                    reliability.extend(reliability_rows(y, probability, event, method, subset))

        peak_mask = test["target_is_peak"].eq(1).to_numpy()
        lgb_arrays = methods["lightgbm"]
        for segment_code, segment_label in SEGMENT_LABELS.items():
            mask = peak_mask & test["segment_code"].eq(segment_code).to_numpy()
            if mask.sum() == 0:
                continue
            subgroup_rows.append(
                evaluate_method(
                    test.loc[mask, META_COLUMNS],
                    y_all[mask],
                    np.asarray(lgb_arrays["score"])[mask],
                    np.asarray(lgb_arrays["alerts"])[mask],
                    event,
                    "lightgbm",
                    f"peak_segment_{segment_label}",
                    np.asarray(lgb_arrays["probability"])[mask],
                )
            )

        probability = np.asarray(methods["lightgbm"]["probability"])
        alerts = np.asarray(methods["lightgbm"]["alerts"], dtype=bool)
        categories = {
            "TP": (y_all == 1) & alerts,
            "FP": (y_all == 0) & alerts,
            "FN": (y_all == 1) & ~alerts,
        }
        for category, mask in categories.items():
            indices = np.flatnonzero(mask & peak_mask)
            if len(indices) == 0:
                continue
            selected = indices[np.argsort(probability[indices])[-20:]]
            cases = prediction_output.iloc[selected].copy()
            cases["event"] = event
            cases["case_type"] = category
            cases["lightgbm_probability"] = probability[selected]
            case_frames.append(cases)

    performance = pd.DataFrame(performance_rows)
    topk = pd.DataFrame(topk_rows)
    reliability_frame = pd.DataFrame(reliability)
    pr_frame = pd.DataFrame(pr_rows)
    subgroup = pd.DataFrame(subgroup_rows)
    cases = pd.concat(case_frames, ignore_index=True) if case_frames else pd.DataFrame()
    if not cases.empty:
        cases = cases.merge(station_lookup, on="station_id", how="left", suffixes=("", "_lookup"))

    data_dir = output_dir / "data"
    performance.to_csv(data_dir / "test_performance.csv", index=False, encoding="utf-8-sig")
    topk.to_csv(data_dir / "test_topk.csv", index=False, encoding="utf-8-sig")
    reliability_frame.to_csv(data_dir / "test_reliability.csv", index=False, encoding="utf-8-sig")
    pr_frame.to_csv(data_dir / "test_pr_curve.csv", index=False, encoding="utf-8-sig")
    subgroup.to_csv(data_dir / "test_subgroup_performance.csv", index=False, encoding="utf-8-sig")
    cases.to_csv(data_dir / "test_case_samples.csv", index=False, encoding="utf-8-sig")
    prediction_output.to_parquet(data_dir / "test_predictions.parquet", compression="zstd", index=False)
    log(f"{holdout_label}: evaluation tables saved")
    del raw_scores, prediction_output
    gc.collect()
    return {
        "performance": performance,
        "topk": topk,
        "reliability": reliability_frame,
        "pr_curve": pr_frame,
        "subgroup": subgroup,
        "cases": cases,
    }


def validate_source_files() -> None:
    missing = [str(path) for path in MONTH_FILES.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing cleaned monthly files: {missing}")
    if not STATION_PROFILE_FILE.exists():
        raise FileNotFoundError(STATION_PROFILE_FILE)


def create_output_layout(output_dir: Path) -> None:
    (output_dir / "data").mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)


def run_pipeline(args: argparse.Namespace) -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    validate_source_files()
    output_dir = args.output_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    create_output_layout(output_dir)
    source_snapshot = output_dir / "model_pipeline_source.py"
    source_snapshot.write_bytes(Path(__file__).read_bytes())
    timings: dict[str, float] = {}
    overall_start = time.perf_counter()
    log(f"Output: {output_dir}")
    log(f"Feature cache: {cache_dir}")
    log(f"Mode: {'SMOKE' if args.smoke else 'FULL'}; LightGBM threads={args.threads}")

    start = time.perf_counter()
    summaries, station_lookup = prepare_feature_cache(
        cache_dir,
        smoke=args.smoke,
        force=args.force_features,
        months=range(1, 6),
    )
    timings["prepare_months_1_5_seconds"] = time.perf_counter() - start
    station_lookup.to_csv(output_dir / "data" / "station_lookup.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summaries).to_csv(output_dir / "data" / "dataset_summary.csv", index=False, encoding="utf-8-sig")

    start = time.perf_counter()
    train_columns = list(dict.fromkeys(META_COLUMNS + LIGHTGBM_FEATURES))
    train = load_months(cache_dir, range(1, 5), columns=train_columns)
    may = load_months(cache_dir, [5], columns=train_columns)
    timings["load_train_may_seconds"] = time.perf_counter() - start
    log(f"Train rows={len(train):,}; May rows={len(may):,}; RAM={memory_gb():.1f} GB")

    start = time.perf_counter()
    transformer, logistic_models = train_logistic_models(train, output_dir / "models", args.smoke)
    timings["train_logistic_seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    lightgbm_models, best_iterations, feature_importance = train_lightgbm_models(
        train, output_dir / "models", args.smoke, args.threads
    )
    timings["train_lightgbm_seconds"] = time.perf_counter() - start
    pd.DataFrame(feature_importance).to_csv(
        output_dir / "data" / "lightgbm_feature_importance.csv", index=False, encoding="utf-8-sig"
    )

    start = time.perf_counter()
    calibrators, thresholds, may_performance = calibrate_and_select_thresholds(
        may, transformer, logistic_models, lightgbm_models, output_dir
    )
    timings["may_calibration_threshold_seconds"] = time.perf_counter() - start

    freeze_manifest = {
        "frozen_at": now_text(),
        "smoke": args.smoke,
        "prediction_horizon_minutes": 30,
        "training_months": [1, 2, 3, 4],
        "calibration_period": "2026-05-01 through 2026-05-20",
        "threshold_period": "2026-05-21 through 2026-05-31; target time in peak hours",
        "test_month": None if args.smoke else 6,
        "peak_hours": sorted(PEAK_HOURS),
        "best_iterations": best_iterations,
        "thresholds": thresholds,
        "calibrators": {
            event: {method: calibrator.as_dict() for method, calibrator in methods.items()}
            for event, methods in calibrators.items()
        },
        "logistic_features": LOGISTIC_FEATURES,
        "lightgbm_features": LIGHTGBM_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "lightgbm_base_parameters": {
            key: value
            for key, value in lgb_parameters(1, args.threads).items()
            if key != "n_estimators"
        },
        "lightgbm_final_tree_count_by_event": best_iterations,
        "segment_reporting_definition": {
            "used_as_model_feature": False,
            "source_file": str(SEGMENT_FILE),
            "source_sha256": file_sha256(SEGMENT_FILE) if SEGMENT_FILE.exists() else None,
            "labels": SEGMENT_LABELS,
            "note": "Reporting subgroup only; never used for fitting or probability scoring.",
        },
        "pipeline_sha256": file_sha256(Path(__file__)),
        "pipeline_snapshot_sha256": file_sha256(source_snapshot),
        "model_sha256": {
            path.name: file_sha256(path)
            for path in sorted((output_dir / "models").glob("*"))
            if path.is_file()
        },
        "test_was_not_used_for_parameter_selection": True,
    }
    write_json(output_dir / "freeze_manifest_before_test.json", freeze_manifest)
    log("Model, calibration, and thresholds frozen before holdout evaluation")

    if args.smoke:
        holdout = may.loc[may["target_datetime"].ge(pd.Timestamp("2026-05-21"))].copy()
        holdout_label = "SMOKE pseudo-holdout (May 21 only)"
    else:
        del train, may
        gc.collect()
        start = time.perf_counter()
        june_summaries, _ = prepare_feature_cache(
            cache_dir,
            smoke=False,
            force=args.force_features,
            months=[6],
        )
        timings["prepare_month_6_seconds"] = time.perf_counter() - start
        summaries.extend(june_summaries)
        pd.DataFrame(summaries).to_csv(output_dir / "data" / "dataset_summary.csv", index=False, encoding="utf-8-sig")
        holdout = load_months(cache_dir, [6], columns=train_columns)
        station_lookup = augment_station_lookup_after_holdout(station_lookup, 6)
        station_lookup.to_csv(output_dir / "data" / "station_lookup.csv", index=False, encoding="utf-8-sig")
        holdout_label = "June sealed holdout"
        log(f"June rows={len(holdout):,}; RAM={memory_gb():.1f} GB")

    start = time.perf_counter()
    results = evaluate_holdout(
        holdout,
        transformer,
        logistic_models,
        lightgbm_models,
        calibrators,
        thresholds,
        station_lookup,
        output_dir,
        holdout_label,
    )
    timings["holdout_evaluation_seconds"] = time.perf_counter() - start
    timings["total_seconds"] = time.perf_counter() - overall_start
    write_json(output_dir / "timings.json", timings)
    write_json(output_dir / "run_complete.json", {
        "completed_at": now_text(),
        "smoke": args.smoke,
        "output_dir": output_dir,
        "cache_dir": cache_dir,
        "timings": timings,
        "performance_rows": len(results["performance"]),
        "status": "complete",
    })
    log(f"Pipeline complete in {timings['total_seconds'] / 60:.1f} minutes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YouBike 30-minute imbalance model v1")
    parser.add_argument("--smoke", action="store_true", help="Run a small end-to-end functional test without June")
    parser.add_argument("--force-features", action="store_true", help="Rebuild feature parquet files")
    parser.add_argument("--threads", type=int, default=16, help="LightGBM CPU threads")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
