from __future__ import annotations

"""Dynamic one-step YouBike empty-state persistence experiment.

Every valid current empty snapshot, including onset and ongoing snapshots, is a
decision row.  The target is whether the same station is still exactly empty at
t+30.  Jan-Mar train the models; April calibrates/selects/gates; the already
opened May set is generated/read only after the policy freeze.  June is never
read.
"""

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from run_current_red_persistence_experiment import (
    CATEGORICAL_FEATURES,
    LGB_PARAMS,
    MODEL_FEATURES,
    PlattCalibrator,
    binary_counts,
    build_metadata_maps,
    build_month,
    wilson_interval,
)


ROOT = Path(__file__).resolve().parent
WORK_DIR = ROOT / "tmp" / "youbike_dynamic_red_persistence_20260904_v2"
OUTPUT_DIR = ROOT / "outputs" / "youbike_dynamic_red_persistence_20260904_v2"
EVENT = "empty"
K = 10
SEED = 20260904

APRIL_START = pd.Timestamp("2026-04-01")
CALIBRATION_END = pd.Timestamp("2026-04-15")
SELECTION_END = pd.Timestamp("2026-04-21")
APRIL_END = pd.Timestamp("2026-05-01")
MAY_END = pd.Timestamp("2026-06-01")

RECALL_FLOOR = 0.20
PRECISION_TARGET = 0.70
MIN_TP = 20
MIN_ALERTS = 50
HISTORY_PRIOR = 20.0

DURATION_FEATURES = [
    "red_duration_capped_6",
    "red_duration_log1p_capped",
    "duration_90plus",
]
FULL_FEATURES = list(dict.fromkeys(MODEL_FEATURES + DURATION_FEATURES))
NO_STATION_FEATURES = [
    feature for feature in FULL_FEATURES if feature not in {"station_id", "longitude", "latitude"}
]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    def default(item: Any) -> Any:
        if isinstance(item, (np.integer,)):
            return int(item)
        if isinstance(item, (np.floating,)):
            return None if not np.isfinite(item) else float(item)
        if isinstance(item, (np.bool_,)):
            return bool(item)
        if isinstance(item, (pd.Timestamp, Path)):
            return str(item)
        raise TypeError(type(item))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=default), encoding="utf-8")


def cache_path(month: int) -> Path:
    return WORK_DIR / f"dynamic_red_{EVENT}_2026_{month:02d}.parquet"


def ensure_cache(month: int) -> None:
    if cache_path(month).is_file():
        return
    district_map, segment_map, _ = build_metadata_maps()
    build_month(
        month,
        district_map,
        segment_map,
        False,
        dynamic_output_dir=WORK_DIR,
    )
    if not cache_path(month).is_file():
        raise RuntimeError(f"Dynamic cache was not created for month {month}")


def load_month(month: int) -> pd.DataFrame:
    frame = pd.read_parquet(cache_path(month))
    for column in ("datetime", "target_datetime", "episode_start_datetime"):
        frame[column] = pd.to_datetime(frame[column])
    frame["red_duration_capped_6"] = frame["red_duration_steps"].clip(upper=6).astype("int8")
    frame["red_duration_log1p_capped"] = np.log1p(frame["red_duration_capped_6"]).astype("float32")
    frame["duration_90plus"] = frame["red_duration_steps"].ge(3).astype("int8")
    return frame


def primary_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["duration_unknown"].eq(0)].copy()


def episode_window(frame: pd.DataFrame, start: pd.Timestamp, stop: pd.Timestamp) -> pd.DataFrame:
    return frame.loc[
        frame["decision_is_peak"].eq(1)
        & frame["duration_unknown"].eq(0)
        & frame["episode_start_datetime"].ge(start)
        & frame["episode_start_datetime"].lt(stop)
        & frame["datetime"].ge(start)
        & frame["target_datetime"].lt(stop)
    ].copy()


def category_schema(frame: pd.DataFrame, features: list[str]) -> dict[str, list[int]]:
    return {
        column: sorted(int(value) for value in frame[column].dropna().unique())
        for column in CATEGORICAL_FEATURES
        if column in features
    }


def model_matrix(
    frame: pd.DataFrame,
    features: list[str],
    schema: dict[str, list[int]],
) -> pd.DataFrame:
    matrix = frame[features].copy()
    for column, categories in schema.items():
        numeric = pd.to_numeric(matrix[column], errors="coerce")
        numeric = numeric.where(numeric.isin(categories))
        matrix[column] = pd.Categorical(numeric, categories=categories)
    for column in set(features) - set(schema):
        matrix[column] = pd.to_numeric(matrix[column], errors="coerce").astype("float32")
    return matrix


def model_params(tree_count: int | None = None) -> dict[str, Any]:
    params = dict(LGB_PARAMS)
    params["random_state"] = SEED
    params["n_jobs"] = 1
    if tree_count is not None:
        params["n_estimators"] = int(tree_count)
    return params


def rolling_cv(
    monthly: dict[int, pd.DataFrame],
    features: list[str],
    model_name: str,
) -> tuple[pd.DataFrame, int]:
    rows: list[dict[str, Any]] = []
    trees: list[int] = []
    for fold, train_months, validation_month in (
        ("jan_to_feb", (1,), 2),
        ("jan_feb_to_mar", (1, 2), 3),
    ):
        train = primary_rows(pd.concat([monthly[m] for m in train_months], ignore_index=True))
        validation = primary_rows(monthly[validation_month])
        validation = validation.loc[validation["decision_is_peak"].eq(1)].copy()
        schema = category_schema(train, features)
        train_x = model_matrix(train, features, schema)
        validation_x = model_matrix(validation, features, schema)
        model = lgb.LGBMClassifier(**model_params())
        model.fit(
            train_x,
            train["y_same_30"].to_numpy(dtype=np.int8),
            categorical_feature=[column for column in CATEGORICAL_FEATURES if column in features],
            eval_set=[(validation_x, validation["y_same_30"].to_numpy(dtype=np.int8))],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(80, verbose=False)],
        )
        tree_count = int(model.best_iteration_ or LGB_PARAMS["n_estimators"])
        trees.append(tree_count)
        score = model.predict_proba(validation_x)[:, 1]
        rows.append(
            {
                "model": model_name,
                "fold": fold,
                "train_months": ",".join(map(str, train_months)),
                "validation_month": validation_month,
                "train_rows": len(train),
                "validation_peak_rows": len(validation),
                "validation_positives": int(validation["y_same_30"].sum()),
                "validation_prevalence": float(validation["y_same_30"].mean()),
                "average_precision": float(average_precision_score(validation["y_same_30"], score)),
                "tree_count": tree_count,
            }
        )
    final_trees = max(10, int(round(float(np.median(trees)))))
    return pd.DataFrame(rows), final_trees


def train_final(
    train: pd.DataFrame,
    features: list[str],
    tree_count: int,
) -> tuple[lgb.Booster, dict[str, list[int]]]:
    train = primary_rows(train)
    schema = category_schema(train, features)
    matrix = model_matrix(train, features, schema)
    model = lgb.LGBMClassifier(**model_params(tree_count))
    model.fit(
        matrix,
        train["y_same_30"].to_numpy(dtype=np.int8),
        categorical_feature=[column for column in CATEGORICAL_FEATURES if column in features],
    )
    return model.booster_, schema


def add_raw_score(
    frame: pd.DataFrame,
    booster: lgb.Booster,
    features: list[str],
    schema: dict[str, list[int]],
    column: str,
) -> pd.DataFrame:
    result = frame.copy()
    result[column] = booster.predict(
        model_matrix(result, features, schema), raw_score=True
    ).astype("float32")
    return result


def history_table(train: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    history_train = primary_rows(train)
    history_train = history_train.loc[history_train["decision_is_peak"].eq(1)].copy()
    global_rate = float(history_train["y_same_30"].mean())
    keys = ["station_id", "target_hour", "target_is_weekend", "red_duration_capped_6"]
    table = (
        history_train.groupby(keys, observed=True, sort=False)["y_same_30"]
        .agg(history_successes="sum", history_n="size")
        .reset_index()
    )
    table["p_history"] = (
        table["history_successes"] + HISTORY_PRIOR * global_rate
    ) / (table["history_n"] + HISTORY_PRIOR)
    return table, global_rate


def attach_history(frame: pd.DataFrame, table: pd.DataFrame, global_rate: float) -> pd.DataFrame:
    keys = ["station_id", "target_hour", "target_is_weekend", "red_duration_capped_6"]
    result = frame.merge(table, on=keys, how="left", validate="many_to_one")
    result["history_n"] = result["history_n"].fillna(0).astype("int32")
    result["history_successes"] = result["history_successes"].fillna(0).astype("int32")
    result["p_history"] = result["p_history"].fillna(global_rate).astype("float32")
    return result


def calibrate_and_score(
    april: pd.DataFrame,
    raw_column: str,
    output_column: str,
) -> tuple[pd.DataFrame, PlattCalibrator]:
    calibration = episode_window(april, APRIL_START, CALIBRATION_END)
    calibrator = PlattCalibrator.fit(
        calibration[raw_column].to_numpy(dtype=np.float32),
        calibration["y_same_30"].to_numpy(dtype=np.int8),
    )
    result = april.copy()
    result[output_column] = calibrator.predict(result[raw_column].to_numpy(dtype=np.float32))
    return result, calibrator


def row_metrics(population: pd.DataFrame, selected: pd.DataFrame, split: str, policy: str) -> dict[str, Any]:
    tp = int(selected["y_same_30"].sum())
    alerts = len(selected)
    positives = int(population["y_same_30"].sum())
    precision = tp / alerts if alerts else 0.0
    recall = tp / positives if positives else 0.0
    low, high = wilson_interval(tp, alerts)
    return {
        "split": split,
        "policy": policy,
        "population_rows": len(population),
        "population_positives": positives,
        "prevalence": positives / len(population) if len(population) else math.nan,
        "alerts": alerts,
        "tp": tp,
        "fp": alerts - tp,
        "fn": positives - tp,
        "precision": precision,
        "recall": recall,
        "precision_ci95_low": low,
        "precision_ci95_high": high,
        "alerts_per_day": alerts / max(1, population["datetime"].dt.normalize().nunique()),
        "alerts_per_timepoint": alerts / max(1, population["datetime"].nunique()),
    }


def first_alert_metrics(
    population: pd.DataFrame,
    selected: pd.DataFrame,
    split: str,
    policy: str,
) -> dict[str, Any]:
    episode_keys = ["station_id", "episode_start_datetime"]
    first = (
        selected.sort_values([*episode_keys, "datetime"], kind="mergesort")
        .groupby(episode_keys, observed=True, sort=False)
        .head(1)
    )
    positive_episodes = int(
        population.groupby(episode_keys, observed=True)["y_same_30"].max().sum()
    )
    tp = int(first["y_same_30"].sum())
    alerts = len(first)
    return {
        "split": split,
        "policy": policy,
        "alerted_episodes": alerts,
        "first_alert_tp": tp,
        "first_alert_fp": alerts - tp,
        "first_alert_precision": tp / alerts if alerts else 0.0,
        "positive_episode_opportunities": positive_episodes,
        "positive_episode_coverage": tp / positive_episodes if positive_episodes else 0.0,
        "median_first_alert_duration_minutes": (
            float(30 * first["red_duration_steps"].median()) if alerts else math.nan
        ),
    }


def threshold_grid(
    population: pd.DataFrame,
    score_column: str,
) -> pd.DataFrame:
    score = population[score_column].to_numpy(dtype=np.float64)
    candidates = np.unique(np.quantile(score[np.isfinite(score)], np.linspace(0, 1, 501)))
    rows = []
    for threshold in candidates:
        selected = population.loc[population[score_column].ge(threshold)]
        metric = row_metrics(population, selected, "april_15_20_selection", score_column)
        metric["threshold"] = float(threshold)
        rows.append(metric)
    return pd.DataFrame(rows)


def choose_thresholds(grid: pd.DataFrame) -> dict[str, dict[str, Any]]:
    evidence = grid.loc[(grid["tp"] >= MIN_TP) & (grid["alerts"] >= MIN_ALERTS)].copy()
    precision_viable = evidence.loc[evidence["recall"].ge(RECALL_FLOOR)]
    if precision_viable.empty:
        precision_viable = evidence
    precision_row = precision_viable.sort_values(
        ["precision", "recall", "alerts", "threshold"],
        ascending=[False, False, True, False],
        kind="mergesort",
    ).iloc[0]
    p70 = evidence.loc[evidence["precision"].ge(PRECISION_TARGET)]
    p70_feasible = not p70.empty
    if p70_feasible:
        p70_row = p70.sort_values(
            ["recall", "precision", "alerts", "threshold"],
            ascending=[False, False, True, True],
            kind="mergesort",
        ).iloc[0]
    else:
        # Do not silently turn an infeasible 70%-precision policy into the
        # all-alert rule.  Preserve the best evidenced precision and flag that
        # the target was not reached.
        p70_row = evidence.sort_values(
            ["precision", "recall", "alerts", "threshold"],
            ascending=[False, False, True, False],
            kind="mergesort",
        ).iloc[0]
    return {
        "precision_first": precision_row.to_dict(),
        "p70_recall": {**p70_row.to_dict(), "feasible": p70_feasible},
    }


def topk(frame: pd.DataFrame, score_column: str) -> pd.DataFrame:
    ordered = frame.sort_values(
        ["datetime", score_column, "station_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    return ordered.groupby("datetime", observed=True, sort=False).head(K).copy()


def selected_by_threshold(frame: pd.DataFrame, score_column: str, threshold: float) -> pd.DataFrame:
    return frame.loc[frame[score_column].ge(threshold)].copy()


def duration_policy(frame: pd.DataFrame, minimum_steps: int) -> pd.DataFrame:
    return frame.loc[frame["red_duration_steps"].ge(minimum_steps)].copy()


def age_breakdown(selected: pd.DataFrame, split: str, policy: str) -> list[dict[str, Any]]:
    bucket = pd.cut(
        selected["red_duration_steps"],
        bins=[-1, 0, 1, 2, np.inf],
        labels=["onset_0m", "ongoing_30m", "ongoing_60m", "ongoing_90m_plus"],
    )
    rows = []
    for name in bucket.cat.categories:
        part = selected.loc[bucket.eq(name)]
        rows.append(
            {
                "split": split,
                "policy": policy,
                "age_bucket": str(name),
                "alerts": len(part),
                "tp": int(part["y_same_30"].sum()),
                "precision": float(part["y_same_30"].mean()) if len(part) else math.nan,
                "alert_share": len(part) / len(selected) if len(selected) else 0.0,
            }
        )
    return rows


def station_concentration(selected: pd.DataFrame) -> float:
    if selected.empty:
        return math.nan
    return float(selected["station_id"].value_counts().head(10).sum() / len(selected))


def evaluate_policies(
    population: pd.DataFrame,
    split: str,
    thresholds: dict[str, dict[str, dict[str, Any]]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    policies: dict[str, pd.DataFrame] = {
        "all_current_red": population,
        "duration_at_least_30m": duration_policy(population, 1),
        "duration_at_least_60m": duration_policy(population, 2),
        "duration_at_least_90m": duration_policy(population, 3),
        "duration_top10": topk(population, "red_duration_steps"),
    }
    score_columns = {
        "lgbm_full": "p_lgbm_full",
        "lgbm_no_station": "p_lgbm_no_station",
        "history": "p_history",
    }
    for model_name, score_column in score_columns.items():
        for objective in ("precision_first", "p70_recall"):
            threshold = float(thresholds[model_name][objective]["threshold"])
            selected = selected_by_threshold(population, score_column, threshold)
            name = f"{model_name}_{objective}"
            policies[name] = selected
            if model_name != "history":
                policies[f"{name}_k10"] = topk(selected, score_column)

    metric_rows = []
    episode_rows = []
    age_rows = []
    for policy, selected in policies.items():
        row = row_metrics(population, selected, split, policy)
        row["top10_station_alert_share"] = station_concentration(selected)
        metric_rows.append(row)
        episode_rows.append(first_alert_metrics(population, selected, split, policy))
        age_rows.extend(age_breakdown(selected, split, policy))
    return pd.DataFrame(metric_rows), pd.DataFrame(episode_rows), pd.DataFrame(age_rows)


def run() -> None:
    for directory in (OUTPUT_DIR, OUTPUT_DIR / "config", OUTPUT_DIR / "metrics", OUTPUT_DIR / "models"):
        directory.mkdir(parents=True, exist_ok=True)
    for month in (1, 2, 3, 4):
        ensure_cache(month)

    log("Loading Jan-Mar dynamic decision rows")
    monthly = {month: load_month(month) for month in (1, 2, 3)}
    training = pd.concat([monthly[1], monthly[2], monthly[3]], ignore_index=True)

    cv_frames = []
    tree_counts: dict[str, int] = {}
    model_specs = {
        "lgbm_full": FULL_FEATURES,
        "lgbm_no_station": NO_STATION_FEATURES,
    }
    for model_name, features in model_specs.items():
        cv, trees = rolling_cv(monthly, features, model_name)
        cv_frames.append(cv)
        tree_counts[model_name] = trees
    pd.concat(cv_frames, ignore_index=True).to_csv(OUTPUT_DIR / "metrics" / "rolling_cv.csv", index=False)

    boosters: dict[str, lgb.Booster] = {}
    schemas: dict[str, dict[str, list[int]]] = {}
    for model_name, features in model_specs.items():
        booster, schema = train_final(training, features, tree_counts[model_name])
        boosters[model_name] = booster
        schemas[model_name] = schema
        model_path = OUTPUT_DIR / "models" / f"{model_name}.txt"
        model_path.write_text(booster.model_to_string(), encoding="utf-8")

    history, history_global = history_table(training)
    history.to_parquet(OUTPUT_DIR / "models" / "dynamic_history.parquet", index=False)

    log("Scoring and freezing with April only")
    april = load_month(4)
    for model_name, features in model_specs.items():
        april = add_raw_score(
            april,
            boosters[model_name],
            features,
            schemas[model_name],
            f"raw_{model_name}",
        )
    april = attach_history(april, history, history_global)
    calibrators: dict[str, PlattCalibrator] = {}
    for model_name in model_specs:
        april, calibrator = calibrate_and_score(april, f"raw_{model_name}", f"p_{model_name}")
        calibrators[model_name] = calibrator

    selection = episode_window(april, CALIBRATION_END, SELECTION_END)
    gate = episode_window(april, SELECTION_END, APRIL_END)
    thresholds: dict[str, dict[str, dict[str, Any]]] = {}
    grids = []
    for model_name in model_specs:
        grid = threshold_grid(selection, f"p_{model_name}")
        grid["model"] = model_name
        grids.append(grid)
        thresholds[model_name] = choose_thresholds(grid)
    history_grid = threshold_grid(selection, "p_history")
    history_grid["model"] = "history"
    grids.append(history_grid)
    thresholds["history"] = choose_thresholds(history_grid)
    pd.concat(grids, ignore_index=True).to_csv(OUTPUT_DIR / "metrics" / "april_threshold_grids.csv", index=False)

    selection_metrics, selection_episode, selection_age = evaluate_policies(
        selection, "april_15_20_selection", thresholds
    )
    gate_metrics, gate_episode, gate_age = evaluate_policies(gate, "april_21_30_gate", thresholds)
    april_metrics = pd.concat([selection_metrics, gate_metrics], ignore_index=True)
    april_episode = pd.concat([selection_episode, gate_episode], ignore_index=True)
    april_age = pd.concat([selection_age, gate_age], ignore_index=True)
    april_metrics.to_csv(OUTPUT_DIR / "metrics" / "april_policy_metrics.csv", index=False)
    april_episode.to_csv(OUTPUT_DIR / "metrics" / "april_first_alert_metrics.csv", index=False)
    april_age.to_csv(OUTPUT_DIR / "metrics" / "april_alert_age_breakdown.csv", index=False)

    freeze_path = OUTPUT_DIR / "config" / "final_policy_freeze_before_may.json"
    write_json(
        freeze_path,
        {
            "created_at": pd.Timestamp.now(),
            "target": "every valid current exact-empty snapshot predicts exact-empty at t+30",
            "training_months": [1, 2, 3],
            "training_rows_known_duration": int(primary_rows(training).shape[0]),
            "training_positives": int(primary_rows(training)["y_same_30"].sum()),
            "cache_rule": "t+30 valid, operational, same capacity only; t+60 availability is not required",
            "episode_rule": "onset and every ongoing snapshot are decisions; unknown-duration rows excluded from primary",
            "features": model_specs,
            "tree_counts": tree_counts,
            "calibrators": {
                name: {"slope": calibrator.slope, "intercept": calibrator.intercept}
                for name, calibrator in calibrators.items()
            },
            "history_global_rate": history_global,
            "thresholds": thresholds,
            "threshold_selection": (
                "Apr15-20 episodes only; precision-first maximizes precision with recall>=20%, "
                "p70-recall maximizes recall subject to precision>=70%; >=20 TP and >=50 alerts"
            ),
            "gate": "Apr21-30 episodes only; no retuning",
            "k10": "optional action cap after frozen threshold; never fills low-risk slots",
            "may_read_before_freeze_in_this_run": False,
            "may_already_opened_in_prior_experiments": True,
            "june_opened": False,
            "code_sha256": sha256(Path(__file__).resolve()),
            "cache_builder_sha256": sha256(ROOT / "run_current_red_persistence_experiment.py"),
        },
    )

    log("Freeze written; now generating/opening May dynamic cache")
    ensure_cache(5)
    may = load_month(5)
    for model_name, features in model_specs.items():
        may = add_raw_score(
            may,
            boosters[model_name],
            features,
            schemas[model_name],
            f"raw_{model_name}",
        )
        may[f"p_{model_name}"] = calibrators[model_name].predict(
            may[f"raw_{model_name}"].to_numpy(dtype=np.float32)
        )
    may = attach_history(may, history, history_global)
    may_population = episode_window(may, pd.Timestamp("2026-05-01"), MAY_END)
    may_metrics, may_episode, may_age = evaluate_policies(
        may_population, "may_exploratory", thresholds
    )
    may_metrics.to_csv(OUTPUT_DIR / "metrics" / "may_policy_metrics.csv", index=False)
    may_episode.to_csv(OUTPUT_DIR / "metrics" / "may_first_alert_metrics.csv", index=False)
    may_age.to_csv(OUTPUT_DIR / "metrics" / "may_alert_age_breakdown.csv", index=False)

    for split_name, frame in (("april_gate", gate), ("may", may_population)):
        score_rows = []
        for model_name in model_specs:
            score_rows.append(
                {
                    "split": split_name,
                    "model": model_name,
                    "rows": len(frame),
                    "positives": int(frame["y_same_30"].sum()),
                    "prevalence": float(frame["y_same_30"].mean()),
                    "average_precision": float(
                        average_precision_score(frame["y_same_30"], frame[f"p_{model_name}"])
                    ),
                }
            )
        pd.DataFrame(score_rows).to_csv(OUTPUT_DIR / "metrics" / f"{split_name}_ranking_metrics.csv", index=False)

    policy_labels = {
        "all_current_red": "全部當下缺車都猜持續",
        "duration_at_least_30m": "已持續至少30分鐘",
        "duration_at_least_60m": "已持續至少60分鐘",
        "duration_at_least_90m": "已持續至少90分鐘",
        "duration_top10": "每時點單純挑持續最久10站",
        "lgbm_full_precision_first": "動態LightGBM高Precision門檻",
        "lgbm_full_precision_first_k10": "動態LightGBM高Precision＋K10",
        "lgbm_full_p70_recall": "動態LightGBM 70%Precision優先Recall",
        "lgbm_full_p70_recall_k10": "動態LightGBM 70%＋K10",
        "lgbm_no_station_precision_first": "移除站點ID高Precision",
        "history_precision_first": "站點×時段×持續時間歷史",
    }
    report_policies = list(policy_labels)
    lines = [
        "# YouBike動態30分鐘缺車持續模型",
        "",
        "- 對每個當下仍為精確0車、且本次Episode起點可知的快照重新預測t+30，不只Episode起點。",
        "- 主指標逐決策列計算，沒有Episode any-hit；另報每Episode第一次警示。",
        "- 1～3月訓練，4/1～4/14校正，4/15～4/20選門檻，4/21～4/30驗收；五月是已開封追加探索，六月未讀。",
        "",
        "## 五月逐決策結果",
        "",
        "|規則|Precision|Recall|警示數|Top10站警示佔比|",
        "|---|---:|---:|---:|---:|",
    ]
    for policy in report_policies:
        match = may_metrics.loc[may_metrics["policy"].eq(policy)]
        if match.empty:
            continue
        row = match.iloc[0]
        lines.append(
            f"|{policy_labels[policy]}|{row.precision:.2%}|{row.recall:.2%}|"
            f"{int(row.alerts):,}|{row.top10_station_alert_share:.2%}|"
        )
    lines.extend(
        [
            "",
            "## 五月每Episode第一次警示",
            "",
            "|規則|第一次警示Precision|警示Episode數|正向Episode覆蓋|中位警示時點|",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for policy in report_policies:
        match = may_episode.loc[may_episode["policy"].eq(policy)]
        if match.empty:
            continue
        row = match.iloc[0]
        lines.append(
            f"|{policy_labels[policy]}|{row.first_alert_precision:.2%}|"
            f"{int(row.alerted_episodes):,}|{row.positive_episode_coverage:.2%}|"
            f"{row.median_first_alert_duration_minutes:.0f}分鐘|"
        )
    lines.extend(
        [
            "",
            "## 關鍵解讀",
            "",
            "動態模型的逐次Precision可能很高，但同一個長時間Episode會每30分鐘再被計分。因此必須連同第一次警示Precision、警示發生時間與站點集中度一起解讀，不能將逐次Precision說成獨立Episode的成功率。",
            "",
            "K10只有在門檻後仍超過10站時才會縮減清單；嚴格70%門檻平均每時點不到10站，因此K10幾乎不會再改善Precision。",
            "",
            "月初已在缺車且起點不可見的Episode已排除；六月完全未讀。",
        ]
    )
    report_path = OUTPUT_DIR / "RESULT_DYNAMIC_EMPTY_30_ZH.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    write_json(
        OUTPUT_DIR / "completion.json",
        {
            "status": "complete",
            "may_used_for_training_calibration_or_threshold_selection": False,
            "may_interpretation": "already-opened exploratory confirmation only",
            "june_opened": False,
            "freeze": {"path": freeze_path, "sha256": sha256(freeze_path)},
            "code_sha256": sha256(Path(__file__).resolve()),
        },
    )
    log(str(report_path.resolve()))


if __name__ == "__main__":
    run()
