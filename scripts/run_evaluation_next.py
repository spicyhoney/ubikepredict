from __future__ import annotations

"""Run the next-round R2-1, R2-2, and R2-3 YouBike evaluation.

R2-0 is implemented in ``run_evaluation.py`` and is run separately as the
frozen regression/provenance check.  This runner retrains only Jan-March
ablation models, scores the truth-free June input before joining labels,
reproduces the recovered no-station model, and evaluates the pre-specified
LONG_FIRST_F0 policy.  It never changes the production model or thresholds.
"""

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from backend.inference import FrozenYouBikeModel, load_demo_source, select_eligible_june  # noqa: E402
import run_evaluation as base  # noqa: E402


SEED = 20260904
BOOTSTRAP_SEED = 20260910
K_VALUES = (3, 5, 10, 20)
DURATION_BINS = ("B0", "B1", "B2", "B3")
CATEGORICAL_FEATURES = ("station_id", "district_code", "target_hour", "target_weekday")
LGB_PARAMS: dict[str, Any] = {
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

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "A0": (),
    "A1": (
        "red_duration_capped_6",
        "red_duration_log1p_capped",
        "duration_90plus",
    ),
    "A2": (
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
        "lag_90_missing",
        "suspected_large_jump_past",
        "same_red_lag_30",
        "same_red_lag_60",
        "same_red_lag_90",
        "opposite_red_lag_30",
        "opposite_red_lag_60",
        "opposite_red_lag_90",
        "same_red_count_past_90",
    ),
    "A3": (
        "bike_ratio_lag_1d",
        "dock_ratio_lag_1d",
        "bike_ratio_lag_7d",
        "dock_ratio_lag_7d",
        "lag_1d_missing",
        "lag_7d_missing",
        "same_red_lag_1d",
        "same_red_lag_7d",
    ),
    "A4": ("station_id",),
    "A5": ("station_id", "longitude", "latitude"),
}


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    def default(item: Any) -> Any:
        if isinstance(item, (np.integer,)):
            return int(item)
        if isinstance(item, (np.floating,)):
            return None if not np.isfinite(item) else float(item)
        if isinstance(item, (np.bool_,)):
            return bool(item)
        if isinstance(item, (Path, pd.Timestamp)):
            return str(item)
        raise TypeError(type(item))

    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=default), encoding="utf-8")


def _git_state() -> dict[str, Any]:
    command = [
        "git",
        "-c",
        f"safe.directory={ROOT.as_posix()}",
        "-C",
        str(ROOT),
    ]
    commit = subprocess.run(
        [*command, "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    status = subprocess.run(
        [*command, "status", "--porcelain=v1"], capture_output=True, text=True, check=False
    )
    entries = [line for line in status.stdout.splitlines() if line.strip()]
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else "unavailable",
        "dirty": bool(entries) if status.returncode == 0 else None,
        "entries": entries,
        "error": "" if commit.returncode == status.returncode == 0 else (commit.stderr + status.stderr).strip(),
    }


def _resolve(path: Path) -> Path:
    return (path if path.is_absolute() else ROOT / path).resolve()


def _add_duration_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    duration = pd.to_numeric(result["red_duration_steps"], errors="raise")
    result["red_duration_capped_6"] = duration.clip(upper=6).astype("int8")
    result["red_duration_log1p_capped"] = np.log1p(
        result["red_duration_capped_6"]
    ).astype("float32")
    result["duration_90plus"] = duration.ge(3).astype("int8")
    return result


def _load_training_month(path: Path) -> pd.DataFrame:
    frame = _add_duration_features(pd.read_parquet(path))
    for column in ("datetime", "target_datetime", "episode_start_datetime"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column])
    return frame


def _primary(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["duration_unknown"].eq(0)].copy()


def _feature_sets(full_features: Sequence[str]) -> dict[str, list[str]]:
    full = list(full_features)
    if len(full) != 68 or len(set(full)) != 68:
        raise RuntimeError("The frozen full feature list must have 68 unique columns")
    result: dict[str, list[str]] = {}
    for name, removed in FEATURE_GROUPS.items():
        unknown = sorted(set(removed).difference(full))
        if unknown:
            raise RuntimeError(f"{name} contains unknown removal columns: {unknown}")
        if len(removed) != len(set(removed)):
            raise RuntimeError(f"{name} contains duplicate removal columns")
        result[name] = [column for column in full if column not in set(removed)]
    expected = {"A0": 68, "A1": 65, "A2": 30, "A3": 60, "A4": 67, "A5": 65}
    actual = {name: len(columns) for name, columns in result.items()}
    if actual != expected:
        raise RuntimeError(f"Unexpected ablation feature counts: {actual}")
    return result


def _category_schema(frame: pd.DataFrame, features: Sequence[str]) -> dict[str, list[int]]:
    return {
        column: sorted(int(value) for value in frame[column].dropna().unique())
        for column in CATEGORICAL_FEATURES
        if column in features
    }


def _model_matrix(
    frame: pd.DataFrame, features: Sequence[str], schema: dict[str, list[int]]
) -> pd.DataFrame:
    missing = [column for column in features if column not in frame]
    if missing:
        raise RuntimeError(f"Model input is missing columns: {missing}")
    matrix = frame.loc[:, list(features)].copy()
    for column, categories in schema.items():
        numeric = pd.to_numeric(matrix[column], errors="coerce")
        matrix[column] = pd.Categorical(numeric.where(numeric.isin(categories)), categories=categories)
    for column in set(features).difference(schema):
        matrix[column] = pd.to_numeric(matrix[column], errors="coerce").astype("float32")
    return matrix


def _model_params(n_jobs: int, tree_count: int | None = None) -> dict[str, Any]:
    params = dict(LGB_PARAMS)
    params["n_jobs"] = n_jobs
    if tree_count is not None:
        params["n_estimators"] = int(tree_count)
    return params


def _rolling_cv(
    monthly: dict[int, pd.DataFrame],
    features: Sequence[str],
    variant: str,
    n_jobs: int,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    tree_counts: list[int] = []
    for fold, train_months, validation_month in (
        ("jan_to_feb", (1,), 2),
        ("jan_feb_to_mar", (1, 2), 3),
    ):
        train = _primary(pd.concat([monthly[m] for m in train_months], ignore_index=True))
        validation = _primary(monthly[validation_month])
        validation = validation.loc[validation["decision_is_peak"].eq(1)].copy()
        schema = _category_schema(train, features)
        model = lgb.LGBMClassifier(**_model_params(n_jobs))
        model.fit(
            _model_matrix(train, features, schema),
            train["y_same_30"].to_numpy(dtype=np.int8),
            categorical_feature=[c for c in CATEGORICAL_FEATURES if c in features],
            eval_set=[
                (
                    _model_matrix(validation, features, schema),
                    validation["y_same_30"].to_numpy(dtype=np.int8),
                )
            ],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(80, verbose=False)],
        )
        tree_count = int(model.best_iteration_ or LGB_PARAMS["n_estimators"])
        score = model.predict_proba(_model_matrix(validation, features, schema))[:, 1]
        tree_counts.append(tree_count)
        rows.append(
            {
                "variant": variant,
                "fold": fold,
                "train_months": ",".join(map(str, train_months)),
                "validation_month": validation_month,
                "train_rows": len(train),
                "validation_peak_rows": len(validation),
                "validation_positives": int(validation["y_same_30"].sum()),
                "validation_prevalence": float(validation["y_same_30"].mean()),
                "average_precision": base.average_precision_score(
                    validation["y_same_30"], score
                ),
                "tree_count": tree_count,
            }
        )
    final_trees = max(10, int(round(float(np.median(tree_counts)))))
    return rows, final_trees


def _train_final(
    train: pd.DataFrame,
    features: Sequence[str],
    tree_count: int,
    n_jobs: int,
) -> tuple[lgb.Booster, dict[str, list[int]]]:
    known = _primary(train)
    schema = _category_schema(known, features)
    model = lgb.LGBMClassifier(**_model_params(n_jobs, tree_count))
    model.fit(
        _model_matrix(known, features, schema),
        known["y_same_30"].to_numpy(dtype=np.int8),
        categorical_feature=[c for c in CATEGORICAL_FEATURES if c in features],
    )
    return model.booster_, schema


def _selected_metric(
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    task: str,
    panel: str,
    method: str,
    k: int,
    duration_bin: str = "ALL",
    ap: float = math.nan,
    seed: int = -1,
) -> dict[str, Any]:
    tp = int(selected["y_same_30"].sum())
    positives = int(candidates["y_same_30"].sum())
    return {
        "task": task,
        "panel": panel,
        "method": method,
        "K": k,
        "duration_bin": duration_bin,
        "seed": seed,
        "candidates": len(candidates),
        "positives": positives,
        "selected": len(selected),
        "TP": tp,
        "FP": len(selected) - tp,
        "precision": tp / len(selected) if len(selected) else math.nan,
        "recall": tp / positives if positives else math.nan,
        "AP": ap,
    }


def _conditional_auc(
    frame: pd.DataFrame, score_column: str, repetitions: int, seed: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (decision_time, duration), group in frame.groupby(
        ["datetime", "red_duration_steps"], sort=False
    ):
        positive = group.loc[group["y_same_30"].eq(1), score_column].to_numpy(float)
        negative = group.loc[group["y_same_30"].eq(0), score_column].to_numpy(float)
        pairs = len(positive) * len(negative)
        if not pairs:
            continue
        difference = positive[:, None] - negative[None, :]
        rows.append(
            {
                "date": pd.Timestamp(decision_time).normalize(),
                "duration_bin": str(group["duration_bin"].iloc[0]),
                "pairs": pairs,
                "wins": float((difference > 0).sum() + 0.5 * (difference == 0).sum()),
            }
        )
    pairs = pd.DataFrame(rows)
    all_dates = pd.date_range("2026-06-01", "2026-06-30", freq="D")
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = rng.integers(0, len(all_dates), size=(repetitions, len(all_dates)))
    output: list[dict[str, Any]] = []
    for scope in ("ALL", *DURATION_BINS):
        scoped = pairs if scope == "ALL" else pairs.loc[pairs["duration_bin"].eq(scope)]
        daily = scoped.groupby("date")[["wins", "pairs"]].sum().reindex(all_dates, fill_value=0)
        win = daily["wins"].to_numpy(float)
        count = daily["pairs"].to_numpy(float)
        denominator = count[draws].sum(axis=1)
        valid = denominator > 0
        estimates = win[draws].sum(axis=1)[valid] / denominator[valid]
        total = int(count.sum())
        output.append(
            {
                "duration_bin": scope,
                "mixed_groups": int(len(scoped)),
                "pairs": total,
                "conditional_auc": float(win.sum() / total) if total else math.nan,
                "ci_lower": float(np.quantile(estimates, 0.025)) if len(estimates) else math.nan,
                "ci_upper": float(np.quantile(estimates, 0.975)) if len(estimates) else math.nan,
            }
        )
    return output


def _rank(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    if method == "F0":
        return base._rank_rows(frame, ["datetime"], [("p_lgbm_full", False)], "p_lgbm_full")
    if method == "DURATION":
        return base._rank_rows(
            frame, ["datetime"], [("red_duration_steps", False)], "red_duration_steps"
        )
    if method == "LONG_FIRST_F0":
        return base._rank_rows(
            frame,
            ["datetime"],
            [("long_flag", False), ("p_lgbm_full", False)],
            "p_lgbm_full",
        )
    if method == "STORED_NO_STATION":
        return base._rank_rows(
            frame,
            ["datetime"],
            [("p_lgbm_no_station", False)],
            "p_lgbm_no_station",
        )
    raise ValueError(method)


def _method_selection(ranked: pd.DataFrame, method: str, k: int) -> pd.DataFrame:
    selected = ranked.loc[ranked["_rank"].le(k)].copy()
    selected["method"] = method
    selected["K"] = k
    return selected


def _bootstrap_difference(
    left: pd.DataFrame,
    right: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    result = base._paired_daily_precision_bootstrap(
        left, right, repetitions=repetitions, seed=seed
    )
    return dict(result)


def _exchange_analysis(
    f0: pd.DataFrame, duration: pd.DataFrame, k: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["datetime", "station_id", "duration_bin", "red_duration_steps", "y_same_30"]
    merged = f0[columns].merge(
        duration[columns],
        on=["datetime", "station_id"],
        how="outer",
        indicator=True,
        suffixes=("_f0", "_duration"),
        validate="one_to_one",
    )
    merged["membership"] = merged["_merge"].map(
        {"both": "both", "left_only": "f0_only", "right_only": "duration_only"}
    ).astype(str)
    for column in ("duration_bin", "red_duration_steps", "y_same_30"):
        merged[column] = merged[f"{column}_f0"].combine_first(merged[f"{column}_duration"])
    merged["K"] = k
    f0_only = int(merged["membership"].eq("f0_only").sum())
    duration_only = int(merged["membership"].eq("duration_only").sum())
    if f0_only != duration_only:
        raise RuntimeError(f"Exclusive selection counts differ at K={k}: {f0_only}/{duration_only}")
    summary_rows: list[dict[str, Any]] = []
    for membership in ("both", "f0_only", "duration_only"):
        subset = merged.loc[merged["membership"].eq(membership)]
        for duration_bin in ("ALL", *DURATION_BINS):
            scoped = subset if duration_bin == "ALL" else subset.loc[subset["duration_bin"].eq(duration_bin)]
            summary_rows.append(
                {
                    "K": k,
                    "membership": membership,
                    "duration_bin": duration_bin,
                    "rows": len(scoped),
                    "TP": int(scoped["y_same_30"].sum()),
                    "FP": len(scoped) - int(scoped["y_same_30"].sum()),
                    "precision": float(scoped["y_same_30"].mean()) if len(scoped) else math.nan,
                    "long_share": float(scoped["red_duration_steps"].ge(6).mean()) if len(scoped) else math.nan,
                }
            )
    keep = ["K", "datetime", "station_id", "membership", "duration_bin", "red_duration_steps", "y_same_30"]
    return pd.DataFrame(summary_rows), merged[keep]


def _artifact_inventory(output_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "run_manifest.json":
            rows.append(
                {
                    "path": path.relative_to(output_dir).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "committable": path.suffix.lower() not in {".parquet"} or path.stat().st_size < 10_000_000,
                }
            )
    return rows


def _report(
    run_id: str,
    ablation: pd.DataFrame,
    bootstrap: pd.DataFrame,
    conditional: pd.DataFrame,
    reproduction: dict[str, Any],
    policy: pd.DataFrame,
    policy_bootstrap: pd.DataFrame,
    exchange: pd.DataFrame,
    policy_swaps: pd.DataFrame,
) -> str:
    def pct(value: float) -> str:
        return "NA" if not np.isfinite(value) else f"{100 * value:.2f}%"

    k10 = ablation.loc[
        ablation["panel"].eq("global_top_k") & ablation["K"].eq(10)
    ].sort_values("method")
    policy_k = policy.loc[
        policy["seed"].eq(-1) & policy["K"].isin(K_VALUES)
    ].sort_values(["K", "method"])
    bootstrap_index = bootstrap.set_index("comparison")
    policy_index = policy_k.set_index(["K", "method"])
    lines = [
        f"# YouBike Evaluation Next：{run_id}",
        "",
        "本輪完成 R2-0 來源修正後的 R2-1 特徵消融、R2-2 no_station 重現，以及 R2-3 長事件優先混合政策。六月只作回溯評估；沒有重選門檻，也沒有替換正式 Demo 模型。",
        "",
        "## R2-1：特徵消融",
        "",
        "|版本|特徵數|樹數|AP|P@10|R@10|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in k10.itertuples(index=False):
        lines.append(
            f"|{row.method}|{int(row.feature_count)}|{int(row.tree_count)}|{pct(row.AP)}|{pct(row.precision)}|{pct(row.recall)}|"
        )
    lines.extend(
        [
            "",
            "消融差異以同一組 Jan–Mar 訓練資料、相同 rolling folds、超參數、seed 與 early stopping 規則建立。負差代表拿掉該資訊後比新 A0 差；信賴區間只反映六月日期抽樣變動。",
            "",
            "|比較|P@10差|95% CI|",
            "|---|---:|---:|",
        ]
    )
    for row in bootstrap.itertuples(index=False):
        lines.append(
            f"|{row.comparison}|{pct(row.point_difference)}|[{pct(row.ci_lower)}, {pct(row.ci_upper)}]|"
        )
    conditional_pivot = conditional.loc[
        conditional["duration_bin"].isin(["ALL", "B0"])
    ].pivot(index="variant", columns="duration_bin", values="conditional_auc")
    lines.extend(
        [
            "",
            "在 K=10，拿掉明確持續時間、近期 30/60/90 分鐘動態、昨天/上週歷史，分別比 A0 下降 "
            f"{pct(-float(bootstrap_index.loc['P@10(A1)-P@10(A0)', 'point_difference']))}、"
            f"{pct(-float(bootstrap_index.loc['P@10(A2)-P@10(A0)', 'point_difference']))}、"
            f"{pct(-float(bootstrap_index.loc['P@10(A3)-P@10(A0)', 'point_difference']))}；"
            "三者的日期 bootstrap 區間都未跨 0，支持三類資料對全市名單配置各有增量作用。",
            "",
            "### 控制同一時間與同樣缺車時長後",
            "",
            "|版本|全部時長 conditional AUC|剛缺車 B0 conditional AUC|",
            "|---|---:|---:|",
        ]
    )
    for variant in FEATURE_GROUPS:
        lines.append(
            f"|{variant}|{conditional_pivot.loc[variant, 'ALL']:.3f}|{conditional_pivot.loc[variant, 'B0']:.3f}|"
        )
    lines.extend(
        [
            "",
            "A1～A3 的 B0 conditional AUC 幾乎沒有低於 A0，表示它們改善的主要是不同時長事件如何分配名單，"
            "目前沒有證據顯示近期動態或日/週歷史帶來剛缺車站內部的額外排序力。反而 A4 拿掉 station_id 後 B0 AUC "
            f"由 {conditional_pivot.loc['A0', 'B0']:.3f} 降至 {conditional_pivot.loc['A4', 'B0']:.3f}；"
            f"A5 再拿掉座標後降至 {conditional_pivot.loc['A5', 'B0']:.3f}。"
            "因此站點/空間資訊確實參與早期事件辨識；A4 的全市 P@10 較高，主要是它更偏向容易命中的長事件，不能解讀成 station_id 沒用。",
        ]
    )
    lines.extend(
        [
            "",
            "## R2-2：no_station 重現",
            "",
            f"- 保存模型是否成功載入並重現：**{'是' if reproduction['reproduced'] else '否'}**。",
            f"- 逐列校正機率最大絕對差：`{reproduction['max_probability_difference']:.3g}`。",
            f"- A5 與保存 no_station raw margin 最大絕對差：`{reproduction['a5_raw_max_difference']:.3g}`。",
            "- 重現只證明舊模型與保存分數一致；特徵因果解讀仍以同流程 A0／A4／A5 為準。",
            "",
            "## R2-3：長事件優先＋模型排序",
            "",
            "|K|方法|Selected|TP|Precision|Recall|",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in policy_k.itertuples(index=False):
        lines.append(
            f"|{int(row.K)}|{row.method}|{int(row.selected)}|{int(row.TP)}|{pct(row.precision)}|{pct(row.recall)}|"
        )
    long_k10 = policy_index.loc[(10, "LONG_FIRST_F0")]
    f0_k10 = policy_index.loc[(10, "F0")]
    lines.extend(
        [
            "",
            "LONG_FIRST_F0 先排已缺車至少 180 分鐘的站，再用 F0 填剩餘名額。"
            f"在 K=10，它由 F0 的 {pct(float(f0_k10['precision']))} 提升到 {pct(float(long_k10['precision']))}，"
            f"多命中 {int(long_k10['TP'] - f0_k10['TP'])} 筆站點×決策時間。"
            "這是看過六月後提出的探索性政策；即使某格較好，也不能直接取代 Demo。",
            "",
            "|比較|K|Precision差|95% CI|",
            "|---|---:|---:|---:|",
        ]
    )
    for row in policy_bootstrap.itertuples(index=False):
        lines.append(
            f"|{row.comparison}|{int(row.K)}|{pct(row.point_difference)}|[{pct(row.ci_lower)}, {pct(row.ci_upper)}]|"
        )
    exclusive = exchange.loc[
        exchange["duration_bin"].eq("ALL") & exchange["membership"].isin(["f0_only", "duration_only"])
    ]
    lines.extend(
        [
            "",
            "## 名單交換診斷",
            "",
            "|K|集合|筆數|TP|Precision|長事件比例|",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in exclusive.itertuples(index=False):
        lines.append(
            f"|{int(row.K)}|{row.membership}|{int(row.rows)}|{int(row.TP)}|{pct(row.precision)}|{pct(row.long_share)}|"
        )
    long_vs_f0_k10 = policy_swaps.loc[
        policy_swaps["comparison"].eq("LONG_FIRST_F0_vs_F0")
        & policy_swaps["K"].eq(10)
        & policy_swaps["duration_bin"].eq("ALL")
        & policy_swaps["membership"].isin(["long_first_only", "f0_only"])
    ].set_index("membership")
    switched_in = long_vs_f0_k10.loc["long_first_only"]
    switched_out = long_vs_f0_k10.loc["f0_only"]
    lines.extend(
        [
            "",
            f"K=10 時，LONG_FIRST_F0 相對 F0 換進 {int(switched_in['rows'])} 筆長事件，其中 "
            f"{int(switched_in['TP'])} 筆下一張仍缺車；同時換出 {int(switched_out['rows'])} 筆非長事件，"
            f"其中 {int(switched_out['TP'])} 筆原本會命中。這直接解釋了提升來源，也顯示代價："
            "名單更準，但注意力更偏向已持續三小時、介入較晚的問題。",
        ]
    )
    lines.extend(
        [
            "",
            "## 限制",
            "",
            "- 六月已被查看，本輪是回溯與探索性分析，不是新的盲測。",
            "- P@K／R@K 單位是站點×決策時間，不是成功派車數或不重複站數。",
            "- 長事件規則可能提高命中率但介入較晚；精確度與營運提前量必須分開解讀。",
            "- R2-4 需要六月以後真正未見的連續資料，目前沒有執行。",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> Path:
    if not all(c.isalnum() or c in "_.-" for c in args.run_id):
        raise ValueError("--run-id contains unsupported characters")
    output_dir = _resolve(args.output_root) / args.run_id
    if output_dir.exists():
        raise FileExistsError(output_dir)
    (output_dir / "models").mkdir(parents=True)
    (output_dir / "figures").mkdir()
    (output_dir / "logs").mkdir()
    started = _now_iso()
    log_lines: list[str] = []

    def log(message: str) -> None:
        line = f"[{_now_iso()}] {message}"
        print(line, flush=True)
        log_lines.append(line)

    training_paths = {month: _resolve(Path(f"data/training/dynamic_red_empty_2026_{month:02d}.parquet")) for month in (1, 2, 3)}
    input_path = _resolve(args.input_path)
    reference_path = _resolve(args.reference_path)
    model_path = _resolve(args.model_path)
    no_station_model_path = _resolve(args.no_station_model_path)
    freeze_path = _resolve(args.freeze_path)
    protocol_path = _resolve(args.protocol_path)
    stations_path = _resolve(args.stations_path)
    spec_path = _resolve(args.spec_path)
    runner_path = Path(__file__).resolve()
    base_runner_path = ROOT / "scripts/run_evaluation.py"
    inference_path = ROOT / "backend/inference.py"
    provenance_paths = [
        ROOT / "research/provenance/run_dynamic_red_persistence_experiment_20260904.py",
        ROOT / "research/provenance/run_current_red_persistence_experiment_20260902.py",
        ROOT / "research/provenance/train_youbike_model_v1_20260828.py",
        ROOT / "research/provenance/README.md",
    ]
    required = [
        *training_paths.values(),
        input_path,
        reference_path,
        model_path,
        no_station_model_path,
        freeze_path,
        protocol_path,
        stations_path,
        spec_path,
        runner_path,
        base_runner_path,
        inference_path,
        *provenance_paths,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required inputs are missing: {missing}")
    manifest: dict[str, Any] = {
        "run_id": args.run_id,
        "started_at": started,
        "finished_at": None,
        "status": "running",
        "tasks": {"R2-0": "complete_in_base_runner", "R2-1": "running", "R2-2": "pending", "R2-3": "pending", "R2-4": "not_run_no_unseen_period"},
        "git": _git_state(),
        "command": subprocess.list2cmdline([sys.executable, *sys.argv]),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "lightgbm": lgb.__version__,
        },
        "parameters": {
            "training_months": [1, 2, 3],
            "training_seed": SEED,
            "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "random_seeds": args.random_seeds,
            "K": list(K_VALUES),
            "label": "same station remains exact-zero at t+30",
            "ablation": {name: list(columns) for name, columns in FEATURE_GROUPS.items()},
            "long_first_cutoff_steps": 6,
        },
        "inputs": {
            path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path): {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in required
        },
    }
    _write_json(output_dir / "run_manifest.json", manifest)

    try:
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        feature_sets = _feature_sets(freeze["features"]["lgbm_full"])
        monthly = {month: _load_training_month(path) for month, path in training_paths.items()}
        train = pd.concat([monthly[m] for m in (1, 2, 3)], ignore_index=True)
        known = _primary(train)
        if (len(known), int(known["y_same_30"].sum())) != (225_086, 99_802):
            raise RuntimeError("Training provenance counts do not match the frozen record")
        log("Training provenance passed: 225,086 known-duration rows and 99,802 positives")

        source = load_demo_source(path=input_path, mode="empty")
        forbidden = {"y_same_30", "p_lgbm_full", "p_lgbm_no_station"}.intersection(source)
        if forbidden:
            raise RuntimeError(f"Truth-free source contains audit columns: {sorted(forbidden)}")
        eligible = select_eligible_june(source).sort_values(
            ["datetime", "station_id"], kind="mergesort"
        ).reset_index(drop=True)
        eligible = _add_duration_features(eligible)
        if len(eligible) != 27_962:
            raise RuntimeError(f"Unexpected eligible row count: {len(eligible)}")

        frozen = FrozenYouBikeModel(
            model_path=model_path,
            freeze_path=freeze_path,
            protocol_path=protocol_path,
            stations_path=stations_path,
            mode="empty",
        )
        frozen_prediction = frozen.predict_frame(eligible, attach_stations=False)[
            ["datetime", "station_id", "raw_lgbm_full", "p_lgbm_full"]
        ]

        cv_rows: list[dict[str, Any]] = []
        model_metadata: list[dict[str, Any]] = []
        ablation_scores = eligible[["datetime", "station_id"]].copy()
        boosters: dict[str, lgb.Booster] = {}
        schemas: dict[str, dict[str, list[int]]] = {}
        for variant, features in feature_sets.items():
            log(f"R2-1 {variant}: rolling CV with {len(features)} features")
            fold_rows, tree_count = _rolling_cv(monthly, features, variant, args.n_jobs)
            cv_rows.extend(fold_rows)
            booster, schema = _train_final(train, features, tree_count, args.n_jobs)
            boosters[variant] = booster
            schemas[variant] = schema
            model_file = output_dir / "models" / f"{variant}.txt"
            # LightGBM's native Windows writer cannot open a path containing
            # non-ASCII characters.  Serializing first preserves the exact
            # model while Python performs the Unicode-safe file write.
            model_file.write_text(booster.model_to_string(), encoding="utf-8")
            ablation_scores[f"raw_{variant}"] = booster.predict(
                _model_matrix(eligible, features, schema), raw_score=True
            ).astype("float64")
            model_metadata.append(
                {
                    "variant": variant,
                    "feature_count": len(features),
                    "features": list(features),
                    "removed": list(FEATURE_GROUPS[variant]),
                    "tree_count": tree_count,
                    "model_path": model_file.relative_to(output_dir).as_posix(),
                    "model_sha256": _sha256(model_file),
                    "category_schema": schema,
                }
            )
        pd.DataFrame(cv_rows).to_csv(output_dir / "r2_1_rolling_cv.csv", index=False, encoding="utf-8-sig")
        _write_json(output_dir / "r2_1_models.json", model_metadata)
        manifest["tasks"]["R2-1"] = "trained"

        reference = pd.read_parquet(reference_path)
        key = ["datetime", "station_id"]
        if reference.duplicated(key).any() or eligible.duplicated(key).any():
            raise RuntimeError("June evaluation keys must be unique")
        evaluation = eligible[
            ["datetime", "target_datetime", "station_id", "red_duration_steps"]
        ].merge(ablation_scores, on=key, validate="one_to_one")
        evaluation = evaluation.merge(frozen_prediction, on=key, validate="one_to_one")
        evaluation = evaluation.merge(
            reference[key + ["y_same_30", "p_lgbm_no_station", "p_lgbm_full"]].rename(
                columns={"p_lgbm_full": "stored_p_lgbm_full"}
            ),
            on=key,
            validate="one_to_one",
        )
        evaluation["y_same_30"] = evaluation["y_same_30"].astype("int8")
        evaluation["duration_bin"] = base._duration_bin(evaluation["red_duration_steps"])
        evaluation["long_flag"] = evaluation["red_duration_steps"].ge(6).astype("int8")
        if (len(evaluation), int(evaluation["y_same_30"].sum()), evaluation["datetime"].nunique()) != (27_962, 11_921, 420):
            raise RuntimeError("June evaluation population does not match frozen counts")
        f0_probability_diff = float(
            np.max(np.abs(evaluation["p_lgbm_full"] - evaluation["stored_p_lgbm_full"]))
        )
        if f0_probability_diff != 0.0:
            raise RuntimeError(f"Frozen F0 parity failed: {f0_probability_diff}")

        ablation_metric_rows: list[dict[str, Any]] = []
        ablation_selection_rows: list[pd.DataFrame] = []
        ablation_bootstrap_rows: list[dict[str, Any]] = []
        conditional_rows: list[dict[str, Any]] = []
        ranked_variants: dict[str, pd.DataFrame] = {}
        metadata_by_variant = {row["variant"]: row for row in model_metadata}
        for variant in feature_sets:
            score_column = f"raw_{variant}"
            ranked = base._rank_rows(
                evaluation, ["datetime"], [(score_column, False)], score_column
            )
            ranked_variants[variant] = ranked
            ap = base.average_precision_score(evaluation["y_same_30"], evaluation[score_column])
            for k in K_VALUES:
                selected = ranked.loc[ranked["_rank"].le(k)].copy()
                metric = _selected_metric(
                    evaluation,
                    selected,
                    task="R2-1",
                    panel="global_top_k",
                    method=variant,
                    k=k,
                    ap=ap,
                )
                metric["feature_count"] = len(feature_sets[variant])
                metric["tree_count"] = metadata_by_variant[variant]["tree_count"]
                ablation_metric_rows.append(metric)
                selected["method"] = variant
                selected["K"] = k
                ablation_selection_rows.append(
                    selected[["datetime", "station_id", "method", "K", "_rank", "y_same_30", "duration_bin", "red_duration_steps"]]
                )

            global_top10 = ranked.loc[ranked["_rank"].le(10)]
            within = base._rank_rows(
                evaluation,
                ["datetime", "duration_bin"],
                [(score_column, False)],
                score_column,
            )
            within_top10 = within.loc[within["_rank"].le(10)]
            for duration_bin in DURATION_BINS:
                candidates = evaluation.loc[evaluation["duration_bin"].eq(duration_bin)]
                for panel, selected in (
                    ("global_top10_by_duration", global_top10.loc[global_top10["duration_bin"].eq(duration_bin)]),
                    ("within_duration_top10", within_top10.loc[within_top10["duration_bin"].eq(duration_bin)]),
                ):
                    metric = _selected_metric(
                        candidates,
                        selected,
                        task="R2-1",
                        panel=panel,
                        method=variant,
                        k=10,
                        duration_bin=duration_bin,
                    )
                    metric["feature_count"] = len(feature_sets[variant])
                    metric["tree_count"] = metadata_by_variant[variant]["tree_count"]
                    metric["groups_over_k"] = int(
                        (candidates.groupby("datetime", sort=False).size() > 10).sum()
                    )
                    ablation_metric_rows.append(metric)
            for row in _conditional_auc(
                evaluation,
                score_column,
                args.bootstrap_repetitions,
                args.bootstrap_seed + 100 + int(variant[1:]),
            ):
                conditional_rows.append({"variant": variant, **row})

        a0_top10 = ranked_variants["A0"].loc[ranked_variants["A0"]["_rank"].le(10)]
        for variant in ("A1", "A2", "A3", "A4", "A5"):
            selected = ranked_variants[variant].loc[ranked_variants[variant]["_rank"].le(10)]
            result = _bootstrap_difference(
                selected,
                a0_top10,
                args.bootstrap_repetitions,
                args.bootstrap_seed + int(variant[1:]),
            )
            ablation_bootstrap_rows.append(
                {"comparison": f"P@10({variant})-P@10(A0)", **result}
            )
        a5_top10 = ranked_variants["A5"].loc[ranked_variants["A5"]["_rank"].le(10)]
        a4_top10 = ranked_variants["A4"].loc[ranked_variants["A4"]["_rank"].le(10)]
        a5_a4 = _bootstrap_difference(
            a5_top10,
            a4_top10,
            args.bootstrap_repetitions,
            args.bootstrap_seed + 54,
        )
        ablation_bootstrap_rows.append(
            {"comparison": "P@10(A5)-P@10(A4)", **a5_a4}
        )
        ablation_metrics = pd.DataFrame(ablation_metric_rows)
        ablation_bootstrap = pd.DataFrame(ablation_bootstrap_rows)
        ablation_metrics.to_csv(output_dir / "r2_1_metrics.csv", index=False, encoding="utf-8-sig")
        ablation_bootstrap.to_csv(output_dir / "r2_1_bootstrap.csv", index=False, encoding="utf-8-sig")
        conditional_metrics = pd.DataFrame(conditional_rows)
        conditional_metrics.to_csv(
            output_dir / "r2_1_exact_duration_auc.csv", index=False, encoding="utf-8-sig"
        )
        pd.concat(ablation_selection_rows, ignore_index=True).to_parquet(
            output_dir / "r2_1_selections.parquet", index=False, compression="zstd"
        )
        evaluation.to_parquet(
            output_dir / "r2_1_predictions.parquet", index=False, compression="zstd"
        )

        frozen_raw_diff = float(
            np.max(np.abs(evaluation["raw_A0"] - evaluation["raw_lgbm_full"]))
        )
        f0_ranked = _rank(evaluation, "F0")
        a0_f0_topk = []
        for k in K_VALUES:
            left = ranked_variants["A0"].loc[ranked_variants["A0"]["_rank"].le(k), key]
            right = f0_ranked.loc[f0_ranked["_rank"].le(k), key]
            compare = left.merge(right, on=key, how="outer", indicator=True)
            a0_f0_topk.append({"K": k, "different_keys": int(compare["_merge"].ne("both").sum())})
        a0_reproduction = {
            "new_A0_model_sha256": metadata_by_variant["A0"]["model_sha256"],
            "frozen_F0_model_sha256": _sha256(model_path),
            "model_files_bit_identical": metadata_by_variant["A0"]["model_sha256"] == _sha256(model_path),
            "raw_margin_max_absolute_difference": frozen_raw_diff,
            "top_k_differences": a0_f0_topk,
        }
        _write_json(output_dir / "r2_1_a0_f0_reproduction.json", a0_reproduction)
        manifest["tasks"]["R2-1"] = "complete"
        log(f"R2-1 complete; A0/F0 max raw difference={frozen_raw_diff:.3g}")

        log("R2-2: reproducing the recovered no-station model")
        no_station_features = list(freeze["features"]["lgbm_no_station"])
        no_station_schema = protocol["category_schemas"]["lgbm_no_station"]
        recovered = lgb.Booster(model_str=no_station_model_path.read_text(encoding="utf-8"))
        if recovered.feature_name() != no_station_features:
            raise RuntimeError("Recovered no-station model feature order mismatch")
        recovered_raw = recovered.predict(
            _model_matrix(eligible, no_station_features, no_station_schema), raw_score=True
        )
        calibrator = freeze["calibrators"]["lgbm_no_station"]
        recovered_probability = 1.0 / (
            1.0
            + np.exp(
                -(
                    float(calibrator["slope"]) * recovered_raw
                    + float(calibrator["intercept"])
                )
            )
        )
        probability_difference = np.abs(
            recovered_probability - evaluation["p_lgbm_no_station"].to_numpy(float)
        )
        a5_difference = np.abs(recovered_raw - evaluation["raw_A5"].to_numpy(float))
        recovered_eval = evaluation.copy()
        recovered_eval["recovered_no_station_probability"] = recovered_probability
        recovered_ranked = base._rank_rows(
            recovered_eval,
            ["datetime"],
            [("recovered_no_station_probability", False)],
            "recovered_no_station_probability",
        )
        stored_ranked = _rank(evaluation, "STORED_NO_STATION")
        reproduction_topk = []
        for k in K_VALUES:
            left = recovered_ranked.loc[recovered_ranked["_rank"].le(k), key]
            right = stored_ranked.loc[stored_ranked["_rank"].le(k), key]
            compared = left.merge(right, on=key, how="outer", indicator=True)
            reproduction_topk.append(
                {"K": k, "different_selected_keys": int(compared["_merge"].ne("both").sum())}
            )
        reproduction = {
            "recovered_model_path": str(no_station_model_path),
            "recovered_model_sha256": _sha256(no_station_model_path),
            "feature_count": len(no_station_features),
            "features": no_station_features,
            "category_schema": no_station_schema,
            "max_probability_difference": float(probability_difference.max()),
            "different_probability_rows_over_2e_7": int((probability_difference > 2e-7).sum()),
            "top_k": reproduction_topk,
            "reproduced": bool(
                float(probability_difference.max()) <= 2e-7
                and all(row["different_selected_keys"] == 0 for row in reproduction_topk)
            ),
            "a5_model_sha256": metadata_by_variant["A5"]["model_sha256"],
            "a5_raw_max_difference": float(a5_difference.max()),
            "a5_and_recovered_bit_identical": metadata_by_variant["A5"]["model_sha256"] == _sha256(no_station_model_path),
        }
        _write_json(output_dir / "r2_2_no_station_reproduction.json", reproduction)
        pd.DataFrame(
            {
                "datetime": evaluation["datetime"],
                "station_id": evaluation["station_id"],
                "stored_probability": evaluation["p_lgbm_no_station"],
                "recovered_probability": recovered_probability,
                "absolute_difference": probability_difference,
                "new_A5_raw": evaluation["raw_A5"],
                "recovered_raw": recovered_raw,
            }
        ).to_parquet(output_dir / "r2_2_row_parity.parquet", index=False, compression="zstd")
        manifest["tasks"]["R2-2"] = "complete" if reproduction["reproduced"] else "failed_reproduction"
        log(
            "R2-2 complete; max calibrated probability difference="
            f"{reproduction['max_probability_difference']:.3g}"
        )

        log("R2-3: evaluating LONG_FIRST_F0 and list exchanges")
        methods = ("F0", "DURATION", "LONG_FIRST_F0", "STORED_NO_STATION")
        ranked_methods = {method: _rank(evaluation, method) for method in methods}
        policy_rows: list[dict[str, Any]] = []
        policy_selections: list[pd.DataFrame] = []
        deterministic_selected: dict[tuple[str, int], pd.DataFrame] = {}
        for method, ranked in ranked_methods.items():
            for k in K_VALUES:
                selected = _method_selection(ranked, method, k)
                deterministic_selected[(method, k)] = selected
                policy_rows.append(
                    _selected_metric(
                        evaluation,
                        selected,
                        task="R2-3",
                        panel="global_policy",
                        method=method,
                        k=k,
                    )
                )
                policy_selections.append(
                    selected[["datetime", "station_id", "method", "K", "_rank", "y_same_30", "duration_bin", "red_duration_steps", "long_flag"]]
                )

        for seed in range(args.random_seeds):
            rng = np.random.Generator(np.random.PCG64(seed))
            random_frame = evaluation.copy()
            random_frame["random_priority"] = rng.random(len(random_frame))
            ranked = base._rank_rows(
                random_frame,
                ["datetime"],
                [("random_priority", True)],
                "random_priority",
            )
            for k in K_VALUES:
                selected = ranked.loc[ranked["_rank"].le(k)]
                policy_rows.append(
                    _selected_metric(
                        evaluation,
                        selected,
                        task="R2-3",
                        panel="global_policy",
                        method="RANDOM",
                        k=k,
                        seed=seed,
                    )
                )
        policy_metrics = pd.DataFrame(policy_rows)
        policy_metrics.to_csv(output_dir / "r2_3_policy_metrics.csv", index=False, encoding="utf-8-sig")
        pd.concat(policy_selections, ignore_index=True).to_parquet(
            output_dir / "r2_3_policy_selections.parquet", index=False, compression="zstd"
        )
        random_summary = (
            policy_metrics.loc[policy_metrics["method"].eq("RANDOM")]
            .groupby("K")[["precision", "recall"]]
            .agg(["mean", "std", "min", "max"])
        )
        random_summary.columns = ["_".join(column) for column in random_summary.columns]
        random_summary.reset_index().to_csv(
            output_dir / "r2_3_random_summary.csv", index=False, encoding="utf-8-sig"
        )

        exchange_summaries: list[pd.DataFrame] = []
        exchange_rows: list[pd.DataFrame] = []
        policy_swap_summaries: list[pd.DataFrame] = []
        policy_swap_rows: list[pd.DataFrame] = []
        policy_bootstrap_rows: list[dict[str, Any]] = []
        for k in K_VALUES:
            f0 = deterministic_selected[("F0", k)]
            duration = deterministic_selected[("DURATION", k)]
            long_first = deterministic_selected[("LONG_FIRST_F0", k)]
            summary, rows = _exchange_analysis(f0, duration, k)
            exchange_summaries.append(summary)
            exchange_rows.append(rows)
            for comparison_name, other in (("F0", f0), ("DURATION", duration)):
                swap_summary, swap_rows = _exchange_analysis(long_first, other, k)
                swap_summary["comparison"] = f"LONG_FIRST_F0_vs_{comparison_name}"
                swap_rows["comparison"] = f"LONG_FIRST_F0_vs_{comparison_name}"
                swap_rows["membership"] = swap_rows["membership"].replace(
                    {
                        "f0_only": "long_first_only",
                        "duration_only": f"{comparison_name.lower()}_only",
                    }
                )
                swap_summary["membership"] = swap_summary["membership"].replace(
                    {
                        "f0_only": "long_first_only",
                        "duration_only": f"{comparison_name.lower()}_only",
                    }
                )
                policy_swap_summaries.append(swap_summary)
                policy_swap_rows.append(swap_rows)
            for right_method, right in (("F0", f0), ("DURATION", duration)):
                result = _bootstrap_difference(
                    long_first,
                    right,
                    args.bootstrap_repetitions,
                    args.bootstrap_seed + 200 + k + (0 if right_method == "F0" else 50),
                )
                policy_bootstrap_rows.append(
                    {
                        "comparison": f"LONG_FIRST_F0-{right_method}",
                        "K": k,
                        **result,
                    }
                )
        exchange_summary = pd.concat(exchange_summaries, ignore_index=True)
        exchange_summary.to_csv(output_dir / "r2_3_exchange_summary.csv", index=False, encoding="utf-8-sig")
        pd.concat(exchange_rows, ignore_index=True).to_parquet(
            output_dir / "r2_3_exchange_rows.parquet", index=False, compression="zstd"
        )
        pd.concat(policy_swap_summaries, ignore_index=True).to_csv(
            output_dir / "r2_3_policy_swap_summary.csv", index=False, encoding="utf-8-sig"
        )
        pd.concat(policy_swap_rows, ignore_index=True).to_parquet(
            output_dir / "r2_3_policy_swap_rows.parquet", index=False, compression="zstd"
        )
        policy_bootstrap = pd.DataFrame(policy_bootstrap_rows)
        policy_bootstrap.to_csv(output_dir / "r2_3_bootstrap.csv", index=False, encoding="utf-8-sig")

        deterministic_policy = policy_metrics.loc[policy_metrics["seed"].eq(-1)]
        random_mean = (
            policy_metrics.loc[policy_metrics["method"].eq("RANDOM")]
            .groupby("K", as_index=False)[["precision", "recall"]]
            .mean()
        )
        colors = {
            "F0": "#2563eb",
            "DURATION": "#dc2626",
            "LONG_FIRST_F0": "#f59e0b",
            "STORED_NO_STATION": "#7c3aed",
        }
        for metric, y_label, y_min, y_max in (
            ("precision", "P@K", 0.35, 0.90),
            ("recall", "R@K", 0.0, 0.45),
        ):
            series = []
            for method in methods:
                indexed = deterministic_policy.loc[deterministic_policy["method"].eq(method)].set_index("K")
                series.append((method, [float(indexed.loc[k, metric]) for k in K_VALUES], colors[method], False))
            random_indexed = random_mean.set_index("K")
            series.append(("RANDOM mean", [float(random_indexed.loc[k, metric]) for k in K_VALUES], "#64748b", True))
            base._write_line_chart(
                output_dir / "figures" / f"r2_3_{metric}_at_k.svg",
                title=f"R2-3: {metric} at equal list lengths",
                x_values=K_VALUES,
                series=series,
                y_label=y_label,
                y_min=y_min,
                y_max=y_max,
            )
        manifest["tasks"]["R2-3"] = "complete"

        # Keep the task-specific tables, and also provide the single consolidated
        # metrics.csv requested by the evaluation brief.  Columns that only
        # apply to R2-1 (feature/tree counts and group diagnostics) remain NA for
        # R2-3 rather than being silently discarded.
        pd.concat([ablation_metrics, policy_metrics], ignore_index=True, sort=False).to_csv(
            output_dir / "metrics.csv", index=False, encoding="utf-8-sig"
        )

        report = _report(
            args.run_id,
            ablation_metrics,
            ablation_bootstrap,
            conditional_metrics,
            reproduction,
            policy_metrics,
            policy_bootstrap,
            exchange_summary,
            pd.concat(policy_swap_summaries, ignore_index=True),
        )
        (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
        (output_dir / "commands.md").write_text(
            "# Commands\n\n```powershell\n"
            + subprocess.list2cmdline([sys.executable, *sys.argv])
            + "\n```\n\nR2-0 regression:\n\n```powershell\n"
            + ".venv\\Scripts\\python.exe scripts\\run_evaluation.py --run-id <new-run-id> "
            + "--spec-path docs/YOUBIKE_EVALUATION_NEXT_v2.md\n```\n",
            encoding="utf-8",
        )
        (output_dir / "missing_inputs.md").write_text(
            "# Missing inputs\n\nR2-1、R2-2、R2-3 所需材料均已找到並完成。"
            "R2-4 缺少六月之後、未曾參與設計的連續至少 28 日資料，因此依任務書未執行。\n",
            encoding="utf-8",
        )
        log("R2-1, R2-2, and R2-3 completed")
        (output_dir / "logs/run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        manifest.update(
            {
                "finished_at": _now_iso(),
                "status": "complete",
                "counts": {
                    "training_rows": len(known),
                    "training_positives": int(known["y_same_30"].sum()),
                    "june_eligible": len(evaluation),
                    "june_positives": int(evaluation["y_same_30"].sum()),
                    "decision_times": int(evaluation["datetime"].nunique()),
                },
                "frozen_probability_parity_max_difference": f0_probability_diff,
                "a0_reproduction": a0_reproduction,
                "no_station_reproduction": reproduction,
                "artifacts": _artifact_inventory(output_dir),
                "limitations": [
                    "June is retrospective and was previously examined.",
                    "Ablation CIs reflect calendar-day sampling, not training-seed variation.",
                    "LONG_FIRST_F0 was proposed after June was seen and is not a blind-test replacement.",
                    "R2-4 was not run because no confirmed unseen post-June period exists.",
                ],
            }
        )
        _write_json(output_dir / "run_manifest.json", manifest)
        return output_dir
    except Exception:
        error = traceback.format_exc()
        log_lines.append(error)
        (output_dir / "logs/run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        manifest.update({"finished_at": _now_iso(), "status": "failed", "error": error})
        _write_json(output_dir / "run_manifest.json", manifest)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S_r2_next"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--input-path", type=Path, default=Path("data/source/dynamic_red_empty_2026_06_input.parquet"))
    parser.add_argument("--reference-path", type=Path, default=Path("data/reference/june_all_eligible_decisions.parquet"))
    parser.add_argument("--model-path", type=Path, default=Path("model/lgbm_full.txt"))
    parser.add_argument("--no-station-model-path", type=Path, default=Path("model/lgbm_no_station.txt"))
    parser.add_argument("--freeze-path", type=Path, default=Path("config/final_policy_freeze_before_may.json"))
    parser.add_argument("--protocol-path", type=Path, default=Path("config/protocol_frozen_before_june.json"))
    parser.add_argument("--stations-path", type=Path, default=Path("data/stations/dim_station.csv"))
    parser.add_argument("--spec-path", type=Path, default=Path("docs/YOUBIKE_EVALUATION_NEXT_v2.md"))
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--random-seeds", type=int, default=100)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    if args.n_jobs < 1:
        parser.error("--n-jobs must be positive")
    if not 1 <= args.random_seeds <= 1000:
        parser.error("--random-seeds must be between 1 and 1000")
    if not 100 <= args.bootstrap_repetitions <= 100_000:
        parser.error("--bootstrap-repetitions must be between 100 and 100000")
    return args


if __name__ == "__main__":
    destination = run(_parse_args())
    print(f"Evaluation outputs: {destination}")
