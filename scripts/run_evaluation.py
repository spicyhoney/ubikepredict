from __future__ import annotations

"""Run the frozen-model EVAL-2, EVAL-3, and EVAL-4 diagnostics.

The runner intentionally uses the truth-free June input for inference and only
joins labels after predictions have been produced.  It does not retrain or
replace the frozen model.
"""

import argparse
import hashlib
import html
import importlib.metadata
import json
import math
import platform
import re
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.inference import FrozenYouBikeModel, load_demo_source, select_eligible_june  # noqa: E402


K_VALUES = (3, 5, 10, 20)
DURATION_BINS = ("B0", "B1", "B2", "B3")
METHOD_LABELS = {
    "F0": "Frozen full model",
    "DURATION": "Longest-duration rule",
    "DURATION_TIE_RANDOM": "Duration rule (random ties)",
    "RANDOM": "Random baseline",
    "STORED_NO_STATION": "Stored no-station score",
    "STORED_HISTORY": "Stored history score",
}
METHOD_COLORS = {
    "F0": "#2563eb",
    "DURATION": "#dc2626",
    "DURATION_TIE_RANDOM": "#f97316",
    "RANDOM": "#64748b",
    "STORED_NO_STATION": "#7c3aed",
    "STORED_HISTORY": "#059669",
}
SELECTION_SCHEMA = pa.schema(
    [
        ("task", pa.string()),
        ("panel", pa.string()),
        ("method", pa.string()),
        ("score_source", pa.string()),
        ("K", pa.int16()),
        ("seed", pa.int16()),
        ("datetime", pa.timestamp("us")),
        ("station_id", pa.int32()),
        ("duration_bin", pa.string()),
        ("red_duration_steps", pa.int16()),
        ("rank", pa.int16()),
        ("ranking_score", pa.float64()),
        ("random_priority", pa.float64()),
        ("y_same_30", pa.int8()),
        ("selected", pa.int8()),
    ]
)
PRIORITY_SCHEMA = pa.schema(
    [
        ("seed", pa.int16()),
        ("row_id", pa.int32()),
        ("datetime", pa.timestamp("us")),
        ("station_id", pa.int32()),
        ("random_priority", pa.float64()),
    ]
)


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _git_commit() -> str:
    git_dir = ROOT / ".git"
    head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        ref = git_dir / head.removeprefix("ref: ")
        if ref.is_file():
            return ref.read_text(encoding="utf-8").strip()
        packed = git_dir / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line and not line.startswith(("#", "^")):
                    commit, name = line.split(" ", 1)
                    if name == head.removeprefix("ref: "):
                        return commit
    return head


def _duration_bin(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise")
    result = np.select(
        [numeric.eq(0), numeric.between(1, 2), numeric.between(3, 5), numeric.ge(6)],
        list(DURATION_BINS),
        default="UNKNOWN",
    )
    return pd.Series(result, index=values.index, dtype="string")


def average_precision_score(y_true: Sequence[int], score: Sequence[float]) -> float:
    """Dependency-free equivalent of sklearn's binary average_precision_score."""
    y = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(score, dtype=np.float64)
    if y.ndim != 1 or scores.ndim != 1 or len(y) != len(scores):
        raise ValueError("y_true and score must be one-dimensional and equally sized")
    positives = int(y.sum())
    if positives == 0:
        return math.nan
    order = np.argsort(-scores, kind="mergesort")
    sorted_y = y[order]
    sorted_scores = scores[order]
    distinct_ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(sorted_scores) - 1]
    true_positives = np.cumsum(sorted_y, dtype=np.int64)[distinct_ends]
    predicted_positives = distinct_ends + 1
    precision = true_positives / predicted_positives
    recall = true_positives / positives
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def _rank_rows(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
    sort_terms: Sequence[tuple[str, bool]],
    ranking_score: str,
) -> pd.DataFrame:
    sort_columns = list(group_columns) + [column for column, _ in sort_terms] + ["station_id"]
    ascending = [True] * len(group_columns) + [value for _, value in sort_terms] + [True]
    ranked = frame.sort_values(sort_columns, ascending=ascending, kind="mergesort").copy()
    ranked["_rank"] = (
        ranked.groupby(list(group_columns), sort=False, observed=True).cumcount() + 1
    ).astype("int16")
    ranked["_ranking_score"] = pd.to_numeric(ranked[ranking_score], errors="raise").astype(
        "float64"
    )
    return ranked


def _metric(
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    task: str,
    panel: str,
    method: str,
    score_source: str,
    seed: int,
    k: int | None,
    duration_bin: str | None = None,
    ap: float | None = None,
    notes: str = "",
) -> dict[str, Any]:
    selected_count = int(len(selected))
    true_positives = int(selected["y_same_30"].sum()) if selected_count else 0
    positive_count = int(candidates["y_same_30"].sum())
    return {
        "task": task,
        "panel": panel,
        "method": method,
        "score_source": score_source,
        "seed": seed,
        "K": k,
        "duration_bin": duration_bin,
        "candidate_count": int(len(candidates)),
        "positive_count": positive_count,
        "selected": selected_count,
        "TP": true_positives,
        "FP": selected_count - true_positives,
        "precision": true_positives / selected_count if selected_count else math.nan,
        "recall": true_positives / positive_count if positive_count else math.nan,
        "AP": ap,
        "selected_share": math.nan,
        "groups_over_k": math.nan,
        "comparison": "",
        "difference": math.nan,
        "ci_lower": math.nan,
        "ci_upper": math.nan,
        "notes": notes,
    }


def _selection_frame(
    selected: pd.DataFrame,
    *,
    task: str,
    panel: str,
    method: str,
    score_source: str,
    seed: int,
    k: int,
) -> pd.DataFrame:
    priority = (
        selected["random_priority"].to_numpy(dtype=np.float64)
        if "random_priority" in selected
        else np.full(len(selected), np.nan, dtype=np.float64)
    )
    return pd.DataFrame(
        {
            "task": task,
            "panel": panel,
            "method": method,
            "score_source": score_source,
            "K": np.full(len(selected), k, dtype=np.int16),
            "seed": np.full(len(selected), seed, dtype=np.int16),
            "datetime": pd.to_datetime(selected["datetime"]).to_numpy(),
            "station_id": selected["station_id"].to_numpy(dtype=np.int32),
            "duration_bin": selected["duration_bin"].astype(str).to_numpy(),
            "red_duration_steps": selected["red_duration_steps"].to_numpy(dtype=np.int16),
            "rank": selected["_rank"].to_numpy(dtype=np.int16),
            "ranking_score": selected["_ranking_score"].to_numpy(dtype=np.float64),
            "random_priority": priority,
            "y_same_30": selected["y_same_30"].to_numpy(dtype=np.int8),
            "selected": np.ones(len(selected), dtype=np.int8),
        }
    )


class _ParquetSink:
    def __init__(self, path: Path, schema: pa.Schema) -> None:
        self.path = path
        self.schema = schema
        self.writer = pq.ParquetWriter(path, schema, compression="zstd")

    def write(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        table = pa.Table.from_pandas(frame, schema=self.schema, preserve_index=False, safe=False)
        self.writer.write_table(table)

    def close(self) -> None:
        self.writer.close()


def _paired_daily_precision_bootstrap(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, float | int]:
    dates = pd.date_range("2026-06-01", "2026-06-30", freq="D")

    def daily(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        work = frame.assign(date=pd.to_datetime(frame["datetime"]).dt.normalize())
        counts = work.groupby("date")["y_same_30"].agg(["sum", "count"]).reindex(
            dates, fill_value=0
        )
        return counts["sum"].to_numpy(float), counts["count"].to_numpy(float)

    left_tp, left_n = daily(left)
    right_tp, right_n = daily(right)
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = rng.integers(0, len(dates), size=(repetitions, len(dates)))
    left_den = left_n[draws].sum(axis=1)
    right_den = right_n[draws].sum(axis=1)
    valid = (left_den > 0) & (right_den > 0)
    differences = left_tp[draws].sum(axis=1)[valid] / left_den[valid]
    differences -= right_tp[draws].sum(axis=1)[valid] / right_den[valid]
    return {
        "point_difference": float(left["y_same_30"].mean() - right["y_same_30"].mean()),
        "ci_lower": float(np.quantile(differences, 0.025)),
        "ci_upper": float(np.quantile(differences, 0.975)),
        "valid_repetitions": int(valid.sum()),
        "na_repetitions": int((~valid).sum()),
    }


def _wilson_interval(successes: int, count: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if count == 0:
        return math.nan, math.nan
    proportion = successes / count
    denominator = 1.0 + z * z / count
    center = (proportion + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(
        proportion * (1 - proportion) / count + z * z / (4 * count * count)
    ) / denominator
    return center - radius, center + radius


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    clipped = np.clip(values, -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _probability_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    probability = np.asarray(probability, dtype=np.float64)
    clipped = np.clip(probability, 1e-15, 1.0 - 1e-15)
    return {
        "brier": float(np.mean((probability - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))),
    }


def _calibration_intercept_slope(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(np.asarray(probability, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    logit = np.log(clipped / (1.0 - clipped))
    design = np.column_stack([np.ones(len(logit)), logit])
    coefficients = np.array([0.0, 1.0], dtype=np.float64)
    for _ in range(100):
        fitted = _sigmoid(design @ coefficients)
        weights = np.maximum(fitted * (1.0 - fitted), 1e-12)
        gradient = design.T @ (y - fitted)
        information = design.T @ (weights[:, None] * design)
        step = np.linalg.solve(information, gradient)
        coefficients += step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    return float(coefficients[0]), float(coefficients[1])


def _calibration_bins(
    y: np.ndarray,
    probability: np.ndarray,
    *,
    score_version: str,
    scheme: str,
) -> pd.DataFrame:
    probability = np.asarray(probability, dtype=np.float64)
    y = np.asarray(y, dtype=np.int8)
    if scheme == "equal_width":
        bin_id = np.minimum((probability * 10).astype(int), 9)
        lower_edges = np.arange(10, dtype=float) / 10.0
        upper_edges = (np.arange(10, dtype=float) + 1.0) / 10.0
        ids: Iterable[int] = range(10)
    elif scheme == "quantile":
        ranked = pd.Series(probability).rank(method="first", pct=True).to_numpy()
        bin_id = np.minimum((ranked * 10).astype(int), 9)
        ids = range(10)
        lower_edges = np.full(10, np.nan)
        upper_edges = np.full(10, np.nan)
    else:
        raise ValueError(f"Unknown calibration scheme: {scheme}")

    rows: list[dict[str, Any]] = []
    for current in ids:
        mask = bin_id == current
        count = int(mask.sum())
        successes = int(y[mask].sum()) if count else 0
        ci_low, ci_high = _wilson_interval(successes, count)
        if count and scheme == "quantile":
            lower_edges[current] = float(np.min(probability[mask]))
            upper_edges[current] = float(np.max(probability[mask]))
        rows.append(
            {
                "score_version": score_version,
                "binning": scheme,
                "bin": current,
                "lower_bound": float(lower_edges[current]),
                "upper_bound": float(upper_edges[current]),
                "count": count,
                "mean_prediction": float(np.mean(probability[mask])) if count else math.nan,
                "observed_rate": successes / count if count else math.nan,
                "observed_ci_lower": ci_low,
                "observed_ci_upper": ci_high,
            }
        )
    return pd.DataFrame(rows)


def _paired_daily_loss_bootstrap(
    frame: pd.DataFrame,
    y: np.ndarray,
    uncalibrated: np.ndarray,
    calibrated: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    dates = pd.to_datetime(frame["datetime"]).dt.normalize()
    unique_dates = pd.date_range("2026-06-01", "2026-06-30", freq="D")
    losses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for metric in ("brier", "log_loss"):
        if metric == "brier":
            before = (uncalibrated - y) ** 2
            after = (calibrated - y) ** 2
        else:
            before_p = np.clip(uncalibrated, 1e-15, 1 - 1e-15)
            after_p = np.clip(calibrated, 1e-15, 1 - 1e-15)
            before = -(y * np.log(before_p) + (1 - y) * np.log(1 - before_p))
            after = -(y * np.log(after_p) + (1 - y) * np.log(1 - after_p))
        daily = pd.DataFrame({"date": dates, "before": before, "after": after}).groupby(
            "date"
        ).agg(before=("before", "sum"), after=("after", "sum"), count=("before", "size"))
        daily = daily.reindex(unique_dates, fill_value=0)
        losses[metric] = (
            (daily["after"] - daily["before"]).to_numpy(float),
            daily["count"].to_numpy(float),
        )
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = rng.integers(0, len(unique_dates), size=(repetitions, len(unique_dates)))
    result: dict[str, dict[str, float]] = {}
    for metric, (difference_sum, count) in losses.items():
        denominator = count[draws].sum(axis=1)
        estimates = difference_sum[draws].sum(axis=1) / denominator
        result[metric] = {
            "difference": float(difference_sum.sum() / count.sum()),
            "ci_lower": float(np.quantile(estimates, 0.025)),
            "ci_upper": float(np.quantile(estimates, 0.975)),
        }
    return result


def _conditional_pairwise_auc(
    frame: pd.DataFrame,
    *,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (decision_time, duration), group in frame.groupby(
        ["datetime", "red_duration_steps"], sort=False
    ):
        positive_scores = group.loc[group["y_same_30"].eq(1), "p_lgbm_full"].to_numpy(float)
        negative_scores = group.loc[group["y_same_30"].eq(0), "p_lgbm_full"].to_numpy(float)
        pairs = len(positive_scores) * len(negative_scores)
        if not pairs:
            continue
        comparisons = positive_scores[:, None] - negative_scores[None, :]
        wins = float((comparisons > 0).sum() + 0.5 * (comparisons == 0).sum())
        rows.append(
            {
                "date": pd.Timestamp(decision_time).normalize(),
                "duration_bin": str(group["duration_bin"].iloc[0]),
                "pairs": pairs,
                "wins": wins,
            }
        )
    pair_frame = pd.DataFrame(rows)
    output: list[dict[str, Any]] = []
    rng = np.random.Generator(np.random.PCG64(seed))
    all_dates = pd.date_range("2026-06-01", "2026-06-30", freq="D")
    draws = rng.integers(0, len(all_dates), size=(repetitions, len(all_dates)))
    for scope in ("ALL", *DURATION_BINS):
        scoped = pair_frame if scope == "ALL" else pair_frame.loc[pair_frame["duration_bin"].eq(scope)]
        daily = scoped.groupby("date")[["wins", "pairs"]].sum().reindex(all_dates, fill_value=0)
        wins = daily["wins"].to_numpy(float)
        pairs = daily["pairs"].to_numpy(float)
        denominators = pairs[draws].sum(axis=1)
        valid = denominators > 0
        estimates = wins[draws].sum(axis=1)[valid] / denominators[valid]
        total_pairs = int(pairs.sum())
        output.append(
            {
                "duration_bin": scope,
                "mixed_label_exact_duration_groups": int(len(scoped)),
                "positive_negative_pairs": total_pairs,
                "conditional_auc": float(wins.sum() / total_pairs) if total_pairs else math.nan,
                "ci_lower": float(np.quantile(estimates, 0.025)) if len(estimates) else math.nan,
                "ci_upper": float(np.quantile(estimates, 0.975)) if len(estimates) else math.nan,
                "valid_bootstrap_repetitions": int(valid.sum()),
            }
        )
    return pd.DataFrame(output)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(map(clean, headers)) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    lines.extend("| " + " | ".join(map(clean, row)) + " |" for row in rows)
    return "\n".join(lines)


def _write_line_chart(
    path: Path,
    *,
    title: str,
    x_values: Sequence[float],
    series: Sequence[tuple[str, Sequence[float], str, bool]],
    y_label: str,
    y_min: float,
    y_max: float,
) -> None:
    width, height = 960, 620
    left, right, top, bottom = 95, 40, 70, 90
    plot_w, plot_h = width - left - right, height - top - bottom

    def x_pos(value: float) -> float:
        return left + (value - min(x_values)) / (max(x_values) - min(x_values)) * plot_w

    def y_pos(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="34" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{html.escape(title)}</text>',
    ]
    for tick in np.linspace(y_min, y_max, 6):
        y = y_pos(float(tick))
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left+plot_w}" y2="{y:.2f}" stroke="#e2e8f0"/>')
        parts.append(f'<text x="{left-12}" y="{y+5:.2f}" text-anchor="end" font-family="Arial" font-size="13">{tick:.2f}</text>')
    for value in x_values:
        x = x_pos(float(value))
        parts.append(f'<text x="{x:.2f}" y="{top+plot_h+28}" text-anchor="middle" font-family="Arial" font-size="14">{value:g}</text>')
    parts.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#334155" stroke-width="1.5"/>',
            f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#334155" stroke-width="1.5"/>',
            f'<text x="{left+plot_w/2}" y="{height-24}" text-anchor="middle" font-family="Arial" font-size="15">K (maximum stations per decision time)</text>',
            f'<text transform="translate(24 {top+plot_h/2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="15">{html.escape(y_label)}</text>',
        ]
    )
    for label, values, color, dashed in series:
        points = " ".join(
            f"{x_pos(float(x)):.2f},{y_pos(float(y)):.2f}" for x, y in zip(x_values, values)
        )
        dash = ' stroke-dasharray="8 6"' if dashed else ""
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"{dash}/>')
        for x, y in zip(x_values, values):
            parts.append(f'<circle cx="{x_pos(float(x)):.2f}" cy="{y_pos(float(y)):.2f}" r="4" fill="{color}"/>')
    legend_x, legend_y = left + 20, top + 18
    for index, (label, _, color, dashed) in enumerate(series):
        y = legend_y + index * 24
        dash = ' stroke-dasharray="8 6"' if dashed else ""
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+30}" y2="{y}" stroke="{color}" stroke-width="3"{dash}/>')
        parts.append(f'<text x="{legend_x+40}" y="{y+5}" font-family="Arial" font-size="13">{html.escape(label)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_grouped_bar_chart(
    path: Path,
    *,
    title: str,
    categories: Sequence[str],
    series: Sequence[tuple[str, Sequence[float], str]],
    y_label: str,
    y_max: float = 1.0,
) -> None:
    width, height = 960, 620
    left, right, top, bottom = 95, 40, 70, 100
    plot_w, plot_h = width - left - right, height - top - bottom
    group_w = plot_w / len(categories)
    bar_w = group_w * 0.72 / len(series)

    def y_pos(value: float) -> float:
        return top + (y_max - value) / y_max * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="34" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{html.escape(title)}</text>',
    ]
    for tick in np.linspace(0, y_max, 6):
        y = y_pos(float(tick))
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left+plot_w}" y2="{y:.2f}" stroke="#e2e8f0"/>')
        parts.append(f'<text x="{left-12}" y="{y+5:.2f}" text-anchor="end" font-family="Arial" font-size="13">{tick:.2f}</text>')
    for group_index, category in enumerate(categories):
        group_left = left + group_index * group_w + group_w * 0.14
        for series_index, (_, values, color) in enumerate(series):
            value = float(values[group_index])
            x = group_left + series_index * bar_w
            y = y_pos(value)
            parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w-3:.2f}" height="{top+plot_h-y:.2f}" fill="{color}"/>')
        x_label = left + group_index * group_w + group_w / 2
        parts.append(f'<text x="{x_label:.2f}" y="{top+plot_h+28}" text-anchor="middle" font-family="Arial" font-size="14">{html.escape(category)}</text>')
    parts.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#334155" stroke-width="1.5"/>',
            f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#334155" stroke-width="1.5"/>',
            f'<text transform="translate(24 {top+plot_h/2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="15">{html.escape(y_label)}</text>',
        ]
    )
    legend_y = height - 42
    for index, (label, _, color) in enumerate(series):
        x = left + index * 240
        parts.append(f'<rect x="{x}" y="{legend_y-12}" width="16" height="16" fill="{color}"/>')
        parts.append(f'<text x="{x+24}" y="{legend_y+1}" font-family="Arial" font-size="13">{html.escape(label)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_reliability_chart(path: Path, bins: pd.DataFrame) -> None:
    width, height = 720, 680
    left, right, top, bottom = 90, 45, 70, 90
    plot_w, plot_h = width - left - right, height - top - bottom

    def x_pos(value: float) -> float:
        return left + value * plot_w

    def y_pos(value: float) -> float:
        return top + (1.0 - value) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="34" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">June reliability diagram</text>',
    ]
    for tick in np.linspace(0, 1, 6):
        x, y = x_pos(float(tick)), y_pos(float(tick))
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left+plot_w}" y2="{y:.2f}" stroke="#e2e8f0"/>')
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top+plot_h}" stroke="#f1f5f9"/>')
        parts.append(f'<text x="{left-12}" y="{y+5:.2f}" text-anchor="end" font-family="Arial" font-size="13">{tick:.1f}</text>')
        parts.append(f'<text x="{x:.2f}" y="{top+plot_h+27}" text-anchor="middle" font-family="Arial" font-size="13">{tick:.1f}</text>')
    parts.append(f'<line x1="{x_pos(0)}" y1="{y_pos(0)}" x2="{x_pos(1)}" y2="{y_pos(1)}" stroke="#94a3b8" stroke-width="2" stroke-dasharray="7 6"/>')
    for score_version, color in (("uncalibrated", "#dc2626"), ("calibrated", "#2563eb")):
        selected = bins.loc[bins["score_version"].eq(score_version) & bins["binning"].eq("equal_width") & bins["count"].gt(0)]
        points = " ".join(
            f"{x_pos(row.mean_prediction):.2f},{y_pos(row.observed_rate):.2f}"
            for row in selected.itertuples()
        )
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        for row in selected.itertuples():
            radius = 3.0 + min(8.0, math.sqrt(row.count) / 13.0)
            parts.append(f'<circle cx="{x_pos(row.mean_prediction):.2f}" cy="{y_pos(row.observed_rate):.2f}" r="{radius:.2f}" fill="{color}" fill-opacity="0.78"/>')
    parts.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#334155" stroke-width="1.5"/>',
            f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#334155" stroke-width="1.5"/>',
            f'<text x="{left+plot_w/2}" y="{height-24}" text-anchor="middle" font-family="Arial" font-size="15">Mean predicted probability</text>',
            f'<text transform="translate(24 {top+plot_h/2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="15">Observed persistence rate</text>',
            f'<line x1="{left+20}" y1="{top+24}" x2="{left+50}" y2="{top+24}" stroke="#2563eb" stroke-width="3"/><text x="{left+60}" y="{top+29}" font-family="Arial" font-size="13">Calibrated</text>',
            f'<line x1="{left+20}" y1="{top+49}" x2="{left+50}" y2="{top+49}" stroke="#dc2626" stroke-width="3"/><text x="{left+60}" y="{top+54}" font-family="Arial" font-size="13">Uncalibrated</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts), encoding="utf-8")


def _summarize_seeded_metrics(metrics: pd.DataFrame, task: str, panel: str) -> pd.DataFrame:
    selected = metrics.loc[metrics["task"].eq(task) & metrics["panel"].eq(panel)].copy()
    group_columns = ["method", "K", "duration_bin"]
    rows: list[dict[str, Any]] = []
    for keys, group in selected.groupby(group_columns, dropna=False, sort=False):
        method, k, duration_bin = keys
        seeded = bool((group["seed"] >= 0).any())
        rows.append(
            {
                "method": method,
                "K": k,
                "duration_bin": duration_bin,
                "runs": int(len(group)),
                "selected": float(group["selected"].mean()),
                "TP": float(group["TP"].mean()),
                "precision": float(group["precision"].mean()),
                "precision_min": float(group["precision"].min()),
                "precision_max": float(group["precision"].max()),
                "recall": float(group["recall"].mean()),
                "AP": float(group["AP"].mean()) if group["AP"].notna().any() else math.nan,
                "candidate_count": float(group["candidate_count"].mean()),
                "positive_count": float(group["positive_count"].mean()),
                "selected_share": float(group["selected_share"].mean())
                if group["selected_share"].notna().any()
                else math.nan,
                "groups_over_k": float(group["groups_over_k"].mean())
                if group["groups_over_k"].notna().any()
                else math.nan,
                "seeded_sensitivity": seeded,
            }
        )
    return pd.DataFrame(rows)


def _format_percent(value: float) -> str:
    return "NA" if pd.isna(value) else f"{100 * float(value):.2f}%"


def _build_report(
    *,
    run_id: str,
    commit: str,
    parity_diff: float,
    eval2: pd.DataFrame,
    bootstrap: dict[str, Any],
    eval3_a: pd.DataFrame,
    eval3_b: pd.DataFrame,
    exact_duration: pd.DataFrame,
    probability_scores: pd.DataFrame,
    calibration_bins: pd.DataFrame,
    random_seed_count: int,
    loss_bootstrap: dict[str, dict[str, float]],
) -> str:
    def format_four(value: float) -> str:
        return "NA" if pd.isna(value) else f"{value:.4f}"

    deterministic = eval2.loc[
        eval2["method"].isin(["F0", "DURATION", "STORED_NO_STATION", "STORED_HISTORY", "RANDOM"])
    ]
    eval2_rows = []
    for k in K_VALUES:
        for method in ("F0", "DURATION", "STORED_NO_STATION", "STORED_HISTORY", "RANDOM"):
            row = deterministic.loc[deterministic["K"].eq(k) & deterministic["method"].eq(method)].iloc[0]
            interval = (
                f" ({_format_percent(row.precision_min)}–{_format_percent(row.precision_max)})"
                if row.seeded_sensitivity
                else ""
            )
            eval2_rows.append(
                [
                    k,
                    METHOD_LABELS[method],
                    int(round(row.selected)),
                    _format_percent(row.precision) + interval,
                    _format_percent(row.recall),
                    f"{row.AP:.4f}" if pd.notna(row.AP) else "NA",
                ]
            )

    tie_rows = []
    for k in K_VALUES:
        row = eval2.loc[
            eval2["K"].eq(k) & eval2["method"].eq("DURATION_TIE_RANDOM")
        ].iloc[0]
        tie_rows.append(
            [
                k,
                _format_percent(row.precision),
                _format_percent(row.precision_min),
                _format_percent(row.precision_max),
            ]
        )

    table_a_rows = []
    for duration_bin in DURATION_BINS:
        for method in ("F0", "DURATION", "RANDOM"):
            row = eval3_a.loc[
                eval3_a["duration_bin"].eq(duration_bin) & eval3_a["method"].eq(method)
            ].iloc[0]
            table_a_rows.append(
                [
                    duration_bin,
                    f"{int(round(row.candidate_count)):,}/{int(round(row.positive_count)):,}",
                    METHOD_LABELS[method],
                    int(round(row.selected)),
                    _format_percent(row.selected_share),
                    _format_percent(row.precision),
                    _format_percent(row.recall),
                ]
            )

    table_b_rows = []
    for duration_bin in DURATION_BINS:
        for method in ("F0", "DURATION", "RANDOM"):
            row = eval3_b.loc[
                eval3_b["duration_bin"].eq(duration_bin) & eval3_b["method"].eq(method)
            ].iloc[0]
            table_b_rows.append(
                [
                    duration_bin,
                    f"{int(round(row.candidate_count)):,}/{int(round(row.positive_count)):,}",
                    int(round(row.groups_over_k)) if pd.notna(row.groups_over_k) else "NA",
                    METHOD_LABELS[method],
                    int(round(row.selected)),
                    _format_percent(row.precision),
                    _format_percent(row.recall),
                ]
            )

    tie_sensitivity_rows = []
    for panel_label, summary in (
        ("全市 Top-10", eval3_a),
        ("粗時長組內 Top-10", eval3_b),
    ):
        for duration_bin in DURATION_BINS:
            fixed = summary.loc[
                summary["duration_bin"].eq(duration_bin)
                & summary["method"].eq("DURATION")
            ].iloc[0]
            randomized = summary.loc[
                summary["duration_bin"].eq(duration_bin)
                & summary["method"].eq("DURATION_TIE_RANDOM")
            ].iloc[0]
            tie_sensitivity_rows.append(
                [
                    panel_label,
                    duration_bin,
                    _format_percent(fixed.precision),
                    _format_percent(randomized.precision),
                    f"{_format_percent(randomized.precision_min)}–{_format_percent(randomized.precision_max)}",
                ]
            )

    exact_rows = [
        [
            row.duration_bin,
            row.mixed_label_exact_duration_groups,
            row.positive_negative_pairs,
            f"{row.conditional_auc:.4f}" if pd.notna(row.conditional_auc) else "NA",
            f"[{row.ci_lower:.4f}, {row.ci_upper:.4f}]" if pd.notna(row.ci_lower) else "NA",
        ]
        for row in exact_duration.itertuples()
    ]

    score_rows = [
        [
            row.score_version,
            f"{row.brier:.6f}",
            f"{row.log_loss:.6f}",
            format_four(row.calibration_intercept),
            format_four(row.calibration_slope),
            f"{row.ece_equal_width:.4f}",
        ]
        for row in probability_scores.itertuples()
    ]
    high_bin = calibration_bins.loc[
        calibration_bins["score_version"].eq("calibrated")
        & calibration_bins["binning"].eq("equal_width")
        & calibration_bins["bin"].ge(8)
    ]
    high_count = int(high_bin["count"].sum())
    all_pairs = int(
        exact_duration.loc[exact_duration["duration_bin"].eq("ALL"), "positive_negative_pairs"].iloc[0]
    )
    b0_pairs = int(
        exact_duration.loc[exact_duration["duration_bin"].eq("B0"), "positive_negative_pairs"].iloc[0]
    )
    b0_pair_share = b0_pairs / all_pairs
    loss_rows = [
        [
            metric,
            f"{values['difference']:.6f}",
            f"[{values['ci_lower']:.6f}, {values['ci_upper']:.6f}]",
        ]
        for metric, values in loss_bootstrap.items()
    ]

    f0_by_k = eval2.loc[eval2["method"].eq("F0")].set_index("K")
    duration_by_k = eval2.loc[eval2["method"].eq("DURATION")].set_index("K")
    crossover_text = ", ".join(
        f"K={k}: {100*(f0_by_k.loc[k, 'precision']-duration_by_k.loc[k, 'precision']):+.2f}pp"
        for k in K_VALUES
    )

    return f"""# YouBike EVAL-2～4 正式回溯評估

- Run ID：`{run_id}`
- Git commit：`{commit}`
- 完成狀態：EVAL-2 完成；EVAL-3 完成並加做 exact-duration 診斷；EVAL-4 完成；EVAL-1 不在本次範圍且缺訓練材料。
- F0 parity：27,962 筆與 frozen reference 的最大絕對機率差為 `{parity_diff:.3g}`。
- 母體流量：truth-free source 92,882 筆 → peak 33,465 筆 → 排除未知起點 5,503 筆 → eligible 27,962 筆；所有 target 都是決策時間後 30 分鐘。

## 一分鐘結論

1. **模型的優勢依名單容量而變。** 固定規則下，F0 相較「缺車最久」的 P@K 差異為 {crossover_text}。小名單 K=3／5 並沒有支持完整模型全面勝出；K=10／20 才由完整模型領先。
2. **模型的額外排序價值主要在較早期缺車。** 粗分組結果要搭配 exact-duration 診斷閱讀；B3 長事件本身正例率很高，不代表模型提供同等資訊增益。
3. **Platt 校正小幅改善整體 loss／ECE，且讓樣本最多的 0.3～0.6 三個機率區間更接近實際比例；並非每個區間都改善。** 最高機率區樣本很少（校正後 0.8 以上共 {high_count} 筆），不可只看最高區間下強結論。

這些結果都是已開封六月資料上的回溯／探索性診斷，不能宣稱為新的盲測、未來泛化或實際調度的因果效果。

## EVAL-2：相同名單長度比較

{_markdown_table(["K", "方法", "入選", "P@K", "R@K", "AP"], eval2_rows)}

隨機方法括號為 {random_seed_count} seeds 的最小到最大 P@K。`STORED_NO_STATION` 與 `STORED_HISTORY` 只有保存分數，沒有對應模型／歷史表，因此只能作保存分數評估，不能稱為本機重新推論。K=10 的 F0 與 duration 數字完全重現既有 E22。

Duration 同分隨機化敏感度（{random_seed_count} seeds）：

{_markdown_table(["K", "平均 P@K", "最小", "最大"], tie_rows)}

K=10 的成對按日 bootstrap：F0 − DURATION = `{100*bootstrap['point_difference']:.2f}pp`，95% 區間 `[{100*bootstrap['ci_lower']:.2f}, {100*bootstrap['ci_upper']:.2f}]pp`，有效抽樣 `{bootstrap['valid_repetitions']}`／`{bootstrap['valid_repetitions'] + bootstrap['na_repetitions']}`。這只反映 30 個六月日期的抽樣變動，沒有解決跨日 episode 相關或多重比較問題。

![EVAL-2 precision](figures/eval2_precision_at_k.svg)

![EVAL-2 recall](figures/eval2_recall_at_k.svg)

## EVAL-3：模型是否只靠長時間缺車？

### 表 A：拆開原本全市 Top-10

{_markdown_table(["時長組", "候選/正例", "方法", "入選", "入選占比", "Precision", "組內 Recall"], table_a_rows)}

這張表保留原本全市最多 10 站的名單，因此可以看不同方法把名額放在哪些時長；但各組入選數不同，不能只靠組內 precision 判定方法勝負。

### 表 B：在相同粗時長區間內各取最多 10 站

{_markdown_table(["時長組", "候選/正例", ">10 時點", "方法", "入選", "Precision", "組內 Recall"], table_b_rows)}

表 B 是診斷政策，一個時刻最多可能選 4×10 站，不能與表 A 的全市 K=10 當成同一營運政策。B3 是 `steps≥6` 的寬區間，仍可能殘留時長差異。

Duration 同分規則在 EVAL-3 的敏感度：

{_markdown_table(["面板", "時長組", "固定 ID Precision", "隨機同分平均", "隨機同分範圍"], tie_sensitivity_rows)}

### Exact-duration 補充診斷

為避免粗分組誤稱已完全控制時長，另在每個 `(datetime, red_duration_steps)` 內，計算模型把正例排在負例前面的成對比例；0.5 相當於沒有區辨力。

{_markdown_table(["時長組", "可比較群組", "正負例配對", "Conditional AUC", "按日 bootstrap 95% CI"], exact_rows)}

這是以正負例配對數加權的 conditional AUC，不是 K=10 precision；全部配對中有 **{100*b0_pair_share:.2f}%** 來自 B0，而 B3 只有 8 對，因此整體值主要反映剛觀察到缺車的站，B3 不足以下結論。

![EVAL-3 duration](figures/eval3_duration_precision.svg)

## EVAL-4：機率校正

{_markdown_table(["分數", "Brier", "Log loss", "校正截距", "校正斜率", "等寬 ECE"], score_rows)}

Brier／log loss 越低越好；理想校正截距為 0、斜率為 1。固定十個等寬區間是主任務；模型分數另保存十等分 quantile bins 作敏感度參考。每區 Wilson 95% 區間是逐列描述性區間，未作 episode／日期叢集校正。沒有在六月重新擬合校正器。

校正後 − 校正前的成對按日 bootstrap（負值代表校正後 loss 較低）：

{_markdown_table(["指標", "差異", "95% CI"], loss_rows)}

![EVAL-4 reliability](figures/eval4_reliability.svg)

## 重要限制

- 六月已在既有研究與 E22 中被查看，本次不能當全新 holdout。
- 標籤是「30 分鐘後仍 exact-zero」，不是調度介入的因果成功率。
- eligible 母體要求 target 快照有效、營運中且容量相同，因此不能直接代表線上所有紅燈站。
- 同一 episode 可每 30 分鐘重複出現；按日 bootstrap 只部分處理相依性。
- 多個 K、方法與時長組屬探索性比較；不要從其中挑最好的一格宣稱確證性顯著。
- Table B 的粗時長 bin 不能完全控制 duration，因此同時交付 exact-duration conditional AUC。

## 可重現檔案

- `metrics.csv`：所有逐 seed 與彙總所需指標。
- `predictions.parquet`：F0、duration 與兩種保存分數的逐列評分資料。
- `selections.parquet`：EVAL-2／3 的逐列入選名單。
- `random_priorities.parquet`：100 seeds 對每列的固定隨機優先值。
- `calibration_bins.csv`、`eval4_scores.csv`：校正明細。
- `run_manifest.json`：hash、版本、參數與執行狀態。
- `commands.md`、`logs/run.log`：實際命令與紀錄。
"""


def run(args: argparse.Namespace) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_id):
        raise ValueError("--run-id may contain only letters, digits, dot, underscore, and hyphen")
    output_dir = args.output_root.resolve() / args.run_id
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    figures_dir = output_dir / "figures"
    logs_dir = output_dir / "logs"
    figures_dir.mkdir(parents=True)
    logs_dir.mkdir()
    started_at = _now_iso()
    log_lines: list[str] = []

    def log(message: str) -> None:
        stamped = f"[{_now_iso()}] {message}"
        print(stamped, flush=True)
        log_lines.append(stamped)

    input_path = ROOT / "data/source/dynamic_red_empty_2026_06_input.parquet"
    reference_path = ROOT / "data/reference/june_all_eligible_decisions.parquet"
    model_path = ROOT / "model/lgbm_full.txt"
    freeze_path = ROOT / "config/final_policy_freeze_before_may.json"
    protocol_path = ROOT / "config/protocol_frozen_before_june.json"
    station_path = ROOT / "data/stations/dim_station.csv"
    spec_path = ROOT.parent / "YOUBIKE_EVALUATION.md"
    tracked_inputs = [input_path, reference_path, model_path, freeze_path, protocol_path, station_path]
    if spec_path.is_file():
        tracked_inputs.append(spec_path)
    code_paths = [Path(__file__).resolve(), ROOT / "tests/test_evaluation.py"]
    command_argv = [str(Path(sys.executable)), *sys.argv]

    manifest: dict[str, Any] = {
        "run_id": args.run_id,
        "started_at": started_at,
        "finished_at": None,
        "status": "running",
        "tasks": {"EVAL-2": "running", "EVAL-3": "pending", "EVAL-4": "pending"},
        "git": {
            "commit": _git_commit(),
            "worktree_note": args.worktree_note,
        },
        "argv": command_argv,
        "command": subprocess.list2cmdline(command_argv),
        "parameters": {
            "task_spec_version": "2026-09-10",
            "K": list(K_VALUES),
            "random_seeds": list(range(args.random_seeds)),
            "random_generator": "numpy.random.Generator(PCG64(seed))",
            "duration_bins": {"B0": "0", "B1": "1-2", "B2": "3-5", "B3": ">=6"},
            "duration_fixed_tie_break": "station_id ascending",
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "bootstrap_seeds": {
                "eval2_p_at_10": args.bootstrap_seed,
                "eval4_loss": args.bootstrap_seed + 1,
                "eval3_exact_duration_auc": args.bootstrap_seed + 2,
            },
            "calibration_bins": "10 equal-width plus 10 equal-count sensitivity bins",
            "label": "same station remains exact-zero at t+30 minutes",
            "population": (
                "decision_is_peak=1; duration_unknown=0; episode start and decision in June; "
                "target before July; current available_bikes=0 and available_docks>0"
            ),
            "methods": {
                "F0": "runtime p_lgbm_full descending, station_id ascending ties",
                "DURATION": "red_duration_steps descending, station_id ascending ties",
                "DURATION_TIE_RANDOM": (
                    "red_duration_steps descending, shared PCG64 priority ascending, "
                    "station_id ascending final ties"
                ),
                "RANDOM": "shared PCG64 priority ascending, station_id ascending final ties",
                "STORED_NO_STATION": "stored p_lgbm_no_station descending",
                "STORED_HISTORY": "stored p_history descending",
            },
        },
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "lightgbm": _package_version("lightgbm"),
        },
        "inputs": {
            str(path.relative_to(ROOT.parent) if path.is_relative_to(ROOT.parent) else path): {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in tracked_inputs
        },
        "code": {
            str(path.relative_to(ROOT)).replace("\\", "/"): {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in code_paths
            if path.is_file()
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    selection_sink: _ParquetSink | None = None
    priority_sink: _ParquetSink | None = None
    try:
        log("Loading truth-free source and applying frozen June eligibility rule")
        source = load_demo_source(path=input_path, mode="empty")
        forbidden = {
            "y_same_30",
            "future_30_balanced",
            "future_30_opposite_red",
            "p_lgbm_full",
            "p_lgbm_no_station",
            "p_history",
        }.intersection(source.columns)
        if forbidden:
            raise RuntimeError(f"Truth-free input unexpectedly contains audit columns: {sorted(forbidden)}")
        peak_mask = source["decision_is_peak"].eq(1)
        peak_rows = int(peak_mask.sum())
        peak_unknown_duration_rows = int(
            (peak_mask & source["duration_unknown"].eq(1)).sum()
        )
        eligible = select_eligible_june(source).sort_values(
            ["datetime", "station_id"], kind="mergesort"
        ).reset_index(drop=True)
        eligible["row_id"] = np.arange(len(eligible), dtype=np.int32)
        if len(source) != 92_882 or len(eligible) != 27_962:
            raise RuntimeError(f"Unexpected source/eligible rows: {len(source)}/{len(eligible)}")
        if peak_rows != 33_465 or peak_unknown_duration_rows != 5_503:
            raise RuntimeError(
                "Unexpected peak/excluded-unknown counts: "
                f"{peak_rows}/{peak_unknown_duration_rows}"
            )
        target_delta = pd.to_datetime(eligible["target_datetime"]) - pd.to_datetime(
            eligible["datetime"]
        )
        if not target_delta.eq(pd.Timedelta(minutes=30)).all():
            raise RuntimeError("Eligible target_datetime must equal datetime + 30 minutes")
        if not (
            eligible["available_bikes"].eq(0) & eligible["available_docks"].gt(0)
        ).all():
            raise RuntimeError("Eligible population must be currently exact-zero with open docks")

        engine = FrozenYouBikeModel(mode="empty")
        log("Running frozen F0 inference before joining June labels")
        predicted = engine.predict_frame(eligible, attach_stations=False)
        reference = pd.read_parquet(reference_path).rename(
            columns={"p_lgbm_full": "stored_p_lgbm_full"}
        )
        key = ["datetime", "station_id"]
        if eligible.duplicated(key).any() or reference.duplicated(key).any():
            raise RuntimeError("Source/reference evaluation keys must be unique")
        evaluation = predicted.merge(
            eligible[["row_id", *key, "duration_unknown"]],
            on=key,
            how="inner",
            validate="one_to_one",
        ).merge(reference, on=key, how="inner", validate="one_to_one", suffixes=("", "_ref"))
        evaluation = evaluation.sort_values(key, kind="mergesort").reset_index(drop=True)
        if len(evaluation) != len(eligible) or set(evaluation["row_id"]) != set(eligible["row_id"]):
            raise RuntimeError("Prediction/reference join did not preserve the eligible population")
        if not pd.to_datetime(evaluation["target_datetime"]).equals(
            pd.to_datetime(evaluation["target_datetime_ref"])
        ):
            raise RuntimeError("Target datetime differs between source and reference")
        if not np.array_equal(
            evaluation["red_duration_steps"].to_numpy(),
            evaluation["red_duration_steps_ref"].to_numpy(),
        ):
            raise RuntimeError("Duration differs between source and reference")
        evaluation = evaluation.rename(columns={"target_datetime": "source_target_datetime"})
        evaluation["target_datetime"] = evaluation.pop("target_datetime_ref")
        evaluation["red_duration_steps"] = evaluation.pop("red_duration_steps_ref").astype("int16")
        evaluation["duration_bin"] = _duration_bin(evaluation["red_duration_steps"])
        evaluation["y_same_30"] = evaluation["y_same_30"].astype("int8")
        parity_diff = float(
            np.max(
                np.abs(
                    evaluation["p_lgbm_full"].to_numpy(float)
                    - evaluation["stored_p_lgbm_full"].to_numpy(float)
                )
            )
        )
        if int(evaluation["y_same_30"].sum()) != 11_921:
            raise RuntimeError("Unexpected June positive count")
        if evaluation["datetime"].nunique() != 420:
            raise RuntimeError("Unexpected decision-time count")
        if parity_diff == 0.0:
            log("F0 parity passed: 27,962 rows, max probability difference 0")
        else:
            log(
                "WARNING: F0 probability parity is non-zero; downstream selection "
                f"differences will be recorded (max absolute difference {parity_diff:.3g})"
            )

        predictions: list[pd.DataFrame] = []
        prediction_specs = [
            ("F0", "runtime_inference", "p_lgbm_full", "raw_lgbm_full", "p_lgbm_full"),
            ("DURATION", "red_duration_steps", "red_duration_steps", None, None),
            ("STORED_NO_STATION", "stored_reference_score", "p_lgbm_no_station", None, "p_lgbm_no_station"),
            ("STORED_HISTORY", "stored_reference_score", "p_history", None, "p_history"),
        ]
        for method, score_source, ranking_column, raw_column, probability_column in prediction_specs:
            predictions.append(
                pd.DataFrame(
                    {
                        "row_id": evaluation["row_id"].astype("int32"),
                        "datetime": evaluation["datetime"],
                        "target_datetime": evaluation["target_datetime"],
                        "station_id": evaluation["station_id"].astype("int32"),
                        "red_duration_steps": evaluation["red_duration_steps"].astype("int16"),
                        "duration_bin": evaluation["duration_bin"].astype(str),
                        "method": method,
                        "score_source": score_source,
                        "raw_score": evaluation[raw_column].astype(float) if raw_column else np.nan,
                        "probability": evaluation[probability_column].astype(float) if probability_column else np.nan,
                        "ranking_score": evaluation[ranking_column].astype(float),
                        "y_same_30": evaluation["y_same_30"].astype("int8"),
                    }
                )
            )
        pd.concat(predictions, ignore_index=True).to_parquet(
            output_dir / "predictions.parquet", index=False, compression="zstd"
        )

        selection_sink = _ParquetSink(output_dir / "selections.parquet", SELECTION_SCHEMA)
        priority_sink = _ParquetSink(output_dir / "random_priorities.parquet", PRIORITY_SCHEMA)
        metrics: list[dict[str, Any]] = []
        score_columns = {
            "F0": ("p_lgbm_full", "runtime_inference"),
            "DURATION": ("red_duration_steps", "red_duration_steps"),
            "STORED_NO_STATION": ("p_lgbm_no_station", "stored_reference_score"),
            "STORED_HISTORY": ("p_history", "stored_reference_score"),
        }
        global_ranked: dict[str, pd.DataFrame] = {}
        global_ap: dict[str, float] = {}
        for method, (column, source_name) in score_columns.items():
            global_ranked[method] = _rank_rows(
                evaluation,
                ["datetime"],
                [(column, False)],
                column,
            )
            global_ap[method] = average_precision_score(
                evaluation["y_same_30"], evaluation[column]
            )
            for k in K_VALUES:
                selected = global_ranked[method].loc[global_ranked[method]["_rank"].le(k)].copy()
                metrics.append(
                    _metric(
                        evaluation,
                        selected,
                        task="EVAL-2",
                        panel="same_length_global",
                        method=method,
                        score_source=source_name,
                        seed=-1,
                        k=k,
                        ap=global_ap[method],
                    )
                )
                selection_sink.write(
                    _selection_frame(
                        selected,
                        task="EVAL-2",
                        panel="same_length_global",
                        method=method,
                        score_source=source_name,
                        seed=-1,
                        k=k,
                    )
                )

        stored_f0_ranked = _rank_rows(
            evaluation,
            ["datetime"],
            [("stored_p_lgbm_full", False)],
            "stored_p_lgbm_full",
        )
        parity_downstream: list[dict[str, Any]] = []
        for k in K_VALUES:
            runtime_selected = global_ranked["F0"].loc[
                global_ranked["F0"]["_rank"].le(k), ["datetime", "station_id", "y_same_30"]
            ]
            stored_selected = stored_f0_ranked.loc[
                stored_f0_ranked["_rank"].le(k), ["datetime", "station_id", "y_same_30"]
            ]
            comparison = runtime_selected.merge(
                stored_selected,
                on=["datetime", "station_id"],
                how="outer",
                indicator=True,
                suffixes=("_runtime", "_stored"),
            )
            parity_downstream.append(
                {
                    "K": k,
                    "different_selected_keys": int(comparison["_merge"].ne("both").sum()),
                    "runtime_TP": int(runtime_selected["y_same_30"].sum()),
                    "stored_TP": int(stored_selected["y_same_30"].sum()),
                }
            )
        manifest["f0_parity"] = {
            "max_absolute_probability_difference": parity_diff,
            "bit_exact": parity_diff == 0.0,
            "downstream_top_k": parity_downstream,
        }

        expected_by_k = {
            k: int(evaluation.groupby("datetime", sort=False).size().clip(upper=k).sum())
            for k in K_VALUES
        }
        for k in K_VALUES:
            counts = [
                row["selected"]
                for row in metrics
                if row["task"] == "EVAL-2" and row["K"] == k
            ]
            if set(counts) != {expected_by_k[k]}:
                raise RuntimeError(f"Deterministic methods selected unequal rows at K={k}: {counts}")

        log(
            f"Running {args.random_seeds} shared-priority random and "
            "duration-tie sensitivity seeds"
        )
        for seed in range(args.random_seeds):
            rng = np.random.Generator(np.random.PCG64(seed))
            seeded = evaluation.copy()
            seeded["random_priority"] = rng.random(len(seeded))
            priority_sink.write(
                pd.DataFrame(
                    {
                        "seed": np.full(len(seeded), seed, dtype=np.int16),
                        "row_id": seeded["row_id"].to_numpy(dtype=np.int32),
                        "datetime": pd.to_datetime(seeded["datetime"]).to_numpy(),
                        "station_id": seeded["station_id"].to_numpy(dtype=np.int32),
                        "random_priority": seeded["random_priority"].to_numpy(float),
                    }
                )
            )
            random_ranked = _rank_rows(
                seeded,
                ["datetime"],
                [("random_priority", True)],
                "random_priority",
            )
            duration_tie_ranked = _rank_rows(
                seeded,
                ["datetime"],
                [("red_duration_steps", False), ("random_priority", True)],
                "red_duration_steps",
            )
            random_ap = average_precision_score(
                seeded["y_same_30"], -seeded["random_priority"].to_numpy(float)
            )
            for k in K_VALUES:
                for method, ranked, score_source, ap_value in (
                    ("RANDOM", random_ranked, "pcg64_priority", random_ap),
                    (
                        "DURATION_TIE_RANDOM",
                        duration_tie_ranked,
                        "duration_then_pcg64_tie",
                        global_ap["DURATION"],
                    ),
                ):
                    selected = ranked.loc[ranked["_rank"].le(k)].copy()
                    if len(selected) != expected_by_k[k]:
                        raise RuntimeError(f"{method} seed {seed} selected wrong count at K={k}")
                    metrics.append(
                        _metric(
                            evaluation,
                            selected,
                            task="EVAL-2",
                            panel="same_length_global",
                            method=method,
                            score_source=score_source,
                            seed=seed,
                            k=k,
                            ap=ap_value,
                        )
                    )
                    selection_sink.write(
                        _selection_frame(
                            selected,
                            task="EVAL-2",
                            panel="same_length_global",
                            method=method,
                            score_source=score_source,
                            seed=seed,
                            k=k,
                        )
                    )

            for method, ranked, score_source in (
                ("RANDOM", random_ranked, "pcg64_priority"),
                ("DURATION_TIE_RANDOM", duration_tie_ranked, "duration_then_pcg64_tie"),
            ):
                selected = ranked.loc[ranked["_rank"].le(10)].copy()
                for duration_bin in DURATION_BINS:
                    candidates_bin = evaluation.loc[evaluation["duration_bin"].eq(duration_bin)]
                    selected_bin = selected.loc[selected["duration_bin"].eq(duration_bin)]
                    record = _metric(
                        candidates_bin,
                        selected_bin,
                        task="EVAL-3",
                        panel="table_a_global_top10",
                        method=method,
                        score_source=score_source,
                        seed=seed,
                        k=10,
                        duration_bin=duration_bin,
                    )
                    record["selected_share"] = len(selected_bin) / len(selected)
                    metrics.append(record)
                selection_sink.write(
                    _selection_frame(
                        selected,
                        task="EVAL-3",
                        panel="table_a_global_top10",
                        method=method,
                        score_source=score_source,
                        seed=seed,
                        k=10,
                    )
                )

            random_within = _rank_rows(
                seeded,
                ["datetime", "duration_bin"],
                [("random_priority", True)],
                "random_priority",
            )
            duration_tie_within = _rank_rows(
                seeded,
                ["datetime", "duration_bin"],
                [("red_duration_steps", False), ("random_priority", True)],
                "red_duration_steps",
            )
            for method, ranked, score_source in (
                ("RANDOM", random_within, "pcg64_priority"),
                ("DURATION_TIE_RANDOM", duration_tie_within, "duration_then_pcg64_tie"),
            ):
                selected = ranked.loc[ranked["_rank"].le(10)].copy()
                for duration_bin in DURATION_BINS:
                    candidates_bin = evaluation.loc[evaluation["duration_bin"].eq(duration_bin)]
                    selected_bin = selected.loc[selected["duration_bin"].eq(duration_bin)]
                    record = _metric(
                        candidates_bin,
                        selected_bin,
                        task="EVAL-3",
                        panel="table_b_within_coarse_bin",
                        method=method,
                        score_source=score_source,
                        seed=seed,
                        k=10,
                        duration_bin=duration_bin,
                    )
                    record["groups_over_k"] = int(
                        (candidates_bin.groupby("datetime", sort=False).size() > 10).sum()
                    )
                    metrics.append(record)
                selection_sink.write(
                    _selection_frame(
                        selected,
                        task="EVAL-3",
                        panel="table_b_within_coarse_bin",
                        method=method,
                        score_source=score_source,
                        seed=seed,
                        k=10,
                    )
                )

        priority_sink.close()
        priority_sink = None
        log("EVAL-2 random baselines and tie sensitivity complete")

        f0_k10 = global_ranked["F0"].loc[global_ranked["F0"]["_rank"].le(10)].copy()
        duration_k10 = global_ranked["DURATION"].loc[
            global_ranked["DURATION"]["_rank"].le(10)
        ].copy()
        if (len(f0_k10), int(f0_k10["y_same_30"].sum())) != (4_152, 2_574):
            raise RuntimeError("F0 K10 does not reproduce E22")
        if (len(duration_k10), int(duration_k10["y_same_30"].sum())) != (4_152, 2_327):
            raise RuntimeError("Duration K10 does not reproduce E22")
        bootstrap = _paired_daily_precision_bootstrap(
            f0_k10,
            duration_k10,
            repetitions=args.bootstrap_repetitions,
            seed=args.bootstrap_seed,
        )
        metrics.append(
            {
                "task": "EVAL-2",
                "panel": "paired_daily_bootstrap",
                "method": "F0_MINUS_DURATION",
                "score_source": "runtime_vs_duration",
                "seed": args.bootstrap_seed,
                "K": 10,
                "duration_bin": None,
                "candidate_count": len(evaluation),
                "positive_count": int(evaluation["y_same_30"].sum()),
                "selected": len(f0_k10),
                "TP": int(f0_k10["y_same_30"].sum()),
                "FP": len(f0_k10) - int(f0_k10["y_same_30"].sum()),
                "precision": float(f0_k10["y_same_30"].mean()),
                "recall": float(f0_k10["y_same_30"].sum() / evaluation["y_same_30"].sum()),
                "AP": global_ap["F0"],
                "comparison": "P@10(F0)-P@10(DURATION)",
                "difference": bootstrap["point_difference"],
                "ci_lower": bootstrap["ci_lower"],
                "ci_upper": bootstrap["ci_upper"],
                "notes": "paired calendar-day percentile bootstrap",
            }
        )
        manifest["tasks"]["EVAL-2"] = "complete"

        log("Running EVAL-3 global-list and within-duration diagnostics")
        for method, ranked, score_source in (
            ("F0", global_ranked["F0"], "runtime_inference"),
            ("DURATION", global_ranked["DURATION"], "red_duration_steps"),
        ):
            selected = ranked.loc[ranked["_rank"].le(10)].copy()
            for duration_bin in DURATION_BINS:
                candidates_bin = evaluation.loc[evaluation["duration_bin"].eq(duration_bin)]
                selected_bin = selected.loc[selected["duration_bin"].eq(duration_bin)]
                record = _metric(
                    candidates_bin,
                    selected_bin,
                    task="EVAL-3",
                    panel="table_a_global_top10",
                    method=method,
                    score_source=score_source,
                    seed=-1,
                    k=10,
                    duration_bin=duration_bin,
                )
                record["selected_share"] = len(selected_bin) / len(selected)
                metrics.append(record)
            selection_sink.write(
                _selection_frame(
                    selected,
                    task="EVAL-3",
                    panel="table_a_global_top10",
                    method=method,
                    score_source=score_source,
                    seed=-1,
                    k=10,
                )
            )

        within_ranked = {
            "F0": _rank_rows(
                evaluation,
                ["datetime", "duration_bin"],
                [("p_lgbm_full", False)],
                "p_lgbm_full",
            ),
            "DURATION": _rank_rows(
                evaluation,
                ["datetime", "duration_bin"],
                [("red_duration_steps", False)],
                "red_duration_steps",
            ),
        }
        for method, ranked in within_ranked.items():
            score_source = "runtime_inference" if method == "F0" else "red_duration_steps"
            selected = ranked.loc[ranked["_rank"].le(10)].copy()
            for duration_bin in DURATION_BINS:
                candidates_bin = evaluation.loc[evaluation["duration_bin"].eq(duration_bin)]
                selected_bin = selected.loc[selected["duration_bin"].eq(duration_bin)]
                groups_over_k = int(
                    (candidates_bin.groupby("datetime", sort=False).size() > 10).sum()
                )
                record = _metric(
                    candidates_bin,
                    selected_bin,
                    task="EVAL-3",
                    panel="table_b_within_coarse_bin",
                    method=method,
                    score_source=score_source,
                    seed=-1,
                    k=10,
                    duration_bin=duration_bin,
                    notes=f"datetime groups with >10 candidates: {groups_over_k}",
                )
                record["groups_over_k"] = groups_over_k
                metrics.append(record)
            selection_sink.write(
                _selection_frame(
                    selected,
                    task="EVAL-3",
                    panel="table_b_within_coarse_bin",
                    method=method,
                    score_source=score_source,
                    seed=-1,
                    k=10,
                )
            )
        exact_duration = _conditional_pairwise_auc(
            evaluation,
            repetitions=args.bootstrap_repetitions,
            seed=args.bootstrap_seed + 2,
        )
        exact_duration.to_csv(output_dir / "eval3_exact_duration.csv", index=False, encoding="utf-8-sig")
        selection_sink.close()
        selection_sink = None
        manifest["tasks"]["EVAL-3"] = "complete"

        log("Running EVAL-4 calibration diagnostics")
        y = evaluation["y_same_30"].to_numpy(dtype=np.float64)
        raw = evaluation["raw_lgbm_full"].to_numpy(dtype=np.float64)
        uncalibrated = _sigmoid(raw)
        calibrated = evaluation["p_lgbm_full"].to_numpy(dtype=np.float64)
        prevalence = np.full(len(y), float(y.mean()), dtype=np.float64)
        score_rows = []
        calibration_frames = []
        for name, probability in (
            ("prevalence_baseline", prevalence),
            ("uncalibrated", uncalibrated),
            ("calibrated", calibrated),
        ):
            base_scores = _probability_metrics(y, probability)
            if name == "prevalence_baseline":
                intercept, slope = math.nan, math.nan
            else:
                intercept, slope = _calibration_intercept_slope(y, probability)
            equal_bins = _calibration_bins(
                y, probability, score_version=name, scheme="equal_width"
            )
            quantile_bins = (
                _calibration_bins(y, probability, score_version=name, scheme="quantile")
                if name != "prevalence_baseline"
                else pd.DataFrame()
            )
            ece = float(
                np.sum(
                    equal_bins["count"]
                    / len(y)
                    * np.abs(equal_bins["mean_prediction"] - equal_bins["observed_rate"])
                )
            )
            score_rows.append(
                {
                    "score_version": name,
                    **base_scores,
                    "mean_prediction": float(np.mean(probability)),
                    "observed_rate": float(np.mean(y)),
                    "calibration_intercept": intercept,
                    "calibration_slope": slope,
                    "ece_equal_width": ece,
                }
            )
            calibration_frames.extend([equal_bins, quantile_bins])
        probability_scores = pd.DataFrame(score_rows)
        calibration_bins = pd.concat(calibration_frames, ignore_index=True)
        probability_scores.to_csv(output_dir / "eval4_scores.csv", index=False, encoding="utf-8-sig")
        calibration_bins.to_csv(
            output_dir / "calibration_bins.csv", index=False, encoding="utf-8-sig"
        )
        loss_bootstrap = _paired_daily_loss_bootstrap(
            evaluation,
            y,
            uncalibrated,
            calibrated,
            repetitions=args.bootstrap_repetitions,
            seed=args.bootstrap_seed + 1,
        )
        for metric_name, values in loss_bootstrap.items():
            metrics.append(
                {
                    "task": "EVAL-4",
                    "panel": "paired_daily_bootstrap",
                    "method": "CALIBRATED_MINUS_UNCALIBRATED",
                    "score_source": metric_name,
                    "seed": args.bootstrap_seed + 1,
                    "K": None,
                    "duration_bin": None,
                    "candidate_count": len(evaluation),
                    "positive_count": int(y.sum()),
                    "selected": len(evaluation),
                    "TP": int(y.sum()),
                    "FP": int(len(y) - y.sum()),
                    "precision": math.nan,
                    "recall": math.nan,
                    "AP": math.nan,
                    "comparison": f"{metric_name}(calibrated)-{metric_name}(uncalibrated)",
                    "difference": values["difference"],
                    "ci_lower": values["ci_lower"],
                    "ci_upper": values["ci_upper"],
                    "notes": "paired calendar-day percentile bootstrap; lower favors calibration",
                }
            )
        manifest["tasks"]["EVAL-4"] = "complete"

        metrics_frame = pd.DataFrame(metrics)
        numeric_identity = metrics_frame.loc[metrics_frame["selected"].notna()]
        if not (
            numeric_identity["TP"].astype(float) + numeric_identity["FP"].astype(float)
            == numeric_identity["selected"].astype(float)
        ).all():
            raise RuntimeError("Metric identity TP + FP = selected failed")
        metrics_frame.to_csv(output_dir / "metrics.csv", index=False, encoding="utf-8-sig")
        eval2_summary = _summarize_seeded_metrics(
            metrics_frame, "EVAL-2", "same_length_global"
        )
        eval3_a = _summarize_seeded_metrics(
            metrics_frame, "EVAL-3", "table_a_global_top10"
        )
        eval3_b = _summarize_seeded_metrics(
            metrics_frame, "EVAL-3", "table_b_within_coarse_bin"
        )
        eval2_summary.to_csv(output_dir / "eval2_summary.csv", index=False, encoding="utf-8-sig")
        eval3_a.to_csv(output_dir / "eval3_table_a.csv", index=False, encoding="utf-8-sig")
        eval3_b.to_csv(output_dir / "eval3_table_b.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([bootstrap]).to_csv(
            output_dir / "eval2_bootstrap.csv", index=False, encoding="utf-8-sig"
        )

        plot_methods = ("F0", "DURATION", "STORED_NO_STATION", "STORED_HISTORY", "RANDOM")
        precision_series = []
        recall_series = []
        for method in plot_methods:
            method_rows = eval2_summary.loc[eval2_summary["method"].eq(method)].set_index("K")
            precision_series.append(
                (
                    METHOD_LABELS[method],
                    [float(method_rows.loc[k, "precision"]) for k in K_VALUES],
                    METHOD_COLORS[method],
                    method == "RANDOM",
                )
            )
            recall_series.append(
                (
                    METHOD_LABELS[method],
                    [float(method_rows.loc[k, "recall"]) for k in K_VALUES],
                    METHOD_COLORS[method],
                    method == "RANDOM",
                )
            )
        _write_line_chart(
            figures_dir / "eval2_precision_at_k.svg",
            title="EVAL-2: precision at equal list lengths",
            x_values=K_VALUES,
            series=precision_series,
            y_label="P@K",
            y_min=0.35,
            y_max=0.90,
        )
        _write_line_chart(
            figures_dir / "eval2_recall_at_k.svg",
            title="EVAL-2: recall at equal list lengths",
            x_values=K_VALUES,
            series=recall_series,
            y_label="R@K",
            y_min=0.0,
            y_max=0.45,
        )
        duration_series = []
        for method in ("F0", "DURATION", "RANDOM"):
            values = []
            for duration_bin in DURATION_BINS:
                row = eval3_b.loc[
                    eval3_b["method"].eq(method)
                    & eval3_b["duration_bin"].eq(duration_bin)
                ].iloc[0]
                values.append(float(row["precision"]))
            duration_series.append((METHOD_LABELS[method], values, METHOD_COLORS[method]))
        _write_grouped_bar_chart(
            figures_dir / "eval3_duration_precision.svg",
            title="EVAL-3: precision within coarse duration bins",
            categories=[
                f"{duration_bin} n={int(eval3_b.loc[eval3_b['duration_bin'].eq(duration_bin) & eval3_b['method'].eq('F0'), 'candidate_count'].iloc[0])}"
                for duration_bin in DURATION_BINS
            ],
            series=duration_series,
            y_label="Precision",
        )
        _write_reliability_chart(figures_dir / "eval4_reliability.svg", calibration_bins)

        report = _build_report(
            run_id=args.run_id,
            commit=manifest["git"]["commit"],
            parity_diff=parity_diff,
            eval2=eval2_summary,
            bootstrap=bootstrap,
            eval3_a=eval3_a,
            eval3_b=eval3_b,
            exact_duration=exact_duration,
            probability_scores=probability_scores,
            calibration_bins=calibration_bins,
            random_seed_count=args.random_seeds,
            loss_bootstrap=loss_bootstrap,
        )
        (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
        (output_dir / "commands.md").write_text(
            "# Commands\n\n"
            f"```powershell\n{manifest['command']}\n```\n\n"
            "Verification:\n\n"
            "```powershell\n"
            ".venv\\Scripts\\python.exe -m unittest tests.test_evaluation -v\n"
            ".venv\\Scripts\\python.exe scripts\\run_evaluation.py --run-id <new-run-id>\n"
            "```\n",
            encoding="utf-8",
        )
        (output_dir / "missing_inputs.md").write_text(
            "# Inputs missing for EVAL-1\n\n"
            "EVAL-1 was not requested in this run and cannot yet be reproduced fairly. "
            "The repository does not contain the original 1–3 month training features/labels, "
            "April calibration/gate splits, feature builder/trainer, or original stopping logic. "
            "Stored no-station scores are evaluation evidence only and cannot replace a new A5 fit.\n",
            encoding="utf-8",
        )

        log("EVAL-2, EVAL-3, and EVAL-4 completed successfully")
        (logs_dir / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        manifest.update(
            {
                "finished_at": _now_iso(),
                "status": "complete",
                "counts": {
                    "source_rows": len(source),
                    "peak_rows": peak_rows,
                    "peak_unknown_duration_excluded": peak_unknown_duration_rows,
                    "eligible_rows": len(evaluation),
                    "positive_rows": int(evaluation["y_same_30"].sum()),
                    "decision_times": int(evaluation["datetime"].nunique()),
                    "calendar_days": int(pd.to_datetime(evaluation["datetime"]).dt.normalize().nunique()),
                    "f0_parity_max_abs_diff": parity_diff,
                },
                "features": {"f0_count": len(engine.features), "f0_ordered": engine.features},
                "artifacts": sorted(
                    str(path.relative_to(output_dir)).replace("\\", "/")
                    for path in output_dir.rglob("*")
                    if path.is_file() and path.name != "run_manifest.json"
                ),
                "limitations": [
                    "June was previously examined; results are retrospective/exploratory.",
                    "Stored no-station/history scores are not fresh local inference.",
                    "Daily bootstrap does not eliminate cross-day episode dependence.",
                    "Exact-zero persistence is not a causal intervention outcome.",
                ],
            }
        )
        (output_dir / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return output_dir
    except Exception:
        if selection_sink is not None:
            selection_sink.close()
        if priority_sink is not None:
            priority_sink.close()
        error_text = traceback.format_exc()
        log_lines.append(error_text)
        (logs_dir / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        manifest.update(
            {
                "finished_at": _now_iso(),
                "status": "failed",
                "error": error_text,
            }
        )
        (output_dir / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        default=datetime.now().astimezone().strftime("%Y%m%d_%H%M%S"),
        help="Unique output directory name.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/evaluation",
        help="Parent directory for evaluation runs.",
    )
    parser.add_argument("--random-seeds", type=int, default=100)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260910)
    parser.add_argument(
        "--worktree-note",
        default="not supplied",
        help="Recorded note from an external git status check.",
    )
    args = parser.parse_args()
    if args.random_seeds <= 0 or args.random_seeds > 1000:
        parser.error("--random-seeds must be between 1 and 1000")
    if args.bootstrap_repetitions < 100 or args.bootstrap_repetitions > 100_000:
        parser.error("--bootstrap-repetitions must be between 100 and 100000")
    return args


if __name__ == "__main__":
    destination = run(_parse_args())
    print(f"Evaluation outputs: {destination}")
