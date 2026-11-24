"""Recount published aggregates only. Requires Python 3; no network or private assets."""
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(a, b, label):
    require(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12), label)


def main():
    dat = json.loads(Path(__file__).with_name("summary.json").read_text(encoding="utf-8"))
    cohort, models = dat["cohort"], dat["models"]
    n, positive = cohort["station_observations"], cohort["positive"]
    require(dat["schema_version"] == 1 and (n, positive) == (27962, 11921), "unexpected cohort")
    require(set(models) == {"lightgbm", "msm_matched", "msm_both_origins"}, "unexpected model")
    scores = {}
    for name, model in models.items():
        bins = model["calibration_bins"]
        require(len(bins) == 10 and sum(b["n"] for b in bins) == n, "bin total")
        require(sum(b["positive"] for b in bins) == positive, "positive total")
        for i, b in enumerate(bins):
            same(b["lower"], i/10, "bin lower bound"); same(b["upper"], (i+1)/10, "bin upper bound")
            require(0 <= b["positive"] <= b["n"] and 0 <= b["sum_squared_error"] <= b["n"], "bin counts or SSE")
            if b["n"]:
                require(b["lower"] <= b["mean_probability"] <= b["upper"], "bin mean")
        brier = math.fsum(b["sum_squared_error"] for b in bins) / n
        same(brier, model["brier"], "Brier")
        same(math.fsum(b["n"]*b["mean_probability"] for b in bins if b["n"])/n,
             model["mean_probability"], "mean prediction")
        same(math.fsum(abs(b["positive"]-b["n"]*b["mean_probability"]) for b in bins if b["n"])/n,
             model["ece10"], "ECE10")
        s = scores[name] = {"Brier": brier}
        for k in (3, 10):
            group = model[f"top{k}"]
            require(0 <= group["hits"] <= group["selected"] <= k*cohort["decision_times"], "shortlist totals")
            s[f"P@{k}"] = group["hits"]/group["selected"]
        print(f"{name:18} Top 3: {s['P@3']:.2%}   Top 10: {s['P@10']:.2%}   Brier: {brier:.5f}")
    for row in dat["paired_differences_vs_lightgbm"]:
        m, metric = row["model"], row["metric"]
        same(scores[m][metric]-scores["lightgbm"][metric], row["difference"], "paired estimate")
    for m in ("msm_matched", "msm_both_origins"):
        for k in (3, 10):
            groups = {r["selection"]: r for r in dat["selection_overlap_vs_lightgbm"]
                      if r["model"] == m and r["k"] == k}
            for name, part in [(m, "MSM only"), ("lightgbm", "LightGBM only")]:
                for field, label in [("selected", "n"), ("hits", "hits")]:
                    require(groups["both"][label] + groups[part][label] == models[name][f"top{k}"][field],
                            "overlap totals")
    print("OK: Top-K, Brier, calibration bins, paired point estimates and overlap agree.")
    # The day-level data are private. Printed intervals are published results, not re-estimated here.


if __name__ == "__main__":
    main()
