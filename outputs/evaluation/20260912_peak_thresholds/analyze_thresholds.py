"""Reproduce frozen June peak-time threshold metrics, then render a static figure.

Run --analyze with the existing project Python (Parquet/LightGBM support).
Run --plot with bundled Python and Node/Sharp for SVG/PNG rendering.
No model, calibration, threshold configuration, or source data is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]


def analyze() -> None:
    import pandas as pd

    sys.path.insert(0, str(ROOT))
    from backend.inference import FrozenYouBikeModel, load_demo_source, select_eligible_june

    input_path = ROOT / "data/source/dynamic_red_empty_2026_06_input.parquet"
    reference_path = ROOT / "data/reference/june_all_eligible_decisions.parquet"
    source = load_demo_source(path=input_path, mode="empty")
    assert not {"y_same_30", "p_lgbm_full"}.intersection(source.columns)
    eligible = select_eligible_june(source)
    assert len(source) == 92_882 and len(eligible) == 27_962
    assert eligible["available_bikes"].eq(0).all()
    assert eligible["available_docks"].gt(0).all()
    assert eligible["datetime"].dt.hour.isin([7, 8, 9, 16, 17, 18, 19]).all()
    assert (eligible["target_datetime"] - eligible["datetime"]).eq(pd.Timedelta(minutes=30)).all()
    keys = ["datetime", "station_id"]
    assert not eligible.duplicated(keys).any()

    engine = FrozenYouBikeModel(mode="empty")
    predicted = engine.predict_frame(eligible, attach_stations=False)
    reference = pd.read_parquet(reference_path)
    assert not reference.duplicated(keys).any()
    evaluation = predicted.merge(reference, on=keys, validate="one_to_one", suffixes=("", "_reference"))
    assert len(evaluation) == len(eligible) == len(reference)
    assert evaluation["target_datetime"].equals(evaluation["target_datetime_reference"])
    assert evaluation["red_duration_steps"].equals(evaluation["red_duration_steps_reference"])
    diff = float(np.abs(evaluation["p_lgbm_full"] - evaluation["p_lgbm_full_reference"]).max())
    assert diff == 0.0, f"Frozen probability parity failed: {diff}"
    for name in ["alert_secondary_balanced", "alert_primary_strict"]:
        assert evaluation[name].equals(evaluation[f"{name}_reference"])

    p = evaluation["p_lgbm_full"].to_numpy()
    y = evaluation["y_same_30"].to_numpy(dtype=np.int8)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    assert set(np.unique(y)) == {0, 1}
    positive_count = int(y.sum())
    assert positive_count == 11_921
    balanced, strict = engine.balanced_threshold, engine.strict_threshold
    thresholds = sorted(set([i / 100 for i in range(101)] + [balanced, strict]))
    records = []
    for threshold in thresholds:
        selected = p >= threshold
        tn, fp, fn, tp = map(int, np.bincount(2 * y + selected.astype(int), minlength=4))
        count = int(selected.sum())
        assert tp == int(y[selected].sum()) and tp + fn == positive_count
        assert tp + fp == count and tp + fp + fn + tn == len(y)
        records.append({
            "threshold": threshold,
            "policy": "balanced" if threshold == balanced else "strict" if threshold == strict else "comparison",
            "selected": count, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / count if count else None,
            "recall": tp / positive_count,
            "alert_fraction": count / len(y),
        })
    assert all(a["selected"] >= b["selected"] for a, b in zip(records, records[1:]))
    assert all(a["recall"] >= b["recall"] for a, b in zip(records, records[1:]))

    file_paths = [input_path, reference_path, engine.model_path, engine.freeze_path, engine.protocol_path]
    meta = {
        "mode": "empty", "evaluation": "June 2026 frozen retrospective threshold sensitivity",
        "decision_hours": [7, 8, 9, 16, 17, 18, 19], "weekends_included": True,
        "source_rows": len(source), "eligible_rows": len(y), "positive_rows": positive_count,
        "decision_timestamps": int(evaluation["datetime"].nunique()),
        "date_count": int(evaluation["datetime"].dt.date.nunique()),
        "first_decision": str(evaluation["datetime"].min()), "last_decision": str(evaluation["datetime"].max()),
        "max_calibrated_probability": float(p.max()), "parity_max_abs_error": diff,
        "balanced_threshold": balanced, "strict_threshold": strict,
        "aggregation": "pooled station-by-decision-time observations; no Top-K cap",
        "target": "available bikes equal zero at t+30 minutes, conditional on currently empty",
        "threshold_operator": ">=", "zero_alert_precision": None,
        "sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in file_paths},
    }
    result = {"metadata": meta, "rows": records}
    (OUT / "threshold_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    highlights = {0, .3, .35, .4, .45, .5, .55, balanced, .6, .65, strict, .7, .75, .8, .85, .9, 1}
    headline_rows = [r for r in records if r["threshold"] in highlights]
    lines = [
        "# 六月尖峰：不同機率門檻的 Precision 與 Recall", "",
        "缺車主模型，沿用凍結的 LightGBM 與 Platt 校正。未重訓、未重新校正、未修改現行門檻。", "",
        "## 評估口徑", "",
        "- 2026 年 6 月，決策小時為 7、8、9、16、17、18、19，包含週末。實際資料每半小時一筆。",
        "- 當下為零車且有空柱，依原有六月 eligibility 規則排除未知缺車起點等不合格資料。",
        "- 有效母體 27,962 筆站點 × 決策時間，30 分鐘後仍零車 11,921 筆。",
        "- 對全市符合條件的案例合併計算，不是每日百分比的平均；沒有 Top-K 上限。",
        "- 警示條件為校正後機率 ≥ 門檻。Precision = TP / (TP + FP)；Recall = TP / 11,921。",
        "- 警示筆數不是獨立站點數、缺車事件數或實際派車次數；同站不同時點可重複計入。",
        "- 圖表每 1 個百分點計算一次，另外加入兩個原始凍結門檻；表內只展示代表門檻。", "",
        "## 門檻比較", "",
        "| 門檻 | Precision | Recall | 警示筆數 | 命中 TP | 誤報 FP | 漏報 FN |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in headline_rows:
        label = f"{r['threshold']:.2%}"
        if r["policy"] != "comparison":
            label += "（現行平衡）" if r["policy"] == "balanced" else "（現行嚴格）"
        precision = f"{r['precision']:.2%}" if r["precision"] is not None else "無警示，未定義"
        lines.append(f"| {label} | {precision} | {r['recall']:.2%} | {r['selected']:,} | {r['tp']:,} | {r['fp']:,} | {r['fn']:,} |")
    lines.extend([
        "", "## 解讀與限制", "",
        "- 現行平衡門檻選出 3,849 筆、命中 2,534 筆；嚴格門檻選出 1,218 筆、命中 835 筆。",
        "- 提高門檻會縮小警示集合，Recall 不會提高；Precision 並不保證逐步提高。",
        "- 80% 門檻只剩 56 筆、85% 只剩 7 筆，不宜根據小樣本宣稱穩定的高 Precision。",
        "- 沒有任何警示時，Precision 未定義，不能當作 0% 或 100%；Recall 為 0%。",
        "- 六月為已檢視的回溯資料。本次是事後門檻敏感度分析，不是用六月重選政策，也不據此宣稱某門檻最佳。",
        "- 預測的是 t+30 的零車快照，不代表中間 30 分鐘連續零車；未估計調度介入的因果效果。",
        "- 本圖為觀察值，沒有信賴區間；同站與相鄰時點可能相關。", "",
        "## 核對", "",
        f"- 重跑 truth-free input 的凍結推論，27,962 筆機率與既有 reference 完全一致；最大差值 {diff}。",
        "- 兩種現行政策的逐筆分類完全一致，混淆矩陣與直接計數交叉核對通過。",
        "- 模型與輸入的 SHA-256、完整 1% 步長數據保留於 threshold_metrics.json。", "",
        f"![Precision、Recall 與門檻]({(OUT / 'peak_threshold_precision_recall.png').as_posix()})", "",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"metadata": meta, "highlights": headline_rows}, ensure_ascii=False, indent=2))


def plot() -> None:
    import html
    import subprocess
    import xml.etree.ElementTree as ET

    result = json.loads((OUT / "threshold_metrics.json").read_text(encoding="utf-8"))
    rows = [r for r in result["rows"] if r["threshold"] <= .9]
    w, h = 1400, 980
    left, right, top, bottom = 130, 1335, 185, 610
    vtop, vbottom = 670, 830
    blue, orange, ink = "#2463A0", "#BD6500", "#384656"
    elements = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-labelledby="title description">',
                '<title id="title">六月尖峰時段：不同機率門檻的 Precision、Recall 與警示數</title>',
                '<desc id="description">缺車主模型，27,962 筆案例；現行平衡門檻 Precision 65.84%、Recall 21.26%；現行嚴格門檻 Precision 68.56%、Recall 7.00%。</desc>',
                f'<rect width="{w}" height="{h}" fill="white"/>',
                '<g font-family="Microsoft JhengHei, sans-serif">']

    def text(x, y, value, size=21, anchor="start", weight="normal", color=ink):
        elements.append(f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" fill="{color}">{html.escape(str(value))}</text>')

    def line(x1, y1, x2, y2, color="#DCE2E8", width=1.2, dash=None):
        dashed = f' stroke-dasharray="{dash}"' if dash else ""
        elements.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width}"{dashed}/>')

    def point(x, y, color, square=False):
        if square:
            elements.append(f'<rect x="{x-5:.2f}" y="{y-5:.2f}" width="10" height="10" fill="{color}"/>')
        else:
            elements.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{color}"/>')

    def sx(t):
        return left + t / .9 * (right - left)

    def sy(value):
        return bottom - value * (bottom - top)

    def vy(value):
        return vbottom - value / 30_000 * (vbottom - vtop)

    def curve(field, scale, color, dash=None):
        path = []
        pen_down = False
        for r in rows:
            if r[field] is None:
                pen_down = False
                continue
            path.append(f'{"L" if pen_down else "M"}{sx(r["threshold"]):.3f},{scale(r[field]):.3f}')
            pen_down = True
        dashed = f' stroke-dasharray="{dash}"' if dash else ""
        elements.append(f'<path d="{" ".join(path)}" fill="none" stroke="{color}" stroke-width="3.3" stroke-linejoin="round"{dashed}/>')

    text(left, 48, "尖峰時段：門檻提高，Precision 與 Recall 如何變化？", size=31, weight="bold")
    text(left, 91, "缺車主模型｜2026 年 6 月回溯｜全市合併、不限制 Top-K", size=22)
    text(left, 127, "07:00–09:59、16:00–19:59（含週末）｜27,962 筆案例，11,921 筆實際持續缺車", size=19, color="#526170")
    for ya, yb in [(top, bottom), (vtop, vbottom)]:
        elements.append(f'<rect x="{sx(.8):.2f}" y="{ya}" width="{right-sx(.8):.2f}" height="{yb-ya}" fill="#EDF0F3"/>')
        line(left, ya, left, yb, "#9AA6B2")
        line(left, yb, right, yb, "#9AA6B2")
    for i in range(6):
        value = i / 5
        line(left, sy(value), right, sy(value))
        text(left - 18, sy(value) + 7, f"{value:.0%}", anchor="end")
    text(left, top - 22, "Precision / Recall", size=21)
    for value in [0, 10_000, 20_000, 30_000]:
        line(left, vy(value), right, vy(value))
        text(left - 18, vy(value) + 7, f"{value:,}", anchor="end", size=19)
    text(left, vtop - 22, "警示筆數", size=21)
    for i in range(10):
        value = i / 10
        line(sx(value), vbottom, sx(value), vbottom + 6, "#9AA6B2")
        text(sx(value), vbottom + 34, f"{value:.0%}", anchor="middle")
    text((left + right) / 2, 904, "校正後機率門檻（預測機率 ≥ 門檻才列入警示）", size=22, anchor="middle")

    for policy, label, label_y in [("balanced", "現行平衡", 217), ("strict", "現行嚴格", 270)]:
        r = next(r for r in rows if r["policy"] == policy)
        x = sx(r["threshold"])
        for ya, yb in [(top, bottom), (vtop, vbottom)]:
            line(x, ya, x, yb, "#8794A0", 1.4, "3 5")
        text(x - 10, label_y, label, anchor="end", size=19)
        text(x - 10, label_y + 27, f"{r['threshold']:.2%}", anchor="end", size=19)

    curve("precision", sy, blue)
    curve("recall", sy, orange, "9 6")
    curve("selected", vy, ink)
    line(158, 329, 202, 329, blue, 3.3)
    text(215, 336, "Precision 精確率", size=22)
    line(158, 368, 202, 368, orange, 3.3, "9 6")
    text(215, 375, "Recall 召回率", size=22)
    text(right - 12, 274, "≥ 80%", anchor="end", size=19)
    text(right - 12, 301, "警示 ≤ 56 筆", anchor="end", size=18)
    for policy in ["balanced", "strict"]:
        r = next(r for r in rows if r["policy"] == policy)
        x = sx(r["threshold"])
        point(x, sy(r["precision"]), blue)
        point(x, sy(r["recall"]), orange, square=True)
        point(x, vy(r["selected"]), ink)
        text(x - 11, sy(r["precision"]) + 48, f"{r['precision']:.2%}", anchor="end", size=20)
        text(x + 12, sy(r["recall"]) - 18, f"{r['recall']:.2%}", anchor="start", size=20)
        text(x - 11, vy(r["selected"]) - 15, f"{r['selected']:,}", anchor="end", size=20)
    text(left, 945, "單位：站點 × 決策時間；同站可重複。目標為 t+30 的零車快照，不代表期間連續零車。", size=17, color="#526170")
    text(left, 972, "事後敏感度分析，未重選門檻。無警示時 Precision 未定義，曲線留白；本圖未呈現信賴區間。", size=17, color="#526170")
    elements.append("</g></svg>")
    svg = "\n".join(elements)
    ET.fromstring(svg)
    svg_path = OUT / "peak_threshold_precision_recall.svg"
    png_path = OUT / "peak_threshold_precision_recall.png"
    svg_path.write_text(svg, encoding="utf-8")
    node = "C:/Users/a5140/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe"
    sharp = "C:/Users/a5140/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/sharp/dist/index.cjs"
    js = "const sharp=require(process.argv[1]); sharp(process.argv[2],{density:144}).png().toFile(process.argv[3]).then(x=>console.log(JSON.stringify(x))).catch(e=>{console.error(e);process.exit(1)});"
    subprocess.run([node, "-e", js, sharp, str(svg_path), str(png_path)], check=True)
    print(str(png_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if args.analyze:
        analyze()
    if args.plot:
        plot()
