"""
A5 · Kiểm định thống kê nghiêm ngặt từ results/raw.csv (không cần chạy lại mô phỏng).

Với mỗi (kịch bản, mức tải, sức chứa, cấu hình ≠ P0): so cặp theo seed với P0.
  - Kiểm định chính: t-test cặp đôi trên log(AWT) (hiệu ứng dạng tỷ lệ), hiệu chỉnh Holm–Bonferroni cho TOÀN BỘ
    phép so sánh. (Wilcoxon với 10 seed có p nhỏ nhất ≈ 0,002 nên không thể qua Holm khi có 288 phép so sánh.)
  - Bổ sung: Wilcoxon (không tham số, chưa hiệu chỉnh), số seed cấu hình tốt hơn P0, kích thước hiệu ứng
    rank-biserial (-1…1) và mức giảm AWT trung vị (%).
Chạy:  python -m experiments.stats
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata, ttest_rel, wilcoxon

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def rank_biserial(x, y):
    d = np.asarray(x) - np.asarray(y)
    d = d[d != 0]
    if not len(d):
        return 0.0
    r = rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def main():
    rows = list(csv.DictReader(open(os.path.join(HERE, "results", "raw.csv"), encoding="utf-8")))
    by = {}
    for r in rows:
        by.setdefault((r["scenario"], r["level"], r["capacity"], r["config"]), {})[int(r["seed"])] = float(r["AWT"])
    tests = []
    for (s, lv, cap, cfg), v in by.items():
        if cfg == "P0":
            continue
        base = by[(s, lv, cap, "P0")]
        seeds = sorted(set(v) & set(base))
        x = [base[k] for k in seeds]
        y = [v[k] for k in seeds]
        try:
            pw = float(wilcoxon(x, y).pvalue) if any(a != b for a, b in zip(x, y)) else 1.0
        except ValueError:
            pw = 1.0
        lx, ly = np.log(np.maximum(x, 0.1)), np.log(np.maximum(y, 0.1))
        p = float(ttest_rel(lx, ly).pvalue) if np.ptp(lx - ly) > 0 else (0.0 if (lx - ly).mean() != 0 else 1.0)
        imp = [(a - b) / a * 100 for a, b in zip(x, y) if a > 0]
        tests.append({"scenario": s, "level": lv, "capacity": int(cap), "config": cfg, "n": len(seeds), "p": p,
                      "p_wilcoxon": pw, "wins": int(sum(1 for a, b in zip(x, y) if b < a)),
                      "effect_rb": rank_biserial(x, y), "median_improve_pct": float(np.median(imp)) if imp else 0.0})
    adj = holm(np.array([t["p"] for t in tests]))
    for t, a in zip(tests, adj):
        t["p_holm"] = float(a)
        t["significant"] = bool(a < 0.05)
    summary = {}
    for cfg in sorted({t["config"] for t in tests}):
        ts = [t for t in tests if t["config"] == cfg]
        summary[cfg] = {"comparisons": len(ts), "significant_better": sum(1 for t in ts if t["significant"] and t["effect_rb"] > 0),
                        "significant_worse": sum(1 for t in ts if t["significant"] and t["effect_rb"] < 0),
                        "all_seeds_better": sum(1 for t in ts if t["wins"] == t["n"]),
                        "median_improve_pct": float(np.median([t["median_improve_pct"] for t in ts]))}
    with open(os.path.join(HERE, "results", "stats.json"), "w", encoding="utf-8") as fh:
        json.dump({"tests": tests, "summary": summary, "n_tests": len(tests)}, fh, indent=1, ensure_ascii=False)
    lines = ["# Kiểm định thống kê (t-test cặp đôi trên log AWT + hiệu chỉnh Holm)", "",
             f"{len(tests)} phép so sánh với P0 (cùng seed = cùng hành khách). Hiệu ứng = rank-biserial (1 = tốt hơn P0 ở mọi seed).", "",
             "| Cấu hình | Số phép so sánh | Tốt hơn có ý nghĩa (Holm) | Tệ hơn có ý nghĩa | Tốt hơn ở mọi seed | Giảm AWT trung vị |",
             "|---|---|---|---|---|---|"]
    for cfg, v in summary.items():
        lines.append(f"| {cfg} | {v['comparisons']} | {v['significant_better']} | {v['significant_worse']} | "
                     f"{v['all_seeds_better']} | {v['median_improve_pct']:.0f} % |")
    lines += ["", "Wilcoxon (không tham số) được ghi kèm trong stats.json; với 10 seed p nhỏ nhất của nó ≈ 0,002 nên",
              "không đủ để qua hiệu chỉnh Holm cho 288 phép so sánh – vì vậy dùng t-test trên log AWT làm kiểm định chính."]
    with open(os.path.join(HERE, "results", "stats.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
