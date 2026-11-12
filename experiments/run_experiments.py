"""
Thí nghiệm theo mục 7 tài liệu: 8 tình huống × 3 mức tải × 2 sức chứa (10, 15) × 4 cấu hình (P0–P3)
× nhiều seed (cùng seed = cùng danh sách hành khách cho mọi cấu hình).

Đầu ra:
  results/raw.csv            mỗi dòng = 1 lượt chạy
  results/summary.csv        trung bình ± khoảng tin cậy 95 % theo (tình huống, tải, sức chứa, cấu hình)
  results/summary.md         bảng kết quả kiểu mục 10 + % cải thiện so với P0 (+ kiểm định Wilcoxon)
  results/ablation.csv       P2 bỏ từng thành phần (cảm biến hoàn hảo, không thời khóa biểu, luật thay RF)
Chạy:  python -m experiments.run_experiments --seeds 10
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.metrics import improvement  # noqa: E402
from simulation.runner import run_one  # noqa: E402
from simulation.scenarios import LEVELS, SCENARIOS, n_for  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
KPIS = ["AWT", "AWT_served", "P90", "P95", "max_wait", "pct_wait_gt60", "ATT", "total_time", "throughput",
        "backlog", "bypass", "stops", "useless_stops", "energy", "energy_per_pax", "RTT", "clearance",
        "worst_floor_awt", "jain", "role_changes"]
LOWER_BETTER = {k: True for k in KPIS}
LOWER_BETTER.update(throughput=False, jain=False)


def _job(a):
    s, lv, cap, cfg, seed, kw = a
    _, m = run_one(s, lv, cap, cfg, seed, **kw)
    row = {"scenario": s, "level": lv, "N": n_for(s, lv), "capacity": cap, "config": cfg, "seed": seed}
    for k in KPIS:
        v = m[k]
        row[k] = float("nan") if v is None else v
    row["A1_pax_share"] = m["cars"]["A1"]["pax"] / max(1, sum(c["pax"] for c in m["cars"].values()))
    return row


def ci95(a):
    a = np.asarray([x for x in a if x == x])
    if len(a) < 2:
        return float(a.mean()) if len(a) else float("nan"), 0.0
    return float(a.mean()), float(1.96 * a.std(ddof=1) / np.sqrt(len(a)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--quick", action="store_true", help="chỉ mức tải trung bình, sức chứa 15")
    a = ap.parse_args()
    seeds = list(range(1, a.seeds + 1))
    levels = ["mid"] if a.quick else list(LEVELS)
    caps = [15] if a.quick else [10, 15]
    configs = ["P0", "P1", "P2", "P3", "P4"]
    jobs = [(s, lv, cap, cfg, sd, {}) for s in SCENARIOS for lv in levels for cap in caps
            for cfg in configs for sd in seeds]
    t0 = time.time()
    with Pool(os.cpu_count() or 1) as pool:
        rows = pool.map(_job, jobs, chunksize=4)
        abl = [
            ("P2", {}), ("P2 cảm biến hoàn hảo", {"sensor_noise": False}),
            ("P2 không thời khóa biểu", {"use_prior": False}),
            ("P2 luật thay RF", {"use_model": False}),
            ("P2 không TKB + luật", {"use_prior": False, "use_model": False}),
        ]
        abl_rows = []
        for name, kw in abl:
            rs = pool.map(_job, [(s, "high", 15, "P2", sd, kw) for s in SCENARIOS for sd in seeds])
            for r in rs:
                r["variant"] = name
            abl_rows += rs
    print(f"{len(rows) + len(abl_rows)} lượt chạy trong {time.time() - t0:.0f} s")
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "raw.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(RES, "ablation.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(abl_rows[0].keys()))
        w.writeheader()
        w.writerows(abl_rows)

    # ---------- tổng hợp
    from scipy.stats import wilcoxon  # noqa: WPS433
    groups = {}
    for r in rows:
        groups.setdefault((r["scenario"], r["level"], r["capacity"], r["config"]), []).append(r)
    summary = []
    for (s, lv, cap, cfg), rs in groups.items():
        rec = {"scenario": s, "level": lv, "N": rs[0]["N"], "capacity": cap, "config": cfg, "n_seeds": len(rs)}
        for k in KPIS + ["A1_pax_share"]:
            m, h = ci95([r[k] for r in rs])
            rec[k] = m
            rec[k + "_ci"] = h
        base = groups[(s, lv, cap, "P0")]
        for k in ("AWT", "P95", "energy"):
            rec[k + "_impr"] = improvement(np.mean([r[k] for r in base]), rec[k])
        if cfg != "P0":
            x = [r["AWT"] for r in sorted(base, key=lambda r: r["seed"])]
            y = [r["AWT"] for r in sorted(rs, key=lambda r: r["seed"])]
            try:
                rec["p_AWT"] = float(wilcoxon(x, y).pvalue) if any(a != b for a, b in zip(x, y)) else 1.0
            except ValueError:
                rec["p_AWT"] = float("nan")
        summary.append(rec)
    with open(os.path.join(RES, "summary.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()) + ["p_AWT"])
        w.writeheader()
        w.writerows(summary)
    with open(os.path.join(RES, "summary.json"), "w", encoding="utf-8") as fh:
        clean = lambda v: None if isinstance(v, float) and v != v else v  # noqa: E731
        json.dump({"summary": [{k: clean(v) for k, v in r.items()} for r in summary],
                   "ablation": [{k: clean(v) for k, v in r.items()} for r in abl_rows], "seeds": seeds},
                  fh, ensure_ascii=False, allow_nan=False)

    # ---------- bảng markdown (mục 10)
    lines = ["# Kết quả thực nghiệm", "",
             f"Seed 1–{a.seeds}; giá trị = trung bình ± nửa khoảng tin cậy 95 %. "
             "AWT tính cả khách chưa được đón (chờ bị cắt tại thời điểm kết thúc). "
             "p = kiểm định Wilcoxon cặp đôi so với P0 (cùng seed, cùng hành khách).", ""]
    for cap in caps:
        lines += [f"## Sức chứa hữu dụng {cap} người/chuyến", "",
                  "| Kịch bản | Tải | N | Cấu hình | AWT (s) | P95 (s) | Tầng tệ nhất (s) | Thông lượng | Backlog | Số dừng | Bypass | Năng lượng (EU) | ΔAWT vs P0 | p |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for s in SCENARIOS:
            for lv in levels:
                for cfg in configs:
                    r = next(x for x in summary if (x["scenario"], x["level"], x["capacity"], x["config"]) == (s, lv, cap, cfg))
                    imp = "" if cfg == "P0" or r["AWT_impr"] is None else f"{r['AWT_impr']:+.0f} %"
                    p = r.get("p_AWT")
                    ps = "" if cfg == "P0" or p is None or p != p else ("<0,01" if p < 0.01 else f"{p:.2f}")
                    lines.append(
                        f"| {s} | {lv} | {r['N']} | {cfg} | {r['AWT']:.1f} ± {r['AWT_ci']:.1f} | {r['P95']:.0f} | "
                        f"{r['worst_floor_awt']:.0f} | {r['throughput']:.0f} | {r['backlog']:.1f} | {r['stops']:.0f} | "
                        f"{r['bypass']:.1f} | {r['energy']:.0f} | {imp} | {ps} |")
        lines.append("")
    # ablation
    lines += ["## Ablation – đóng góp từng thành phần của P2 (tải cao, sức chứa 15)", "",
              "| Biến thể | " + " | ".join(SCENARIOS) + " |", "|---|" + "---|" * len(SCENARIOS)]
    for name, _ in abl:
        cells = []
        for s in SCENARIOS:
            v = [r["AWT"] for r in abl_rows if r["variant"] == name and r["scenario"] == s]
            cells.append(f"{np.mean(v):.1f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    with open(os.path.join(RES, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("Đã ghi results/raw.csv, summary.csv, summary.md, ablation.csv")


if __name__ == "__main__":
    main()
