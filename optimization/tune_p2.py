"""
Dò tham số điều phối P2 bằng Optuna (TPE).

Hàm mục tiêu (càng nhỏ càng tốt), trung bình trên 8 tình huống × mức tải trung bình/cao × seed huấn luyện:
    J = (AWT + 0,5·P95 + 0,25·AWT_tầng_tệ_nhất) / (cùng đại lượng của P0)  +  0,1 · Năng lượng / Năng lượng_P0
Kiểm định trên seed KHÁC (11–15); chỉ ghi data/tuned_params.json nếu tốt hơn bộ mặc định.
Chạy:  python -m optimization.tune_p2 --trials 40
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import OPT_PARAMS  # noqa: E402
from simulation.runner import run_one  # noqa: E402
from simulation.scenarios import SCENARIOS  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEVELS = ("mid", "high")


def _job(args):
    s, lv, seed, cfg, params = args
    _, m = run_one(s, lv, 15, cfg, seed, params=params)
    return (s, lv, seed), m


def _service(m):
    return m["AWT"] + 0.5 * m["P95"] + 0.25 * m["worst_floor_awt"]


def evaluate(params, seeds, ref, pool):
    jobs = [(s, lv, sd, "P2", params) for s in SCENARIOS for lv in LEVELS for sd in seeds]
    vals = []
    for key, m in pool.map(_job, jobs):
        r = ref[key]
        vals.append(_service(m) / max(1e-6, _service(r)) + 0.1 * m["energy"] / max(1e-6, r["energy"]))
    return float(np.mean(vals))


def references(seeds, pool):
    jobs = [(s, lv, sd, "P0", None) for s in SCENARIOS for lv in LEVELS for sd in seeds]
    return dict(pool.map(_job, jobs))


def main():
    import optuna

    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    a = ap.parse_args()
    train_seeds, val_seeds = [1, 2, 3], [11, 12, 13, 14, 15]
    defaults = {k: OPT_PARAMS[k] for k in ["replan_interval", "min_hold", "improve_ratio", "window",
                                           "min_floor_share", "fallback_wait", "pool_util", "w_load",
                                           "w_role", "w_stops", "wake_wait"]}
    with Pool(max(1, os.cpu_count() or 1)) as pool:
        ref_tr = references(train_seeds, pool)
        history = []

        def objective(trial):
            p = {
                "replan_interval": trial.suggest_float("replan_interval", 10, 40),
                "min_hold": trial.suggest_float("min_hold", 10, 90),
                "improve_ratio": trial.suggest_float("improve_ratio", 0.05, 0.4),
                "window": trial.suggest_float("window", 60, 240),
                "min_floor_share": trial.suggest_float("min_floor_share", 0.02, 0.15),
                "fallback_wait": trial.suggest_float("fallback_wait", 30, 120),
                "pool_util": trial.suggest_float("pool_util", 0.2, 1.2),
                "w_load": trial.suggest_float("w_load", 0, 50),
                "w_role": trial.suggest_float("w_role", 0, 40),
                "w_stops": trial.suggest_float("w_stops", 0, 10),
                "wake_wait": trial.suggest_float("wake_wait", 15, 60),
            }
            v = evaluate(p, train_seeds, ref_tr, pool)
            history.append({"trial": trial.number, "value": v})
            print(f"trial {trial.number:3d}  J = {v:.4f}", flush=True)
            return v

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
        study.enqueue_trial(defaults)
        study.optimize(objective, n_trials=a.trials)
        best = study.best_params
        ref_va = references(val_seeds, pool)
        j_def = evaluate(defaults, val_seeds, ref_va, pool)
        j_best = evaluate(best, val_seeds, ref_va, pool)
    print(f"Kiểm định (seed mới): mặc định J = {j_def:.4f} | tối ưu J = {j_best:.4f}")
    report = {"best_params": best, "train_best": study.best_value, "val_default": j_def, "val_tuned": j_best,
              "history": history, "importance": {}}
    try:
        report["importance"] = {k: float(v) for k, v in optuna.importance.get_param_importances(study).items()}
    except Exception:  # noqa: BLE001
        pass
    with open(os.path.join(HERE, "results", "optuna_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    if j_best < j_def:
        with open(os.path.join(HERE, "data", "tuned_params.json"), "w", encoding="utf-8") as fh:
            json.dump(best, fh, indent=2)
        print("Đã ghi data/tuned_params.json (mô phỏng P2 sẽ dùng bộ tham số này)")
    else:
        print("Bộ tối ưu không tốt hơn trên seed kiểm định -> giữ tham số mặc định")


if __name__ == "__main__":
    main()
