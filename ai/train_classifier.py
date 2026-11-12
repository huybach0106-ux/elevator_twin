"""
Huấn luyện bộ nhận diện chế độ giao thông (Random Forest) từ dữ liệu MÔ PHỎNG có nhiễu cảm biến.

Mỗi 20 s trong mỗi lượt chạy: đặc trưng = window_features(quan sát cảm biến cabin, nút gọi tầng),
nhãn = chế độ THẬT tính từ OD thật trong cửa sổ.  Tách tập theo seed (train ≠ test) để tránh rò rỉ.
Chạy:  python -m ai.train_classifier
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.features import FEATURE_NAMES, MODES, rule_mode, true_mode, window_features  # noqa: E402
from config import OPT_PARAMS  # noqa: E402
from simulation.engine import Simulation  # noqa: E402
from simulation.runner import make_dispatcher  # noqa: E402
from simulation.scenarios import LEVELS, SCENARIOS, make_passengers  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = OPT_PARAMS["window"]


class Recorder:
    """Bọc một dispatcher, ghi (đặc trưng, nhãn) mỗi 20 s."""

    def __init__(self, inner, every=20.0):
        self.inner, self.every, self.rows = inner, every, []
        self.update_interval = inner.update_interval
        self._last = 0.0

    def __getattr__(self, k):
        return getattr(self.inner, k)

    def attach(self, sim):
        self.sim = sim
        self.inner.attach(sim)

    def update(self, t):
        if t - self._last >= self.every and t > 0:
            self._last = t
            x = window_features(self.sim.obs, t, W, self.sim.hall_calls())
            self.rows.append((x, true_mode(self.sim.passengers, t, W, OPT_PARAMS["offpeak_rate"])))
        self.inner.update(t)


def collect(seeds, configs=("P0", "P2")):
    X, y = [], []
    for s, sc in SCENARIOS.items():
        for lv in LEVELS:
            for seed in seeds:
                for cfg in configs:
                    ps = make_passengers(s, lv, seed)
                    rec = Recorder(make_dispatcher(cfg, s, lv, seed, use_model=False))
                    Simulation(ps, rec, capacity=15 if seed % 2 else 10, horizon=sc["horizon"],
                               seed=seed, record=False).run()
                    for xi, yi in rec.rows:
                        X.append(xi)
                        y.append(yi)
    return np.array(X), np.array(y)


def main():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import joblib

    Xtr, ytr = collect(range(1, 9))
    Xte, yte = collect(range(101, 105))
    print("train", Xtr.shape, "test", Xte.shape)
    clf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=2,
                                 class_weight="balanced", random_state=0, n_jobs=-1)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    rule = np.array([rule_mode(x, OPT_PARAMS["offpeak_rate"]) for x in Xte])
    labels = [m for m in MODES if m in set(yte) | set(pred)]
    rep = {
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "acc_rf": float(accuracy_score(yte, pred)),
        "acc_rule": float(accuracy_score(yte, rule)),
        "labels": labels,
        "confusion_rf": confusion_matrix(yte, pred, labels=labels).tolist(),
        "confusion_rule": confusion_matrix(yte, rule, labels=labels).tolist(),
        "feature_importance": dict(sorted(zip(FEATURE_NAMES, map(float, clf.feature_importances_)),
                                          key=lambda kv: -kv[1])),
    }
    print(classification_report(yte, pred, zero_division=0))
    print(f"Độ chính xác RF = {rep['acc_rf']:.3f} | luật ngưỡng = {rep['acc_rule']:.3f}")
    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    joblib.dump(clf, os.path.join(HERE, "models", "mode_rf.joblib"), compress=("xz", 6))
    with open(os.path.join(HERE, "results", "classifier_report.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2, ensure_ascii=False)
    print("Đã lưu models/mode_rf.joblib và results/classifier_report.json")


if __name__ == "__main__":
    main()
