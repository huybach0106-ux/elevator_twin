"""
Bộ nhận diện chế độ giao thông (UP_PEAK / DOWN_PEAK / TWO_WAY / INTERFLOOR / OFF_PEAK)
từ quan sát cảm biến cabin + nút gọi tầng.

- Nếu có models/mode_rf.joblib (huấn luyện bằng ai/train_classifier.py) -> Random Forest.
- Nếu chưa có -> luật ngưỡng (ai/features.rule_mode).
"""
from __future__ import annotations

import os

from ai.features import MODES, rule_mode

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(_HERE, "models", "mode_rf.joblib")
_CACHE = {}


def _load():
    if "model" not in _CACHE:
        model = None
        if os.path.exists(MODEL_PATH):
            try:
                import joblib
                model = joblib.load(MODEL_PATH)
                model.n_jobs = 1   # dự đoán từng mẫu: tắt song song để nhanh
            except Exception:  # noqa: BLE001
                model = None
        _CACHE["model"] = model
    return _CACHE["model"]


class ModeDetector:
    def __init__(self, use_model: bool = True, offpeak_rate: float = 7.2):
        self.model = _load() if use_model else None
        self.offpeak_rate = offpeak_rate
        self.kind = "RandomForest" if self.model is not None else "Luật ngưỡng"

    def predict(self, x) -> str:
        if self.model is not None:
            return str(self.model.predict(x.reshape(1, -1))[0])
        return rule_mode(x, self.offpeak_rate)

    def proba(self, x) -> dict:
        if self.model is None:
            m = self.predict(x)
            return {k: float(k == m) for k in MODES}
        pr = self.model.predict_proba(x.reshape(1, -1))[0]
        return {str(c): float(p) for c, p in zip(self.model.classes_, pr)}
