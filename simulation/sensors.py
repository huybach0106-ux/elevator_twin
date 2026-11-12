"""
Cảm biến thang máy thông thường – lớp cảm biến của Digital Twin (KHÔNG dùng camera).

  - Cân tải trong cabin: đo kg -> ước số người = kg / 65 (mỗi người nặng khác nhau -> có sai số).
  - Cảm biến hồng ngoại ở cửa (thanh tia an toàn): đếm lượt đi qua cửa; hai người đi sát nhau có thể bị tính là một.
  - Nút gọi tầng: sau khi cửa đóng, nếu còn người chờ thì nút được bấm lại -> hệ thống chỉ biết CÓ / KHÔNG,
    không biết bao nhiêu người.
Bộ điều phối tối ưu chỉ đọc các quan sát này, không đọc dữ liệu thật của mô phỏng.
noise=False -> cảm biến hoàn hảo (dùng cho ablation).
"""
from __future__ import annotations

import numpy as np

from config import AVG_PAX_KG, SENSORS


class CabinSensors:
    def __init__(self, seed: int = 0, noise: bool = True, params: dict | None = None):
        self.rng = np.random.default_rng(seed + 7777)
        self.noise = noise
        self.p = dict(SENSORS, **(params or {}))

    def load(self, true_n: int) -> int:
        """Cân tải: tổng khối lượng / khối lượng trung bình -> số người ước tính."""
        if not self.noise or true_n == 0:
            return int(true_n)
        kg = self.rng.normal(AVG_PAX_KG, self.p["kg_sd"], true_n).sum() + self.rng.normal(0, self.p["scale_sd"])
        return max(0, int(round(kg / AVG_PAX_KG)))

    def door_count(self, true_n: int) -> int:
        """Cảm biến hồng ngoại ở cửa đếm lượt qua cửa."""
        if not self.noise:
            return int(true_n)
        return int(self.rng.binomial(true_n, self.p["door_detect"]))

    def landing(self, true_n: int) -> int:
        """Nút gọi tầng còn sáng sau khi đóng cửa: chỉ biết còn người chờ (1) hay không (0)."""
        return 1 if true_n > 0 else 0
