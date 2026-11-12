"""
Cấu hình mô hình Digital Twin thang máy – 13 tầng, 6 cabin.

Mọi giá trị dưới đây là THAM SỐ MÔ PHỎNG (giả định / tham chiếu), không phải số đo thực tế.
Khi có số liệu khảo sát (bấm giờ tại cabin, đếm người), ghi đè trong
data/calibration.json – file này sẽ được nạp tự động ở cuối module.
"""
import json
import os

# ---------------------------------------------------------------------------
# Tầng
# ---------------------------------------------------------------------------
FLOOR_NAMES = ["B2", "B1", "G"] + [str(i) for i in range(1, 11)]
N_FLOORS = len(FLOOR_NAMES)                 # 13
IDX = {name: i for i, name in enumerate(FLOOR_NAMES)}
G = IDX["G"]                                # chỉ số tầng sảnh = 2
LOBBY = {IDX["B2"], IDX["B1"], G}           # sảnh + tầng hầm (đầu mối ra/vào tòa nhà)
UPPER = [IDX[str(i)] for i in range(1, 11)]  # tầng 1..10


def F(name) -> int:
    """Tên tầng -> chỉ số. F('G') = 2, F(8) = 10."""
    return IDX[str(name)]


def fname(idx: int) -> str:
    return FLOOR_NAMES[idx]


# ---------------------------------------------------------------------------
# Động học & thời gian (giây)
# ---------------------------------------------------------------------------
TRAVEL_TIME_PER_FLOOR = 2.2   # s/tầng (giả định, theo tài liệu)
DOOR_TIME = 4.0               # s mỗi lần dừng: mở + đóng cửa
TRANSFER_TIME = 0.6           # s mỗi hành khách lên hoặc xuống
STANDBY_WAKE_TIME = 3.0       # s để cabin standby sẵn sàng chạy lại
DT = 0.1                      # bước mô phỏng
LOG_EVERY = 1.0               # chu kỳ ghi khung hình cho trực quan hóa

# ---------------------------------------------------------------------------
# Sức chứa & tải trọng
# ---------------------------------------------------------------------------
RATED_CAPACITY = 20           # sức chứa danh định (người)
RATED_LOAD_KG = 1350
AVG_PAX_KG = 65               # sinh viên + cặp sách (giả định)
USEFUL_CAPACITY = 15          # sức chứa hữu dụng mặc định (thí nghiệm dùng 10 và 15)
# Cảm biến cân tải truyền thống chỉ báo "đầy" khi vượt 80 % tải định mức (theo kg).
# Với sức chứa hữu dụng 10–15 người, cân tải KHÔNG bao giờ báo đầy -> cabin vẫn dừng vô ích.
WEIGHT_BYPASS_RATIO = 0.8

# ---------------------------------------------------------------------------
# Đội thang (A1..A6). baseline_stops = cấu hình gốc P0.
# reach = các tầng cabin CÓ THỂ phục vụ nếu đổi tuyến (giả định mô phỏng!).
# Nếu giếng thang A4–A6 thực tế chỉ lên tới tầng 8, sửa reach tương ứng.
# ---------------------------------------------------------------------------
_ALL = list(range(N_FLOORS))
CARS = [
    {"id": "A1", "alias": "X",    "baseline": ["B2", "B1", "G"] + [str(i) for i in range(1, 11)], "reach": _ALL, "home": "G"},
    {"id": "A2", "alias": "G5-A", "baseline": ["B2", "B1", "G", "5"], "reach": _ALL, "home": "G"},
    {"id": "A3", "alias": "G5-B", "baseline": ["B2", "B1", "G", "5"], "reach": _ALL, "home": "G"},
    {"id": "A4", "alias": "G8-A", "baseline": ["G", "8"], "reach": [F(x) for x in ["G"] + [str(i) for i in range(1, 11)]], "home": "G"},
    {"id": "A5", "alias": "G8-B", "baseline": ["G", "8"], "reach": [F(x) for x in ["G"] + [str(i) for i in range(1, 11)]], "home": "G"},
    {"id": "A6", "alias": "G8-C", "baseline": ["G", "8"], "reach": [F(x) for x in ["G"] + [str(i) for i in range(1, 11)]], "home": "G"},
]
LOCAL_CAR = "A1"              # luôn giữ ít nhất 1 cabin Local phục vụ toàn bộ tầng

# ---------------------------------------------------------------------------
# Mô hình năng lượng tương đối (đơn vị EU – KHÔNG phải kWh)
# ---------------------------------------------------------------------------
E_FLOOR = 1.0                 # năng lượng mỗi tầng di chuyển khi cân bằng
E_IMBALANCE = 1.0             # hệ số tăng theo mất cân bằng tải – đối trọng
COUNTERWEIGHT_RATIO = 0.45    # đối trọng = 45 % tải định mức
E_START = 1.5                 # mỗi lần khởi động (tăng tốc)
E_DOOR = 0.1                  # mỗi chu kỳ đóng/mở cửa
# Công suất chờ ~7 % công suất chạy (≈ 0,45 EU/s). Theo VDI 4707, với thang ít dùng,
# năng lượng chờ có thể chiếm phần lớn tổng tiêu thụ -> standby có ý nghĩa ở thấp điểm.
P_IDLE = 0.030                # EU/s khi cabin hoạt động nhưng rảnh (đèn, quạt, bộ điều khiển)
P_STANDBY = 0.006             # EU/s khi standby (tắt đèn/quạt cabin)

# ---------------------------------------------------------------------------
# Cảm biến thang máy thông thường (xem simulation/sensors.py) – không dùng camera
# ---------------------------------------------------------------------------
SENSORS = {
    "kg_sd": 12.0,           # độ lệch chuẩn khối lượng mỗi người (kg) -> sai số ước số người từ cân tải
    "scale_sd": 10.0,        # sai số của cân (kg)
    "door_detect": 0.95,     # xác suất cảm biến hồng ngoại ở cửa ghi nhận một lượt qua cửa
}

# ---------------------------------------------------------------------------
# Tham số điều phối tối ưu (P2) – có thể được Optuna ghi đè (data/tuned_params.json)
# ---------------------------------------------------------------------------
OPT_PARAMS = {
    "replan_interval": 20.0,     # s giữa hai lần tính lại phân vai
    "min_hold": 40.0,            # s tối thiểu cabin giữ một vai (chống dao động)
    "improve_ratio": 0.15,       # chỉ đổi phân vai khi điểm cải thiện ≥ 15 %
    "window": 120.0,             # s cửa sổ quan sát nhu cầu
    "min_floor_share": 0.06,     # tầng có tỷ trọng nhu cầu < 6 % để A1 lo
    "fallback_wait": 60.0,       # s – cuộc gọi chờ quá ngưỡng thì A1 nhận mọi hành trình
    "offpeak_rate": 7.2,         # người/phút = 20 % tải tham chiếu (180 khách / 5 phút) -> Off-Peak
    "standby_keep": 2,           # số cabin vẫn hoạt động khi thấp điểm
    "wake_wait": 35.0,           # s – cuộc gọi chờ quá ngưỡng thì đánh thức cabin standby
    "offpeak_park": None,        # thấp điểm: cabin đứng yên tại chỗ, không chạy về sảnh (tiết kiệm năng lượng)
    "pool_util": 0.55,           # hệ số sử dụng ước tính < ngưỡng -> mọi cabin phục vụ chung (pooling),
                                 # ≥ ngưỡng -> chia Express/Zone (giảm số điểm dừng mỗi vòng)
    "w_eta": 1.0, "w_load": 25.0, "w_role": 15.0, "w_stops": 3.0,
}

# ---------------------------------------------------------------------------
# Ghi đè bằng dữ liệu hiệu chuẩn / tham số đã dò (nếu có)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_CALIB = os.path.join(_HERE, "data", "calibration.json")
_TUNED = os.path.join(_HERE, "data", "tuned_params.json")

if os.path.exists(_CALIB):
    with open(_CALIB, encoding="utf-8") as fh:
        _c = json.load(fh)
    TRAVEL_TIME_PER_FLOOR = _c.get("travel_time_per_floor", TRAVEL_TIME_PER_FLOOR)
    DOOR_TIME = _c.get("door_time", DOOR_TIME)
    TRANSFER_TIME = _c.get("transfer_time", TRANSFER_TIME)
    USEFUL_CAPACITY = _c.get("useful_capacity", USEFUL_CAPACITY)

if os.path.exists(_TUNED):
    with open(_TUNED, encoding="utf-8") as fh:
        OPT_PARAMS.update(json.load(fh))
