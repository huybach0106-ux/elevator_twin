# Digital Twin thang máy – Tòa nhà học 13 tầng, 6 cabin

Mô phỏng và tối ưu điều phối 6 thang máy (A1–A6) theo tài liệu *Digital Twin A2*:
12 kịch bản đã chốt (4 nhóm giao thông × 3), 5 cấu hình P0–P4, 2 mức sức chứa hữu dụng (10 và 15 người), 3 mức tải.
Công nghệ tích hợp: **AI** (nhận diện chế độ giao thông bằng Random Forest, dò tham số bằng Optuna),
**LLM** (trợ lý vận hành tự chạy mô phỏng để trả lời), **Web** (dashboard Flask chạy song song P0 và phương án tối ưu),
**Thời gian thực** (tòa nhà ảo ↔ twin qua MQTT, vòng điều khiển hai chiều, dự báo 10 phút, what-if).
Tài liệu kỹ thuật đầy đủ: mở `http://127.0.0.1:5000/docs` khi chạy web.

> Mọi kết quả là **kết quả mô phỏng** với tham số giả định. Việc đổi tuyến dừng của A2–A6
> là **phương án điều phối giả định**, không khẳng định khả năng can thiệp bộ điều khiển thật.

## Cài đặt & chạy nhanh

```bash
pip install -r requirements.txt
python -m pytest -q                          # 56 kiểm thử
python web_app.py                            # mở http://localhost:5000
```

Tái tạo toàn bộ kết quả (theo thứ tự):

```bash
python -m ai.train_classifier                # ~1 phút  -> models/mode_rf.joblib
python -m optimization.tune_p2 --trials 40   # ~25 phút -> data/tuned_params.json
python -m experiments.run_experiments --seeds 10   # ~6 phút -> results/summary.md
python -m experiments.threshold_study        # ~5 phút  -> results/threshold.json (D3 vs công thức; chia vùng vs phục vụ chung)
python -m experiments.stats                  # Wilcoxon + Holm + kích thước hiệu ứng -> results/stats.md
python -m experiments.pareto                 # đánh đổi chờ – năng lượng -> results/pareto.json
python -m experiments.sensitivity            # độ nhạy ±20 % -> results/sensitivity.json
python export_static.py                      # dist/digital_twin.html (xem không cần máy chủ)
```

## 12 kịch bản

Tải tham chiếu = 180 khách / 5 phút (tải mẫu của tài liệu), nên Off-Peak là dưới 36 khách mỗi 5 phút.
Mức `mid` = số người trong bảng; `low` = 0,6×, `high` = 1,4× (Off-Peak: 0,5× và 1,5×).

| Nhóm | Mã | Kịch bản | Khung giờ | Luồng khách | Số người (mid) |
|---|---|---|---|---|---|
| Up-Peak | U1 | Vào tiết 1 buổi sáng | 6:45–7:00 | 85 % từ G/B1/B2 lên tầng 5–10; 10 % liên tầng; 5 % xuống | 300 / 15 phút |
| Up-Peak | U2 | Vào ca chiều sau giờ nghỉ trưa | 12:40–13:00 | 80 % từ G lên; 12 % liên tầng; 8 % xuống | 240 / 20 phút |
| Up-Peak | U3 | Vào hội thảo tầng 10 | 10 phút trước giờ bắt đầu | 85 % G → 10; 10 % G → tầng khác; 5 % còn lại | 200 / 10 phút |
| Down-Peak | D1 | Tan tiết cuối buổi sáng | 11:25–11:40 | 85 % từ tầng 5–10 về G; 10 % liên tầng; 5 % lên | 300 / 15 phút |
| Down-Peak | D2 | Tan ca cuối ngày | 17:00–17:15 | 82 % về sảnh (1/3 xuống B1/B2); 12 % liên tầng; 6 % lên | 360 / 15 phút |
| Down-Peak | D3 | Hết giờ thi, cả tầng 8 ra cùng lúc | Ngay sau giờ thi | 80 % tầng 8 → G; 15 % tầng 8 → tầng khác; 5 % còn lại | 120 / 5 phút |
| Inter-Floor | I1 | Đổi phòng giữa hai tiết (lớp → lab) | 9:25–9:35 | 45 % liên tầng 5–10; 30 % xuống G; 25 % từ G lên | 150 / 10 phút |
| Inter-Floor | I2 | Thư viện, lab, văn phòng khoa giữa buổi | 8:00–11:00 | 40 % liên tầng (cả tầng 1–4); 30 % lên; 30 % xuống | 100 / 15 phút |
| Inter-Floor | I3 | Ngày hội câu lạc bộ, tầng 5–7 | 8:30–11:00 | 50 % giữa tầng 5, 6, 7; 30 % từ G lên; 20 % xuống | 200 / 30 phút |
| Off-Peak | O1 | Giữa buổi chiều khi đang học | 14:30–15:30 | Lác đác, mọi chiều (≈ 6 % tải tham chiếu) | 30 / 15 phút |
| Off-Peak | O2 | Buổi tối sau 19:00 | 19:00–21:00 | Lớp tối, lao công, bảo vệ (≈ 3 %) | 15 / 15 phút |
| Off-Peak | O3 | Cuối tuần hoặc ngày nghỉ | Cả ngày | Thư viện, người trực (≈ 1,5 %) | 8 / 15 phút |

## Cấu trúc (theo mục 9 tài liệu)

| Tệp | Vai trò |
|---|---|
| `config.py` | Tầng, cabin A1–A6 (tuyến gốc + tầm với giả định), thời gian chạy/cửa/khách, sức chứa, mô hình năng lượng, mô hình sai số cảm biến, tham số P2 |
| `simulation/passenger_generator.py` | Sinh OD theo công thức N = số lớp × sĩ số × tỷ lệ tham gia × tỷ lệ chọn thang; đọc/ghi CSV khảo sát |
| `simulation/scenarios.py` | 12 kịch bản U1–O3 × 3 mức tải; cùng seed → cùng hành khách |
| `simulation/engine.py` | Mô phỏng bước 0,1 s: điều khiển tập hợp, lên/xuống, bypass do đầy, nhật ký cảm biến, khung hình cho web |
| `simulation/sensors.py` | Cảm biến thường: cân tải (ước số người), hồng ngoại ở cửa (đếm lượt), nút gọi tầng – có sai số |
| `simulation/metrics.py` | AWT (tính cả khách chưa đón), P90/P95, ATT, thông lượng, backlog, bypass, số dừng, năng lượng EU, công bằng tầng (Jain), thời gian giải tỏa, RTT |
| `algorithms/baseline.py` | **P0** – tuyến cố định + FCFS, chỉ có cân tải theo kg |
| `algorithms/roles.py` | Vai cabin: Local / Express / Zone / Shuttle / Standby |
| `algorithms/planner.py` | Bộ phân vai: duyệt mọi cách chia nhóm tầng liên tiếp × số cabin, tối thiểu thời gian giải tỏa lớn nhất |
| `algorithms/optimized.py` | **P1** phân vai tĩnh theo thời khóa biểu · **P2** điều phối động (cảm biến + AI + hysteresis) · **P3** thấp điểm/standby · **P4** P2 tiết kiệm năng lượng |
| `ai/` | Đặc trưng cửa sổ trượt từ cảm biến cabin + nút gọi, Random Forest nhận diện chế độ, huấn luyện |
| `optimization/tune_p2.py` | Optuna dò 11 tham số P2, kiểm định trên seed khác |
| `experiments/` | Ma trận thí nghiệm nhiều seed + kiểm định Wilcoxon; nghiên cứu ngưỡng |
| `web_app.py`, `templates/index.html` | Dashboard: mô phỏng song song, trợ lý LLM, kết quả thí nghiệm |
| `realtime/` | Thời gian thực: `bus.py` (MQTT / nội bộ), `building.py` (tòa nhà ảo), `twin.py` (twin), `timetable.py` (thời khóa biểu), `live.py` (phiên trên web) |
| `simulation/whatif.py` | What-if: cabin hỏng, lượng khách khác – so P0 với cấu hình tối ưu |
| `templates/docs.html`, `templates/report.html` | Trang tài liệu (`/docs`), form báo hàng chờ cho điện thoại (`/bao-cao`) |
| `Dockerfile`, `render.yaml` | Đưa lên mạng (Render, Hugging Face Spaces) |
| `llm/assistant.py` | Trợ lý LLM có công cụ chạy mô phỏng (Claude / API tương thích OpenAI / ngoại tuyến) |
| `tests/` | Không bỏ khách, không vượt tải, Express không dừng trung gian, cùng seed, cảm biến, API web |

## Ý tưởng chính

1. **Nút thắt là A1.** Ở P0, tầng 1–4, 6, 7, 9, 10 chỉ có A1 phục vụ, nên A1 chạy gần 100 % thời gian còn A2–A6 rảnh.
2. **Phân vai từng cabin.** Planner chia các tầng có nhu cầu thành nhóm, giao cabin Express/Zone theo
   `W_g = D_g × T_vòng(g) / (n_g × sức chứa)`. A1 luôn Local; ở sảnh G, A1 nhường các tầng đã có cabin chuyên trách,
   nhưng nhận lại mọi hành trình nếu khách chờ quá `fallback_wait`.
3. **Chia vùng hay phục vụ chung.** P2 ước lượng hệ số sử dụng ρ; dưới ngưỡng thì mọi cabin phục vụ chung (pooling),
   trên ngưỡng thì chia Express/Zone. Ngưỡng do Optuna dò.
4. **Cảm biến thường, không camera.** Cân tải ước số người (kg / 65): đủ sức chứa hữu dụng thì không dừng vô ích
   (P0 chỉ báo đầy ở 80 % tải định mức theo kg nên vẫn dừng khi cabin đã chật). Cảm biến hồng ngoại ở cửa đếm lượt
   lên/xuống để ước lượng nhu cầu từng tầng; nút gọi tầng cho biết còn người chờ hay không.
5. **AI** nhận diện chế độ (lên / xuống / hai chiều / liên tầng / thấp điểm) để P2 đổi chiến lược mà không cần người chọn.

## Kết quả

Xem `results/summary.md` (bảng mục 10 đầy đủ), `results/threshold.json`, `results/classifier_report.json`,
`results/optuna_report.json`, và phần "Kết quả thí nghiệm" trên dashboard.

## Thời gian thực (tòa nhà ảo ↔ twin)

Trên dashboard, mục **Tòa nhà ảo và Digital Twin chạy trực tiếp**: chọn giờ bắt đầu (ví dụ 06:40 để xem giờ vào lớp),
tốc độ (1× = giờ thật, 30× để demo), chế độ điều khiển (khép kín / tư vấn) rồi bấm **Bắt đầu**.
- Bên trái: tòa nhà ảo (dữ liệu thật, twin không đọc). Bên phải: những gì twin thấy từ cảm biến.
- Twin nhận diện chế độ bằng AI, gửi lệnh phân vai, dự báo 10 phút tới (giữ nguyên vs theo twin), đo sai số ước lượng hàng chờ.
- **Gây sự cố** một cabin để xem twin tự phân vai lại; **Báo hàng chờ** hoặc mở `/bao-cao` trên điện thoại (dán mã QR ở sảnh).
- Thời khóa biểu: `data/timetable.csv` hoặc Google Sheet công bố dạng CSV (`TIMETABLE_URL` trong `.env`).
- MQTT thật: đặt `MQTT_HOST` (ví dụ `broker.hivemq.com`) và `MQTT_PREFIX=elevtwin/<ma_nhom>` trong `.env`.
  Chạy tòa nhà và twin thành hai tiến trình: `python -m realtime.building --clock 06:40 --speed 30` và `python -m realtime.twin`.

## What-if

Mục **Thử giả định**: chọn kịch bản, cabin hỏng (từ phút nào, bao lâu), hệ số lượng khách → so P0 và P2/P4 khi bình thường
và khi có giả định, kèm đồ thị số người chờ theo thời gian. Trợ lý LLM cũng gọi được công cụ này
("Nếu A1 hỏng 10 phút ở U1 thì sao?").

## Đưa lên mạng

1. Đẩy project lên GitHub (`.gitignore` đã chặn `.env`).
2. **Render**: New → Blueprint → chọn repo (`render.yaml`), nhập `GROQ_API_KEY` trong Environment.
3. **Hugging Face Spaces**: tạo Space loại Docker, đẩy code, thêm `GROQ_API_KEY` trong Settings → Secrets.
Gói miễn phí cho máy chủ ngủ khi không ai truy cập; lần mở đầu mất 30–60 s.

## Trợ lý LLM trên web

**Cách đặt khóa dễ nhất:** đổi tên `.env` thành `.env`, điền `GROQ_API_KEY=gsk_...`, rồi chạy `python web_app.py`. Dòng đầu tiên server in ra LLM đang dùng.

**Trợ lý vận hành (LLM).** Khung *Trợ lý vận hành* trả lời câu hỏi bằng cách **tự gọi công cụ**: chạy mô phỏng,
so sánh P0–P3 trên cùng hành khách, đọc kết quả nhiều seed (`llm/assistant.py`).

```bash
# Claude
export ANTHROPIC_API_KEY=...            # tuỳ chọn: LLM_MODEL=claude-sonnet-5-5 (mặc định claude-haiku-5-5)
# hoặc Groq (miễn phí có giới hạn) – tuỳ chọn GROQ_MODEL=openai/gpt-oss-120b
export GROQ_API_KEY=gsk_...
# hoặc bất kỳ API tương thích OpenAI (OpenAI, Gemini, Groq, Ollama…)
export OPENAI_API_KEY=...  OPENAI_BASE_URL=https://api.openai.com/v1  OPENAI_MODEL=gpt-4o-mini
python web_app.py
```

Không đặt khóa nào thì trợ lý chạy **ngoại tuyến**: hiểu câu hỏi bằng luật đơn giản (kịch bản, số người, "nếu…",
"vì sao…") nhưng số liệu vẫn lấy từ mô phỏng thật. Trên Windows dùng `set` thay cho `export`.

## Giới hạn

- Tham số thời gian (2,2 s/tầng, 4 s cửa, 0,6 s/khách) là giả định; cần thay bằng số đo khảo sát (`data/calibration.json`).
- Bỏ qua gia tốc/giảm tốc; năng lượng là đơn vị tương đối EU, không phải kWh.
- Mô hình sai số cảm biến (cân tải, hồng ngoại) là giả định, chưa đối chiếu với thiết bị thật.
- Chế độ thời gian thực được kiểm chứng với **tòa nhà ảo** (software-in-the-loop), chưa nối thang thật.
  Tòa nhà ảo dùng chung bộ mô phỏng với twin nên sai số ước lượng đo được lạc quan hơn thực tế.
- Số khách trong 12 kịch bản là giá trị thiết kế thí nghiệm; thay bằng thời khóa biểu thật qua CSV (`load_csv`) khi có.
