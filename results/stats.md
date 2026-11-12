# Kiểm định thống kê (t-test cặp đôi trên log AWT + hiệu chỉnh Holm)

288 phép so sánh với P0 (cùng seed = cùng hành khách). Hiệu ứng = rank-biserial (1 = tốt hơn P0 ở mọi seed).

| Cấu hình | Số phép so sánh | Tốt hơn có ý nghĩa (Holm) | Tệ hơn có ý nghĩa | Tốt hơn ở mọi seed | Giảm AWT trung vị |
|---|---|---|---|---|---|
| P1 | 72 | 53 | 0 | 50 | 80 % |
| P2 | 72 | 68 | 0 | 68 | 85 % |
| P3 | 72 | 62 | 0 | 62 | 73 % |
| P4 | 72 | 62 | 1 | 62 | 63 % |

Wilcoxon (không tham số) được ghi kèm trong stats.json; với 10 seed p nhỏ nhất của nó ≈ 0,002 nên
không đủ để qua hiệu chỉnh Holm cho 288 phép so sánh – vì vậy dùng t-test trên log AWT làm kiểm định chính.
