# Team Execution Playbook - SNA Twitch Influencer Project

## 1. Mục tiêu của tài liệu

Tài liệu này là hướng dẫn thao tác chi tiết cho nhóm 3 người, từ setup môi trường đến chia nhánh song song.
Mục tiêu là:

- Mỗi thành viên biết rõ mình cần làm gì.
- Tất cả đầu việc có thứ tự phụ thuộc rõ ràng.
- Có tiêu chí hoàn thành (Definition of Done) cho từng bước.
- Giảm rủi ro làm trùng, làm thiếu, hoặc sai scope.

---

## 2. Cơ cấu nhóm và vai trò

- Vai trò 1: Phạm Quốc Vĩnh
- Vai trò 2: Trần Hùng Vĩ
- Vai trò 3: Trần Quốc Hải

Nguyên tắc:

- Lead chốt scope, merge cuối, và giải quyết xung đột kỹ thuật.
- Mỗi thành viên vẫn phải code, chạy artifact, và review chéo.
- Mọi thay đổi tham số bắt buộc ghi vào experiment registry.

### Bức tranh toàn cảnh nhiệm vụ (Big Picture)
Để dễ hình dung trước khi bắt tay làm, khối lượng công việc được chia tổng quan như sau:
- **Lead (Vĩnh)** - *Người ghép nối & Mô phỏng:* Lo setup luồng chung, kiểm soát quy trình. Đảm nhiệm phần khắt khe nhất là chạy mô phỏng mô hình lan truyền (IC Model) để tìm ra chiến lược gieo mầm (seed) tối ưu (giải quyết **RQ2, RQ3**).
- **Thành viên A (Vĩ)** - *Chuyên gia Cấu trúc mạng:* Tập trung phân tích nền tảng, tính toán các độ đo trung tâm (Centrality), phân tích cộng đồng (Community) và K-shell để rút ra phân loại (typology) của các influencer (giải quyết **RQ1**).
- **Thành viên B (Hải)** - *Kỹ sư Machine Learning:* Dùng các đặc trưng rút trích từ dữ liệu mạng lưới để huấn luyện mô hình phân loại (Machine Learning), nhằm dự đoán độ chính xác của việc tìm ra influencer (giải quyết **RQ4**).
*(**Lưu ý:** Đoạn đầu setup môi trường làm sạch dữ liệu và đoạn cuối ráp báo cáo thì cả 3 sẽ cùng phối hợp làm tuần tự).*

---

## 3. Tổng quan luồng công việc

Thứ tự tổng thể:

1. Setup môi trường và dữ liệu (cả 3 người cùng làm).
2. Khóa scope kỹ thuật (cả 3 người xác nhận, Lead chốt).
3. Chạy tuyến nền stage 0 -> 3 (tuần tự, không tách nhánh).
4. Chia nhánh song song (3 nhánh cho 3 người).
5. Tích hợp kết quả, viết báo cáo, review chéo.
6. Chạy reproducibility check và chốt nộp.

---

## 4. Giai đoạn A - Setup môi trường (bắt buộc cho cả 3 người)

### A1. Dựng đúng thư mục dự án

Thư mục làm việc:

- sna_twitch_influencer_project

Lưu ý:

- Không dùng nhầm virtual environment của thư mục khác.
- Nên tạo venv riêng trong chính dự án này.

### A2. Tạo và kích hoạt virtual environment

PowerShell:

    cd C:\Users\ASUS\Documents\UIT\Nam3\Semester2\Social\sna_twitch_influencer_project
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

### A3. Cài dependencies

    python -m pip install --upgrade pip
    pip install -r requirements.txt

### A4. Smoke test import

    python -c "import networkx, pandas, sklearn, yaml; print('OK')"

### A5. Kiểm tra dữ liệu đầu vào

- Kiểm tra tồn tại data/raw và các file nguồn cần thiết.
- Nếu thiếu file, dừng pipeline và thông báo ngay cho Lead.

### A6. Definition of Done cho Giai đoạn A

- Mỗi thành viên tự chạy được lệnh import test và ra OK.
- Mỗi thành viên xác nhận đã ở đúng venv của project.
- Dữ liệu raw đủ điều kiện để chạy stage 0.

---

## 5. Giai đoạn B - Khóa scope (bắt buộc trước khi chạy lớn)

### B1. Đối soát file scope

Cần đối soát các file:

- docs/implementation_notes.md
- src/config/base.yaml
- run_all.py
- reports/final_report.md

### B2. Chốt một scope duy nhất

- Thống nhất chính xác các mô hình, metric, và các phần ngoài scope.
- Loại bỏ nội dung không còn phù hợp khỏi runner và report template.

### B3. Ghi nhật ký thay đổi

- Thêm entry vào docs/experiment_registry.md với timestamp, lý do, tác động.

### B4. Definition of Done cho Giai đoạn B

- Không còn mâu thuẫn scope giữa config, code runner, và report.
- Registry có entry mới cho quyết định scope.

---

## 6. Giai đoạn C - Chạy tuyến nền (stage 0 -> 3, không song song)

### C1. Stage 0: Data audit + preprocess

Chạy:

    python run_all.py --stage 0

Cần có:

- Artifact data quality
- Dữ liệu processed để làm đầu vào cho stage sau

### C2. Stage 1: Centrality

Chạy:

    python run_all.py --stage 1

Cần có:

- Bảng centrality có đầy đủ cột cần thiết

### C3. Stage 2: Community + k-shell

Chạy:

    python run_all.py --stage 2

Cần có:

- Nhãn community
- Kết quả k-shell

### C4. Stage 3: SIS + typology + robustness

Chạy:

    python run_all.py --stage 3

Cần có:

- sis_table
- typology labels
- robustness summary

### C5. Quality gate của Giai đoạn C

Chỉ được tách nhánh song song nếu:

- Stage 0 -> 3 chạy xong không lỗi nghiêm trọng.
- Các artifact cốt lõi tồn tại và đọc được.
- Lead xác nhận đầu vào cho stage 4, 5, 6 đã sẵn sàng.

---

## 7. Giai đoạn D - Chia nhánh song song cho 3 người

Sau quality gate của stage 3, chia 3 nhánh như sau.

### Nhánh D1 - Lead (RQ2 + RQ3)

Phần việc:

- Chạy IC calibration
- Chạy single-seed validation
- Chạy multi-seed benchmark
- Tổng hợp kết quả RQ2, RQ3

Lệnh gợi ý:

    python run_all.py --stage 4
    python run_all.py --stage 5

Đầu ra bắt buộc:

- Bảng kết quả stage4_single_seed
- Bảng kết quả stage5_multi_seed
- Figure so sánh strategy

### Nhánh D2 - Thành viên A (RQ1 + quality structure)

Phần việc:

- Kiểm tra tính nhất quán output stage 1 -> 3
- Tạo bảng/figure cho RQ1
- Hỗ trợ đối soát centrality và typology

Đầu ra bắt buộc:

- Table RQ1
- Figure divergence views vs structural metrics
- Ghi chú quality check

### Nhánh D3 - Thành viên B (RQ4 ML)

Phần việc:

- Chuẩn bị feature surface cho ML
- Chạy stage ML theo scope đã khóa
- Xuất metric và confusion matrix

Lệnh gợi ý:

    python run_all.py --stage 6

Đầu ra bắt buộc:

- Bảng kết quả RQ4
- Figure confusion matrix
- Ghi chú kết luận detectability

### Rule phối hợp trong giai đoạn song song

- Mỗi nhánh làm trên branch riêng.
- Không sửa file owner của nhánh khác nếu chưa thông báo.
- Mọi thay đổi tham số phải ghi ngay vào experiment registry.
- Cuối ngày, mọi người cập nhật trạng thái: Done, Doing, Blocked.

---

## 8. Giai đoạn E - Tích hợp và viết báo cáo

### E1. Thứ tự merge

1. Merge nhánh D2 (RQ1 và structure checks)
2. Merge nhánh D1 (simulation RQ2, RQ3)
3. Merge nhánh D3 (ML RQ4)
4. Lead chạy lại tổng hợp sau merge

### E2. Hoàn thiện báo cáo

- Điền kết quả thật vào reports/final_report.md
- Đồng bộ figure vào reports/figures
- Đồng bộ table vào reports/tables
- Mọi claim phải có evidence từ artifact

### E3. Review chéo

- Vòng 1: review kỹ thuật (logic, metric, reproducibility)
- Vòng 2: review trình bày (rõ ràng, nhất quán RQ1 -> RQ4)

### E4. Definition of Done cho Giai đoạn E

- Báo cáo không còn placeholder quan trọng.
- Figure/table khớp 100% với nội dung kết luận.
- Tất cả phần thay đổi đã được review ít nhất 1 lần.

---

## 9. Giai đoạn F - Kiểm tra trước khi nộp

### F1. Reproducibility run

- Chạy lại pipeline từ đầu trên một máy sạch hoặc venv sạch.
- Xác nhận artifact sinh ra đầy đủ.

### F2. Checklist nộp bài

- Outputs stage cần thiết có đủ
- reports/figures có đủ hình cần nộp
- reports/tables có đủ bảng cần nộp
- docs/experiment_registry.md đã cập nhật đầy đủ
- Báo cáo final đã chốt ngày tháng và thành viên

### F3. Definition of Done cho toàn dự án

- Đủ 4 RQ với bảng và hình minh chứng.
- Chạy lại được theo quy trình đã ghi.
- Nhóm thống nhất bản nộp cuối.

---

## 10. Bảng phân công nhánh (có thể copy vào task board)

- Lead:
  - Scope lock và merge cuối
  - Stage 4 + Stage 5
  - Tổng hợp RQ2, RQ3
- Thành viên A:
  - Stage 1 -> 3 quality check
  - RQ1 tables + figures
  - Review nhánh ML
- Thành viên B:
  - Stage 6 ML
  - RQ4 tables + figures
  - Review nhánh simulation

---

## 11. Mốc tiến độ tham chiếu

- Xong Giai đoạn A + B + C: 40%
- Xong các nhánh song song D1 + D2 + D3: 75%
- Xong tích hợp và review (E): 90%
- Xong reproducibility check và chốt nộp (F): 100%

