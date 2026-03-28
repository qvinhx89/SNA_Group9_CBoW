# Team Execution Playbook - SNA Twitch Influencer Project

## 1. Muc tieu cua tai lieu

Tai lieu nay la huong dan thao tac chi tiet cho nhom 3 nguoi, tu setup moi truong den chia nhanh song song.
Muc tieu la:

- Moi thanh vien biet ro minh can lam gi.
- Tat ca dau viec co thu tu phu thuoc ro rang.
- Co tieu chi hoan thanh (Definition of Done) cho tung buoc.
- Giam rui ro lam trung, lam thieu, hoac sai scope.

---

## 2. Co cau nhom va vai tro

- Vai tro 1: Pham Quoc Vinh
- Vai tro 2: Tran Hung Vi
- Vai tro 3: Tran Quoc Hai

Nguyen tac:

- Lead chot scope, merge cuoi, va giai quyet xung dot ky thuat.
- Moi thanh vien van phai code, chay artifact, va review cheo.
- Moi thay doi tham so bat buoc ghi vao experiment registry.

---

## 3. Tong quan luong cong viec

Thu tu tong the:

1. Setup moi truong va du lieu (ca 3 nguoi cung lam).
2. Khoa scope ky thuat (ca 3 nguoi xac nhan, Lead chot).
3. Chay tuyen nen stage 0 -> 3 (tuan tu, khong tach nhanh).
4. Chia nhanh song song (3 nhanh cho 3 nguoi).
5. Tich hop ket qua, viet bao cao, review cheo.
6. Chay reproducibility check va chot nop.

---

## 4. Giai doan A - Setup moi truong (bat buoc cho ca 3 nguoi)

### A1. Dung dung thu muc du an

Thu muc lam viec:

- sna_twitch_influencer_project

Luu y:

- Khong dung nham virtual environment cua thu muc khac.
- Nen tao venv rieng trong chinh du an nay.

### A2. Tao va kich hoat virtual environment

PowerShell:

    cd C:\Users\ASUS\Documents\UIT\Nam3\Semester2\Social\sna_twitch_influencer_project
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

### A3. Cai dependencies

    python -m pip install --upgrade pip
    pip install -r requirements.txt

### A4. Smoke test import

    python -c "import networkx, pandas, sklearn, yaml; print('OK')"

### A5. Kiem tra du lieu dau vao

- Kiem tra ton tai data/raw va cac file nguon can thiet.
- Neu thieu file, dung pipeline va thong bao ngay cho Lead.

### A6. Definition of Done cho Giai doan A

- Moi thanh vien tu chay duoc lenh import test va ra OK.
- Moi thanh vien xac nhan da o dung venv cua project.
- Du lieu raw du dieu kien de chay stage 0.

---

## 5. Giai doan B - Khoa scope (bat buoc truoc khi chay lon)

### B1. Doi soat file scope

Can doi soat cac file:

- docs/implementation_notes.md
- src/config/base.yaml
- run_all.py
- reports/final_report.md

### B2. Chot mot scope duy nhat

- Thong nhat chinh xac cac mo hinh, metric, va cac phan ngoai scope.
- Loai bo noi dung khong con phu hop khoi runner va report template.

### B3. Ghi nhat ky thay doi

- Them entry vao docs/experiment_registry.md voi timestamp, ly do, tac dong.

### B4. Definition of Done cho Giai doan B

- Khong con mau thuan scope giua config, code runner, va report.
- Registry co entry moi cho quyet dinh scope.

---

## 6. Giai doan C - Chay tuyen nen (stage 0 -> 3, khong song song)

### C1. Stage 0: Data audit + preprocess

Chay:

    python run_all.py --stage 0

Can co:

- Artifact data quality
- Du lieu processed de lam dau vao cho stage sau

### C2. Stage 1: Centrality

Chay:

    python run_all.py --stage 1

Can co:

- Bang centrality co day du cot can thiet

### C3. Stage 2: Community + k-shell

Chay:

    python run_all.py --stage 2

Can co:

- Nhan community
- Ket qua k-shell

### C4. Stage 3: SIS + typology + robustness

Chay:

    python run_all.py --stage 3

Can co:

- sis_table
- typology labels
- robustness summary

### C5. Quality gate cua Giai doan C

Chi duoc tach nhanh song song neu:

- Stage 0 -> 3 chay xong khong loi nghiem trong.
- Cac artifact cot loi ton tai va doc duoc.
- Lead xac nhan dau vao cho stage 4, 5, 6 da san sang.

---

## 7. Giai doan D - Chia nhanh song song cho 3 nguoi

Sau quality gate cua stage 3, chia 3 nhanh nhu sau.

### Nhanh D1 - Lead (RQ2 + RQ3)

Phan viec:

- Chay IC calibration
- Chay single-seed validation
- Chay multi-seed benchmark
- Tong hop ket qua RQ2, RQ3

Lenh goi y:

    python run_all.py --stage 4
    python run_all.py --stage 5

Dau ra bat buoc:

- Bang ket qua stage4_single_seed
- Bang ket qua stage5_multi_seed
- Figure so sanh strategy

### Nhanh D2 - Thanh vien A (RQ1 + quality structure)

Phan viec:

- Kiem tra tinh nhat quan output stage 1 -> 3
- Tao bang/figure cho RQ1
- Ho tro doi soat centrality va typology

Dau ra bat buoc:

- Table RQ1
- Figure divergence views vs structural metrics
- Ghi chu quality check

### Nhanh D3 - Thanh vien B (RQ4 ML)

Phan viec:

- Chuan bi feature surface cho ML
- Chay stage ML theo scope da khoa
- Xuat metric va confusion matrix

Lenh goi y:

    python run_all.py --stage 6

Dau ra bat buoc:

- Bang ket qua RQ4
- Figure confusion matrix
- Ghi chu ket luan detectability

### Rule phoi hop trong giai doan song song

- Moi nhanh lam tren branch rieng.
- Khong sua file owner cua nhanh khac neu chua thong bao.
- Moi thay doi tham so phai ghi ngay vao experiment registry.
- Cuoi ngay, moi nguoi cap nhat trang thai: Done, Doing, Blocked.

---

## 8. Giai doan E - Tich hop va viet bao cao

### E1. Thu tu merge

1. Merge nhanh D2 (RQ1 va structure checks)
2. Merge nhanh D1 (simulation RQ2, RQ3)
3. Merge nhanh D3 (ML RQ4)
4. Lead chay lai tong hop sau merge

### E2. Hoan thien bao cao

- Dien ket qua that vao reports/final_report.md
- Dong bo figure vao reports/figures
- Dong bo table vao reports/tables
- Moi claim phai co evidence tu artifact

### E3. Review cheo

- Vong 1: review ky thuat (logic, metric, reproducibility)
- Vong 2: review trinh bay (ro rang, nhat quan RQ1 -> RQ4)

### E4. Definition of Done cho Giai doan E

- Bao cao khong con placeholder quan trong.
- Figure/table khop 100% voi noi dung ket luan.
- Tat ca phan thay doi da duoc review it nhat 1 lan.

---

## 9. Giai doan F - Kiem tra truoc khi nop

### F1. Reproducibility run

- Chay lai pipeline tu dau tren mot may sach hoac venv sach.
- Xac nhan artifact sinh ra day du.

### F2. Checklist nop bai

- Outputs stage can thiet co du
- reports/figures co du hinh can nop
- reports/tables co du bang can nop
- docs/experiment_registry.md da cap nhat day du
- Bao cao final da chot ngay thang va thanh vien

### F3. Definition of Done cho toan du an

- Du 4 RQ voi bang va hinh minh chung.
- Chay lai duoc theo quy trinh da ghi.
- Nhom thong nhat ban nop cuoi.

---

## 10. Bang phan cong nhanh (co the copy vao task board)

- Lead:
  - Scope lock va merge cuoi
  - Stage 4 + Stage 5
  - Tong hop RQ2, RQ3
- Thanh vien A:
  - Stage 1 -> 3 quality check
  - RQ1 tables + figures
  - Review nhanh ML
- Thanh vien B:
  - Stage 6 ML
  - RQ4 tables + figures
  - Review nhanh simulation

---

## 11. Moc tien do tham chieu

- Xong Giai doan A + B + C: 40%
- Xong cac nhanh song song D1 + D2 + D3: 75%
- Xong tich hop va review (E): 90%
- Xong reproducibility check va chot nop (F): 100%

