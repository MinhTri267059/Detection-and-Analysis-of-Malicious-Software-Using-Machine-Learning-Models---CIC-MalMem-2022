# Detection and Analysis of Malicious Software Using Machine Learning Models (CIC-MalMem-2022)

Repository này chứa mã nguồn Python nhằm tái hiện lại các kịch bản thực nghiệm phân loại mã độc dựa trên bộ dữ liệu **CIC-MalMem-2022**. Mã nguồn thực hiện quá trình tiền xử lý, huấn luyện và đánh giá 5 thuật toán Học máy (Random Tree, Random Forest, J-48, Naive Bayes, XGBoost) qua 3 kịch bản phân loại với 2 phương pháp kiểm thử (Percentage Split 80% và 10-Fold Cross-Validation).

## 📥 Tải xuống Bộ dữ liệu (Dataset)

Để chạy mã nguồn, bạn cần tải xuống tập dữ liệu gốc `MalMem2022.csv`.
🔗 **[Tải xuống Dataset CIC-MalMem-2022 (Google Drive)](https://drive.google.com/drive/folders/1TooIQ5HmU1n5X34zYpIgmtTW49X9_QMY?usp=sharing)**

Sau khi tải xong, hãy đặt file `MalMem2022.csv` vào thư mục gốc của repository này.

## 🚀 Hướng dẫn Sử dụng

1. **Cài đặt thư viện yêu cầu:**
   Bạn cần cài đặt các thư viện Python như `pandas`, `scikit-learn`, và `xgboost`.
   ```bash
   pip install pandas numpy scikit-learn xgboost joblib
   ```

2. **Chạy kịch bản thực nghiệm:**
   Đảm bảo file `MalMem2022.csv` đã nằm trong thư mục gốc. Chạy lệnh sau:
   ```bash
   python3 reproduce_experiment.py
   ```

   **Mã nguồn sẽ tự động thực hiện:**
   - Xóa bỏ dữ liệu trùng lặp (giữ lại 58.062 mẫu hợp lệ).
   - Xuất dữ liệu đã chuẩn hóa của 3 kịch bản ra các file CSV (`Dataset_Scenario_1.csv`, `Dataset_Scenario_2.csv`, v.v.).
   - Lưu trữ trọng số mô hình đã huấn luyện (file `.pkl`) và tham số (file `.json`) vào thư mục `Models/`.
   - Xuất bảng chỉ số đánh giá ra các file `Metrics_Scenario_*.csv`.

*(Ghi chú: Các file Dataset dung lượng lớn và file model nặng đã được `.gitignore` để tránh phình to kích thước repository).*

---

## 📊 Kết Quả Thực Nghiệm (Tái hiện)

Dưới đây là tóm tắt độ chính xác (Accuracy) đạt được sau quá trình chạy mô phỏng bằng Python. Nhìn chung, kết quả bám rất sát và phản ánh đúng xu hướng của bài báo gốc.

### 1. Kịch bản Phân loại Nhị phân (Binary - 2 lớp: Benign và Malware)
Sự phân tách giữa lành tính và mã độc rất rõ ràng. Các thuật toán Tree-based và Ensemble đều đạt hiệu suất gần như tuyệt đối (100%).

| Mô hình | Phương pháp | Accuracy (Tái hiện) |
| :--- | :--- | :--- |
| **XGBoost** | 80% Split | **100.0%** |
| **Random Forest (RF)** | 80% Split | **100.0%** |
| **J-48 (Decision Tree)**| 80% Split | 99.98% |
| **Naive Bayes (NB)** | 10-Fold CV | 99.25% |

### 2. Kịch bản Đa lớp 4 nhóm (Multi-4: Benign, Ransomware, Spyware, Trojan)
Bài toán trở nên phức tạp hơn, XGBoost và Random Forest là 2 mô hình dẫn đầu, trong khi Naive Bayes giảm sút mạnh.

| Mô hình | Phương pháp | Accuracy (Tái hiện) |
| :--- | :--- | :--- |
| **XGBoost** | 10-Fold CV | **87.71%** |
| **Random Forest (RF)** | 10-Fold CV | **87.78%** |
| **J-48 (Decision Tree)**| 10-Fold CV | 85.09% |
| **Naive Bayes (NB)** | 10-Fold CV | 68.70% |

### 3. Kịch bản Đa lớp 16 nhóm (Multi-16: 1 Benign + 15 họ mã độc)
Trong kịch bản khó nhất (phân biệt chi tiết 15 dòng mã độc), XGBoost thể hiện rõ ưu thế tuyệt đối so với các thuật toán khác.

| Mô hình | Phương pháp | Accuracy (Tái hiện) |
| :--- | :--- | :--- |
| **XGBoost** | 10-Fold CV | **76.60%** |
| **Random Forest (RF)** | 10-Fold CV | 75.68% |
| **J-48 (Decision Tree)**| 10-Fold CV | 73.65% |
| **Naive Bayes (NB)** | 10-Fold CV | 56.83% |
