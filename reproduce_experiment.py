import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. Hàm Tiền xử lý dữ liệu (Preprocessing)
# ---------------------------------------------------------
def preprocess_data(df, scenario):
    df = df.drop_duplicates()
    
    labels = []
    for index, row in df.iterrows():
        if scenario == 1:
            labels.append(row['Class'])
        elif scenario == 2:
            category = row['Category']
            if 'Benign' in category:
                labels.append('Benign')
            elif 'Ransomware' in category:
                labels.append('Ransomware')
            elif 'Spyware' in category:
                labels.append('Spyware')
            elif 'Trojan' in category:
                labels.append('Trojan')
            else:
                labels.append('Unknown')
        elif scenario == 3:
            category = row['Category']
            if category == 'Benign':
                labels.append('Benign')
            else:
                parts = str(category).split('-')
                if len(parts) > 1:
                    labels.append(parts[1])
                else:
                    labels.append(category)
    
    df['Label'] = labels
    
    cols_to_drop = ['Label']
    for col in ['Filename', 'Class', 'Category']:
        if col in df.columns:
            cols_to_drop.append(col)
            
    X = df.drop(columns=cols_to_drop)
    y = df['Label']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Tạo dataset đã tiền xử lý để xuất ra file CSV
    df_preprocessed = X_scaled.copy()
    df_preprocessed['Label'] = y.values
    
    return X_scaled, y, df_preprocessed

# ---------------------------------------------------------
# 2. Hàm Đánh giá mô hình (Evaluation Metrics)
# ---------------------------------------------------------
def calculate_metrics(y_true, y_pred, labels):
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    
    FPR_array = np.divide(FP, (FP + TN), out=np.zeros_like(FP, dtype=float), where=(FP+TN)!=0)
    FPR = np.mean(FPR_array)
    
    return acc, recall, precision, f1, FPR

# ---------------------------------------------------------
# 3. Hàm Huấn luyện và Kiểm thử (Training & Testing)
# ---------------------------------------------------------
def run_experiment(file_path, scenario):
    print(f"\n========================================================")
    print(f"ĐANG CHẠY KỊCH BẢN {scenario} VỚI FILE: {file_path}")
    print(f"========================================================")
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {file_path}")
        return
        
    X, y, df_preprocessed = preprocess_data(df, scenario)
    
    # Lưu CSV kịch bản mới
    csv_filename = f"Dataset_Scenario_{scenario}.csv"
    df_preprocessed.to_csv(csv_filename, index=False)
    print(f"✅ Đã lưu tập dữ liệu tiền xử lý (Kịch bản {scenario}) vào: {csv_filename}")
    
    unique_classes = y.unique()
    print(f"Số lượng mẫu sau khi xóa trùng lặp: {len(X)}")
    print(f"Các lớp được phát hiện ({len(unique_classes)}): {list(unique_classes)}")
    
    if len(unique_classes) < 2:
        print(f"⚠️ CẢNH BÁO: Dữ liệu chỉ có {len(unique_classes)} lớp. Không thể huấn luyện mô hình phân loại!")
        return

    label_mapping = {label: idx for idx, label in enumerate(unique_classes)}
    y_mapped = y.map(label_mapping)

    models = {
        'Random Tree (RT)': DecisionTreeClassifier(random_state=42),
        'Random Forest (RF)': RandomForestClassifier(random_state=42, n_jobs=-1),
        'J-48 (Decision Tree)': DecisionTreeClassifier(criterion='entropy', random_state=42),
        'Naive Bayes (NB)': GaussianNB(),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1)
    }

    # Lưu thông số (hyperparameters) của các mô hình
    model_params = {name: model.get_params() for name, model in models.items()}
    os.makedirs('Models', exist_ok=True)
    with open(f'Models/Hyperparameters_Scenario_{scenario}.json', 'w') as f:
        json.dump(model_params, f, indent=4)

    results = []
    
    # Tạo thư mục lưu mô hình cho kịch bản
    model_dir = f'Models/Scenario_{scenario}'
    os.makedirs(model_dir, exist_ok=True)
    
    # Lưu mapping của nhãn
    with open(f'{model_dir}/label_mapping.json', 'w') as f:
        json.dump(label_mapping, f, indent=4)

    for name, model in models.items():
        print(f"---> Đang huấn luyện: {name}")
        
        # 3.1. 80% Split
        X_train, X_test, y_train, y_test = train_test_split(X, y_mapped, test_size=0.2, random_state=42, stratify=y_mapped)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Lưu mô hình đã train (từ 80% split) ra file
        safe_model_name = name.split(' (')[0].replace(' ', '_').replace('-', '_')
        joblib.dump(model, f'{model_dir}/{safe_model_name}.pkl')
        
        labels_mapped_list = list(label_mapping.values())
        acc_split, rec_split, prec_split, f1_split, fpr_split = calculate_metrics(y_test, y_pred, labels_mapped_list)
        
        # 3.2. 10-Fold CV
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        cv_scores = cross_validate(model, X, y_mapped, cv=cv, scoring=['accuracy', 'recall_weighted', 'precision_weighted', 'f1_weighted'], n_jobs=-1)
        
        acc_cv = cv_scores['test_accuracy'].mean()
        rec_cv = cv_scores['test_recall_weighted'].mean()
        prec_cv = cv_scores['test_precision_weighted'].mean()
        f1_cv = cv_scores['test_f1_weighted'].mean()
        
        results.append({
            'Model': name, 'Test_Method': '80% Split',
            'Accuracy': acc_split, 'Recall': rec_split, 'Precision': prec_split, 'F1-Score': f1_split, 'FPR': fpr_split
        })
        results.append({
            'Model': name, 'Test_Method': '10-Fold CV',
            'Accuracy': acc_cv, 'Recall': rec_cv, 'Precision': prec_cv, 'F1-Score': f1_cv, 'FPR': '-'
        })

    # Lưu kết quả
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'Metrics_Scenario_{scenario}.csv', index=False)
    print(f"✅ Đã lưu kết quả độ chính xác vào: Metrics_Scenario_{scenario}.csv")
    print(f"✅ Đã lưu các thông số và trọng số mô hình vào thư mục: {model_dir}/")

# ---------------------------------------------------------
# Chạy thực nghiệm
# ---------------------------------------------------------
if __name__ == "__main__":
    dataset_file = 'MalMem2022.csv'
    run_experiment(dataset_file, scenario=1)
    run_experiment(dataset_file, scenario=2)
    run_experiment(dataset_file, scenario=3)
