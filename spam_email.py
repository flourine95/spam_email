import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

# Load dataset Spambase
data = pd.read_csv("spambase/spambase.data", header=None)
# Định nghĩa tên cột cho các thuộc tính
column_names = [
    'word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 'word_freq_our',
    'word_freq_over', 'word_freq_remove', 'word_freq_internet', 'word_freq_order', 'word_freq_mail',
    'word_freq_receive', 'word_freq_will', 'word_freq_people', 'word_freq_report', 'word_freq_addresses',
    'word_freq_free', 'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit',
    'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money', 'word_freq_hp',
    'word_freq_hpl', 'word_freq_george', 'word_freq_650', 'word_freq_lab', 'word_freq_labs',
    'word_freq_telnet', 'word_freq_857', 'word_freq_data', 'word_freq_415', 'word_freq_85',
    'word_freq_technology', 'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct',
    'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project', 'word_freq_re',
    'word_freq_edu', 'word_freq_table', 'word_freq_conference',
    'char_freq_;', 'char_freq_(', 'char_freq_[', 'char_freq_!', 'char_freq_$', 'char_freq_#',
    'capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total', 'spam'
]

data.columns = column_names
print(data.info())  # Loại dữ liệu, giá trị null
print(data.describe())  # Thống kê cơ bản
print(data.head())  # Xem trước vài hàng
print(data['spam'].value_counts(normalize=True))
plt.figure(figsize=(6, 3))

plt.hist(data['word_freq_free'], bins=30, color='blue', alpha=0.7)
plt.title('Distribution of word_freq_free')
plt.xlabel('Frequency of "word_freq_free"')
plt.ylabel('Count')

plt.tight_layout()
plt.show()

# Chia thành feature và label
X = data.iloc[:, :-1]  # Các cột đặc trưng (từ cột 0 đến cột cuối trừ 1)
y = data.iloc[:, -1]  # Nhãn (cột cuối cùng)

# Chia dữ liệu thành tập train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Naive Bayes ---
print("=== Naive Bayes ===")
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

# Tính toán các chỉ số
accuracy_nb = accuracy_score(y_test, y_pred_nb)
precision_nb = precision_score(y_test, y_pred_nb)
recall_nb = recall_score(y_test, y_pred_nb)
f1_nb = f1_score(y_test, y_pred_nb)

print(f"Accuracy: {accuracy_nb:.2f}")
print(f"Precision: {precision_nb:.2f}")
print(f"Recall: {recall_nb:.2f}")
print(f"F1 Score: {f1_nb:.2f}")
print(classification_report(y_test, y_pred_nb))

# --- Decision Tree ---
print("\n=== Decision Tree ===")
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# Tính toán các chỉ số
accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)

print(f"Accuracy: {accuracy_dt:.2f}")
print(f"Precision: {precision_dt:.2f}")
print(f"Recall: {recall_dt:.2f}")
print(f"F1 Score: {f1_dt:.2f}")
print(classification_report(y_test, y_pred_dt))

# --- Random Forest ---
print("\n=== Random Forest ===")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Tính toán các chỉ số
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

print(f"Accuracy: {accuracy_rf:.2f}")
print(f"Precision: {precision_rf:.2f}")
print(f"Recall: {recall_rf:.2f}")
print(f"F1 Score: {f1_rf:.2f}")
print(classification_report(y_test, y_pred_rf))

# --- Multi-Layer Perceptron (MLP) ---
print("\n=== Multi-Layer Perceptron ===")
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)

# Tính toán các chỉ số
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
precision_mlp = precision_score(y_test, y_pred_mlp)
recall_mlp = recall_score(y_test, y_pred_mlp)
f1_mlp = f1_score(y_test, y_pred_mlp)

print(f"Accuracy: {accuracy_mlp:.2f}")
print(f"Precision: {precision_mlp:.2f}")
print(f"Recall: {recall_mlp:.2f}")
print(f"F1 Score: {f1_mlp:.2f}")
print(classification_report(y_test, y_pred_mlp))

# Trực quan hóa độ chính xác của các mô hình
models = ['Naive Bayes', 'Decision Tree', 'Random Forest', 'MLP']
accuracies = [accuracy_nb, accuracy_dt, accuracy_rf, accuracy_mlp]
precisions = [precision_nb, precision_dt, precision_rf, precision_mlp]
recalls = [recall_nb, recall_dt, recall_rf, recall_mlp]
f1_scores = [f1_nb, f1_dt, f1_rf, f1_mlp]

# Tạo biểu đồ so sánh
x = range(len(models))
plt.figure(figsize=(24, 12))

plt.subplot(1, 4, 1)
plt.bar(x, accuracies, color='blue')
plt.xticks(x, models)
plt.ylabel('Accuracy')
plt.title('Model Accuracy')

plt.subplot(1, 4, 2)
plt.bar(x, precisions, color='orange')
plt.xticks(x, models)
plt.ylabel('Precision')
plt.title('Model Precision')

plt.subplot(1, 4, 3)
plt.bar(x, recalls, color='green')
plt.xticks(x, models)
plt.ylabel('Recall')
plt.title('Model Recall')

plt.subplot(1, 4, 4)
plt.bar(x, f1_scores, color='red')
plt.xticks(x, models)
plt.ylabel('F1 Score')
plt.title('Model F1 Score')

plt.tight_layout()
plt.show()


# Tạo biểu đồ radar
def create_radar_chart(models, metrics, title):
    num_vars = len(metrics)

    # Tạo một mảng cho các góc
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Đóng vòng cho biểu đồ
    metrics += metrics[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for i, model in enumerate(models):
        values = [accuracies[i], precisions[i], recalls[i], f1_scores[i]]
        values += values[:1]  # Đóng vòng
        ax.fill(angles, values, alpha=0.25, label=model)
        ax.plot(angles, values, linewidth=2)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1 Score'])
    plt.title(title)
    plt.legend(loc='upper right')
    plt.show()


# Gọi hàm để tạo biểu đồ radar
create_radar_chart(models, [accuracy_nb, precision_nb, recall_nb, f1_nb], 'Model Performance Comparison')


# Tạo biểu đồ nhóm
def create_grouped_bar_chart(models, accuracies, precisions, recalls, f1_scores):
    x = np.arange(len(models))  # Vị trí của các mô hình
    width = 0.2  # Chiều rộng của các thanh

    fig, ax = plt.subplots(figsize=(12, 6))

    # Vẽ các thanh cho từng chỉ số
    ax.bar(x - width * 1.5, accuracies, width, label='Accuracy', color='blue')
    ax.bar(x - width / 2, precisions, width, label='Precision', color='orange')
    ax.bar(x + width / 2, recalls, width, label='Recall', color='green')
    ax.bar(x + width * 1.5, f1_scores, width, label='F1 Score', color='red')

    # Thêm nhãn và tiêu đề
    ax.set_ylabel('Scores')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    plt.tight_layout()
    plt.show()


# Gọi hàm để tạo biểu đồ nhóm
create_grouped_bar_chart(models, accuracies, precisions, recalls, f1_scores)
