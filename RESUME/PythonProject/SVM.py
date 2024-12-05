import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import matplotlib.pyplot as plt
import seaborn as sns

# 假设数据集已下载并保存在当前工作目录中
df = pd.read_csv('train.csv')

# 查看数据的基本信息
print("Data shape:", df.shape)

# 特征列是前 54 列（从0到53），目标列是第55列
X = df.iloc[:, :-1].values  # 特征
y = df.iloc[:, -1].values  # 目标变量（森林类型）

# 对特征进行标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 设置K折交叉验证，k=5
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 初始化SVM模型
svm = SVC(kernel='rbf')

# 存储每次交叉验证的结果
accuracies = []
y_true_all = []  # 存储所有折叠的真实标签
y_pred_all = []  # 存储所有折叠的预测标签

# 使用 tqdm 显示 K-fold 的进度
start_time = time.time()
with tqdm(total=5, desc="K-fold Cross Validation", unit="fold") as pbar:
    for train_index, test_index in kf.split(X):
        # 获取训练集和测试集
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # 训练 SVM 模型
        svm.fit(X_train, y_train)

        # 预测
        y_pred = svm.predict(X_test)

        # 计算并存储准确率
        accuracies.append(accuracy_score(y_test, y_pred))

        # 存储每一折的真实标签和预测标签
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

        # 更新进度条
        pbar.update(1)

end_time = time.time()

# 输出交叉验证结果
# print(f"Cross-validation completed in {end_time - start_time:.2f} seconds.")
print(f"Average accuracy across all folds: {np.mean(accuracies):.4f}")
print(f"Standard deviation of accuracy: {np.std(accuracies):.4f}")

# 输出综合分类报告和混淆矩阵
print("Overall Classification Report:\n", classification_report(y_true_all, y_pred_all))
print("Overall Confusion Matrix:\n", confusion_matrix(y_true_all, y_pred_all))

# 1. 绘制准确率的条形图
plt.figure(figsize=(8, 6))
plt.bar(range(1, 6), accuracies, color='skyblue', edgecolor='black')
plt.xlabel('Fold Number')
plt.ylabel('Accuracy')
plt.title('Accuracy for Each Fold in 5-Fold Cross-Validation')
plt.xticks(range(1, 6))
plt.ylim(0, 1)
plt.show()

# 2. 绘制混淆矩阵的热力图
conf_matrix = confusion_matrix(y_true_all, y_pred_all)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
