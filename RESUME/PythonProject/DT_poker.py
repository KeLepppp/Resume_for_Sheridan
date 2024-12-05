import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import torch.nn as nn
import torch
# 加载数据
df = pd.read_csv('poker-hand-testing.data')

# 查看数据的前几行，了解数据结构
# print(df.head())

# 特征列是前 54 列（从0到53），目标列是第55列
X = df.iloc[:, :-1].values  # 特征
y = df.iloc[:, -1].values   # 目标变量（森林类型）

# 拆分数据集为训练集和测试集（80%训练，20%测试）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 对特征进行标准化，因为一些算法（例如SVM）对特征尺度敏感，但随机森林本身对标准化不敏感
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # 使用训练集的scaler来转换测试集

# 定义RandomForestClassifier并设置指定的超参数
rf_classifier = RandomForestClassifier(
    n_estimators=100,          # 减少决策树数量
    max_depth=10,              # 限制树的最大深度
    min_samples_split=5,     # 增加节点划分时的最小样本数
    min_samples_leaf=5,       # 增加叶子节点的最小样本数
    bootstrap=True,           # 使用bootstrap（可根据需要设置为False）
    random_state=42,          # 固定随机数种子，确保结果可复现
    n_jobs=-1,                # 使用所有CPU核心并行训练
    warm_start=True,          # 允许温启动，节省训练时间
    max_samples=0.8           # 每棵树使用80%的训练数据
)


# 记录开始时间
start_time = time.time()

# 使用 tqdm 显示进度条
# 训练过程会进行 100 次树的训练，并显示进度
for _ in tqdm(range(100), desc="Training Random Forest", ncols=100):
    rf_classifier.fit(X_train, y_train)

# 训练模型
rf_classifier.fit(X_train, y_train)

# 预测测试集
y_pred = rf_classifier.predict(X_test)

# 打印准确率
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')

# 打印分类报告
print("Classification Report:\n", classification_report(y_test, y_pred))

# 打印混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# 记录结束时间
end_time = time.time()
print(f"Training time: {end_time - start_time:.2f} seconds")

# 画出混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
