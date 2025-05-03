import numpy as np
import matplotlib.pyplot as plt

# 1. 准备数据 (模拟一些简单的线性数据)
np.random.seed(43)  # 为了可重复性
X = 2 * np.random.rand(100, 1)  # 生成100个随机的X值
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3 * X + 噪声

# 2. 计算模型参数 (使用最小二乘法)
# 添加一个列向量1，代表偏置项（截距）
X_b = np.c_[np.ones((100, 1)), X]  # 在X中添加一个全为1的列（表示偏置项）

# 最小二乘法公式：w = (X^T * X)^(-1) * X^T * y
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# 3. 输出计算得到的参数
print(f"偏置项 (w_0): {theta_best[0]}")
print(f"斜率 (w_1): {theta_best[1]}")

# 4. 使用模型进行预测
X_new = np.array([[0], [2]])  # 假设我们想预测X=0和X=2时的y值
X_new_b = np.c_[np.ones((2, 1)), X_new]  # 为X_new添加偏置项（全1列）
y_predict = X_new_b.dot(theta_best)  # 预测

# 5. 可视化结果
plt.scatter(X, y, color='blue', label='数据点')  # 绘制原始数据
plt.plot(X_new, y_predict, color='red', label='拟合直线')  # 绘制拟合直线
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
