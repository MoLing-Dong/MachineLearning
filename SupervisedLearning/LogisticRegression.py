import numpy as np
import matplotlib.pyplot as plt

# 生成数据：两类数据，类别0和类别1
np.random.seed(42)
m = 100  # 样本数量
X = np.random.randn(m, 2)  # 生成100个二维数据点
y = (X[:, 0] + X[:, 1] > 0).astype(int)  # 如果 x1 + x2 > 0，则为1，否则为0

# 绘制数据分布
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Generated Data')
plt.show()


# 逻辑回归的Sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 损失函数（交叉熵损失）
def compute_cost(X, y, theta):
    m = len(y)
    z = X.dot(theta)
    h = sigmoid(z)
    cost = -1 / m * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost


# 梯度下降
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        # 计算预测
        z = X.dot(theta)
        h = sigmoid(z)

        # 计算梯度
        gradient = (1 / m) * X.T.dot(h - y)

        # 更新theta
        theta -= learning_rate * gradient

        # 计算当前的损失
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

    return theta, cost_history


# 数据预处理：添加偏置项（全为1的列）
X_b = np.c_[np.ones((m, 1)), X]  # 在X中添加偏置项
theta_initial = np.zeros(X_b.shape[1])  # 初始化theta为0

# 设置学习率和迭代次数
learning_rate = 0.1
iterations = 1000

# 使用梯度下降法训练模型
theta_optimal, cost_history = gradient_descent(X_b, y, theta_initial, learning_rate, iterations)

# 打印最终的theta值
print("Optimal theta:", theta_optimal)

# 绘制损失函数的变化过程
plt.plot(range(iterations), cost_history, color='blue')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost function during training')
plt.show()


# 预测函数
def predict(X, theta):
    z = X.dot(theta)
    return sigmoid(z) >= 0.5


# 使用训练好的模型进行预测
y_pred = predict(X_b, theta_optimal)

# 计算准确率
accuracy = np.mean(y_pred == y) * 100
print(f'Accuracy: {accuracy:.2f}%')

# 绘制决策边界
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = predict(np.c_[np.ones((xx.ravel().shape[0], 1)), xx.ravel(), yy.ravel()], theta_optimal)
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.Paired)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary')
plt.show()
