from collections import Counter

import numpy as np
import pandas as pd


# 计算熵 (Entropy)
def entropy(y):
    # 计算类别的分布
    class_counts = Counter(y)
    total = len(y)
    return -sum((count / total) * np.log2(count / total) for count in class_counts.values())


# 计算信息增益
def information_gain(X, y, feature_index):
    # 计算数据集的熵
    total_entropy = entropy(y)

    # 获取特征列
    feature_values = X[:, feature_index]
    unique_values = np.unique(feature_values)

    # 计算划分后的熵
    weighted_entropy = 0
    for value in unique_values:
        # 找到特征值为value的子集
        subset_y = y[feature_values == value]
        weighted_entropy += (len(subset_y) / len(y)) * entropy(subset_y)

    # 信息增益 = 总熵 - 划分后的熵
    return total_entropy - weighted_entropy


# 选择信息增益最大的特征
def best_split(X, y):
    best_feature = None
    best_gain = -1
    for feature_index in range(X.shape[1]):
        gain = information_gain(X, y, feature_index)
        if gain > best_gain:
            best_gain = gain
            best_feature = feature_index
    return best_feature


# 构建决策树
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    # 递归构建树
    def fit(self, X, y, depth=0):
        # 如果所有样本属于同一类，停止分裂
        if len(set(y)) == 1:
            return y[0]

        # 如果达到最大深度，停止分裂
        if self.max_depth is not None and depth >= self.max_depth:
            return Counter(y).most_common(1)[0][0]

        # 选择最佳分裂特征
        best_feature = best_split(X, y)

        # 创建子节点
        tree = {best_feature: {}}
        feature_values = np.unique(X[:, best_feature])

        # 递归地为每个子集构建树
        for value in feature_values:
            subset_X = X[X[:, best_feature] == value]
            subset_y = y[X[:, best_feature] == value]
            tree[best_feature][value] = self.fit(subset_X, subset_y, depth + 1)

        return tree

    # 预测
    def predict(self, X):
        return [self._predict_single(sample) for sample in X]

    # 预测单个样本
    def _predict_single(self, sample):
        tree = self.tree
        while isinstance(tree, dict):
            feature_index = list(tree.keys())[0]
            feature_value = sample[feature_index]
            tree = tree[feature_index].get(feature_value, None)
            if tree is None:
                return None
        return tree


# 测试决策树
if __name__ == '__main__':
    # 模拟一个数据集
    data = {
        'Feature1': [1, 1, 2, 2, 3, 3],
        'Feature2': [1, 2, 1, 2, 1, 2],
        'Label': [0, 0, 1, 1, 0, 1]
    }

    df = pd.DataFrame(data)
    X = df[['Feature1', 'Feature2']].values
    y = df['Label'].values

    # 构建决策树并训练
    tree = DecisionTree(max_depth=3)
    tree.tree = tree.fit(X, y)

    print("决策树结构：")
    print(tree.tree)

    # 预测
    predictions = tree.predict(X)
    print("\n预测结果：", predictions)
