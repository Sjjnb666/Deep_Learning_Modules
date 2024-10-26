import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self):
        """
        初始化线性回归模型
        """
        self.w = None
        self.learning_rate = None
        self.epochs = None

    def fit(self, X, y, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs
        # 转换X和y的维度
        X = X.values
        y = y.values.reshape(-1, 1)
        ones = np.ones((X.shape[0], 1))
        X_b = np.hstack((ones, X))

        # 初始化参数w
        self.w = np.random.randn(X_b.shape[1], 1)
        m = X_b.shape[0]

        # 梯度下降
        for iteration in range(self.epochs):
            y_pred = np.dot(X_b, self.w)
            gradients = 2 / m * np.dot(X_b.T, (y_pred - y))
            self.w -= self.learning_rate * gradients

    def predict(self, X):
        X = X.values
        ones = np.ones((X.shape[0], 1))
        X_b = np.hstack((ones, X))
        return np.dot(X_b, self.w)

    def MSEloss(self, X, y):
        y_pred = self.predict(X)
        y_true = y.values.reshape(-1, 1)
        loss = np.mean((y_true - y_pred) ** 2)
        return loss

    def plot(self, X, y, X_new, y_pred):
        plt.scatter(X['Feature1'], y, color='blue', label='Actual')
        plt.plot(X_new['Feature1'], y_pred, color='red', label='Predicted')
        plt.xlabel('Feature1')
        plt.ylabel('Target')
        plt.title('Linear Regression Prediction')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    # 生成示例数据
    import pandas as pd
    import numpy as np

    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)  # 100 个样本，每个样本 1 个特征
    y = 4 + 3 * X[:, 0] + np.random.randn(100)

    # 转换为 pandas DataFrame 和 Series
    X = pd.DataFrame(X, columns=['Feature1'])
    y = pd.Series(y)

    # 创建模型实例
    model = LinearRegression()

    # 训练模型
    model.fit(X, y, learning_rate=0.01, epochs=1000)

    # 计算训练精度
    mse = model.MSEloss(X, y)
    print(f"MSE: {mse}")

    # 进行预测
    X_new = pd.DataFrame({
        'Feature1': np.linspace(0, 2, 100)
    })
    y_predict = model.predict(X_new)

    # 绘制预测图
    model.plot(X, y, X_new, y_predict)
