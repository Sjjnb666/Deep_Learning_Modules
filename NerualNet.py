import numpy as np
import GPU
import torch
import torch.nn.functional as F
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Linear:
    def __init__(self, in_features, out_features, bias=True, activation='relu', init_method='normal', device=None):
        """
        初始化线性层
        :param in_features: 输入特征数
        :param out_features: 输出特征数
        :param bias: 是否需要偏置
        :param activation: 激活函数(relu,tanh,sigmoid,None)
        :param init_method: 参数初始化(normal,xavier,kai_ming)
        :param device: 计算设备
        """
        self.output = None
        self.input = None
        self.bias_grad = None
        self.weights_grad = None
        self.in_features = in_features
        self.out_features = out_features
        self.device = device if device else get_device()
        self.activation = activation
        self.loss = None
        self.epochs = None

        # 参数初始化
        self.weights = self.init_weights(init_method)
        self.bias = torch.zeros(out_features, device=self.device, requires_grad=True) if bias else None

    def init_weights(self, init_method):
        """
        :param init_method: 参数初始化方式
        :return: 初始化好的权重向量
        """
        if init_method == 'normal':
            return torch.randn(self.out_features, self.in_features, device=self.device, requires_grad=True) * 0.01
        elif init_method == 'xavier':
            return torch.randn(self.out_features, self.in_features, device=self.device, requires_grad=True) * np.sqrt(
                1. / self.in_features)
        elif init_method == 'kai_ming':
            return torch.randn(self.out_features, self.in_features, device=self.device, requires_grad=True) * np.sqrt(
                2. / self.in_features)
        else:
            raise ValueError("Unknown initialization method")

    def forward(self, x):
        """
        前向传播
        :param x: 输入张量
        :return: 输出张量
        """
        self.input = x
        self.output = x.mm(self.weights.T)
        if self.bias is not None:
            self.output += self.bias
        if self.activation == 'relu':
            self.output = torch.relu(self.output)
        elif self.activation == 'tanh':
            self.output = torch.tanh(self.output)
        elif self.activation == 'sigmoid':
            self.output = torch.sigmoid(self.output)
        return self.output

    def compute_loss(self, output, target):
        """
        计算损失和梯度
        :param output: 模型输出
        :param target: 目标值
        :return: 损失和损失相对于输出的梯度
        """
        if self.loss == 'mse':
            loss = F.mse_loss(output, target)
            grad_output = 2 * (output - target) / target.size(0)
        elif self.loss == 'cross_entropy':
            loss = F.cross_entropy(output, target)
            grad_output = -(target / output - (1 - target) / (1 - output)) / target.size(0)
        else:
            raise ValueError(f"Unsupported loss function: {self.loss}")
        return loss, grad_output

    def backward(self, grad_output):
        """
        :param grad_output: 输出梯度向量
        :return: 输入梯度
        """
        if self.activation == 'relu':
            grad_output = grad_output * (self.output > 0).float()
        elif self.activation == 'tanh':
            grad_output = grad_output * (1 - self.output ** 2)
        elif self.activation == 'sigmoid':
            grad_output = grad_output * self.output * (1 - self.output)

        self.weights_grad = grad_output.T.mm(self.input)
        if self.bias is not None:
            self.bias_grad = grad_output.sum(dim=0)

        grad_input = grad_output.mm(self.weights)
        return grad_input

    def update(self, lr):
        """
        更新权重
        :param lr: 学习率
        :return:
        """
        with torch.no_grad():
            self.weights -= lr * self.weights_grad
            if self.bias is not None:
                self.bias -= lr * self.bias_grad

    def get_weights(self):
        """
        获取当前的权重值
        :return: 权重值
        """
        return self.weights, self.bias

    def train(self, X, y, lr, epochs=10, loss='mse'):
        """
        训练模型
        :param loss: 损失函数
        :param epochs: 训练轮数
        :param X: 输入数据
        :param y: 目标值
        :param lr: 学习率
        :return: None
        """
        self.loss = loss
        self.epochs = epochs
        X, y = X.to(self.device), y.to(self.device)
        self.weights = self.weights.to(self.device)
        if self.bias is not None:
            self.bias = self.bias.to(self.device)

        for epoch in range(self.epochs):
            output = self.forward(X)
            loss, grad_output = self.compute_loss(output, y)
            self.weights.grad = None
            if self.bias is not None:
                self.bias_grad = None

            self.backward(grad_output)
            self.update(lr)

            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    def predict(self, X):
        """
        预测结果
        :param self:
        :param X: 输入数据
        :return: 预测结果
        """
        X = X.to(self.device)
        self.weights = self.weights.to(self.device)
        if self.bias is not None:
            self.bias = self.bias.to(self.device)

        return self.forward(X).cpu().detach().numpy()


class MLP:
    def __init__(self, layer_sizes, activation='relu', init_method='Normal', optimizer='SGD', lr=0.01, device=None):
        """
        初始化多层感知机
        :param layer_sizes:每一层的尺寸(包括输入和输出)
        :param activation:激活函数
        :param init_method:参数初始化方法
        :param optimizer:优化器
        :param lr:学习率
        :param device:计算设备
        """
        self.layers = []
        self.activation = activation
        self.init_method = init_method
        self.device = device if device else get_device()
        self.lr = lr
        self.optimizer = optimizer

        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                Linear(layer_sizes[i], layer_sizes[i + 1], activation=activation if i < len(layer_sizes) - 2 else None,
                       init_method=init_method, device=self.device))
        self.optimizers = self.init_optimizers()

    def init_optimizers(self):
        """初始化优化器"""
        optimizers = []
        for layer in self.layers:
            params = [layer.weights, layer.bias] if layer.bias is not None else [layer.weights]
            if self.optimizer == 'SGD':
                optim = torch.optim.SGD(params, lr=self.lr)
            elif self.optimizer == 'Adam':
                optim = torch.optim.Adam(params, lr=self.lr)
            elif self.optimizer == 'NAG':
                optim = torch.optim.SGD(params, lr=self.lr, momentum=0.9, nesterov=True)
            else:
                raise ValueError(f'Unknow optimizer : {self.optimizer}')
            optimizers.append(optim)
        return optimizers

    def forward(self, x):
        """
        前向传播
        :param x: 输入张量
        :return: 输出张量
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def compute_loss(self, output, target, loss_fn):
        """
        计算损失
        :param output: 模型输出
        :param target: 目标值
        :param loss_fn: 损失函数
        :return: 损失和梯度
        """
        loss, grad_output = loss_fn(output, target)
        return loss, grad_output

    def backward(self, grad_out):
        """
        反向传播
        :param grad_out: 输出梯度向量
        :return: None
        """
        for layer in reversed(self.layers):
            grad_out = layer.backward(grad_out)

    def update(self):
        for optim in self.optimizers:
            optim.step()

    def zero_grad(self):
        """梯度清零"""
        for optim in self.optimizers:
            optim.zero_grad()

    def train(self, X, y, epochs, loss='mse'):
        """
        训练模型
        :param X: 模型输入
        :param y: 目标值
        :param epochs: 训练轮数
        :param loss: 损失函数
        :return: None
        """
        loss_fn = self.get_loss_function(loss)
        X, y = X.to(self.device), y.to(self.device)
        for epoch in range(epochs):
            output = self.forward(X)
            loss, grad_output = self.compute_loss(output, y, loss_fn)
            self.zero_grad()
            self.backward(grad_output)
            self.update()
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    def predict(self, X):
        """
        预测结果
        :param X: 输入数据
        :return: 预测结果
        """
        X = X.to(self.device)
        output = self.forward(X)
        return output.cpu().detach().numpy()

    def get_loss_function(self, loss_name):
        """
        获取损失函数
        :param loss_name: 损失函数名称
        :return: 损失函数
        """
        if loss_name == 'mse':
            def loss_fn(output, target):
                loss = F.mse_loss(output, target)
                grad_output = 2 * (output - target) / target.size(0)
                return loss, grad_output
        elif loss_name == 'cross_entropy':
            def loss_fn(output, target):
                loss = F.cross_entropy(output, target)
                grad_output = -(target / output - (1 - target) / (1 - output)) / target.size(0)
                return loss, grad_output
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")

        return loss_fn


def get_device():
    """
    检测是否有可用的GPU资源
    :return: 返回可用的设备 'cuda' 或 'cpu'
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    # 加载加利福尼亚房价数据集
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    # 标准化数据
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = y.reshape(-1, 1)  # 将目标值调整为列向量
    # 拆分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 转换为Tensor
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    # print(X_train.shape,y_train.shape,X_test.shape)
    # 设备
    device = get_device()

    # Linear:
    model = Linear(in_features=X_train.shape[1], out_features=1, activation='relu', device=device)
    model.train(X_train, y_train, lr=0.01, epochs=100, loss='mse')
    predictions = model.predict(X_test)
    print(predictions)

    """
    MLP:
    model = MLP(layer_sizes=[X_train.shape[1], 5, 5, 1], activation='relu', init_method='normal', optimizer='SGD', lr=0.01,
              device=get_device()) 
    model.train(X, y, epochs=100, loss='mse')
    """


if __name__ == "__main__":
    main()
