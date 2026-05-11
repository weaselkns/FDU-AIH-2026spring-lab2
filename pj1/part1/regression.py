'''
单隐层BP回归模型
'''


import numpy as np
import matplotlib.pyplot as plt


def set_seed(seed: int = 42):
    '''
    设置随机种子
    参数：
        seed: 随机种子
    '''
    np.random.seed(seed)


def generate_data(n_samples: int = 256, x_min: float = -np.pi, x_max: float = np.pi):
    '''
    生成正弦函数的数据
    参数：
        n_samples: 样本数量
        x_min: x的最小值
        x_max: x的最大值
    Returns:
        x(np.ndarray(n_samples, 1)): 自变量(是否均匀采样需要根据训练效果选择, linspace 为均匀采样, uniform 为非均匀采样)
        y(np.ndarray(n_samples, 1)): 因变量
    '''
    x = np.linspace(x_min, x_max, n_samples)
    y = np.sin(x)
    # 用reshape将一维数组转换为二维数组
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    # print(type(x), x.shape, type(y), y.shape)
    return x, y


def init_params(input_dim: int, hidden_dim: int, output_dim: int):
    '''
    初始化参数放入字典中, 权重使用正态分布初始化, 偏置使用0初始化
    参数：
        input_dim: 输入维度
        hidden_dim: 隐藏维度
        output_dim: 输出维度
    Returns:
        params(dict): 权重与偏置参数(其中b1和b2需要保持一维数组形式)
            W1(np.ndarray(input_dim, hidden_dim))、b1(np.ndarray(1, hidden_dim)): 输入层到隐藏层的权重与偏置
            W2(np.ndarray(hidden_dim, output_dim))、b2(np.ndarray(1, output_dim)): 隐藏层到输出层的权重与偏置
    '''
    W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(1 / input_dim)
    b1 = np.zeros((1, hidden_dim))
    W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(1 / hidden_dim)
    b2 = np.zeros((input_dim, output_dim))
    params = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    # print(type(params), params.keys(), params["W1"].shape, params["b1"].shape, params["W2"].shape, params["b2"].shape)
    return params


def forward(x: np.ndarray, params: dict):
    '''
    前向传播
    参数：
        x(np.ndarray): 输入数据
        params(dict): 权重与偏置参数
    Returns:
        cache(dict): 缓存
            z1(np.ndarray(n_samples, hidden_dim))
            a1(np.ndarray(n_samples, hidden_dim))
            y_pred(np.ndarray(n_samples, 1))
    '''
    W1, W2, b1, b2 = params["W1"], params["W2"], params["b1"], params["b2"]
    z1 = x @ W1 + b1
    a1 = np.tanh(z1)
    y_pred = a1 @ W2 + b2
    cache = {
        "z1": z1,
        "a1": a1,
        "y_pred": y_pred
    }
    return cache

def mse_loss(y_pred: np.ndarray, y_true: np.ndarray):
    '''
    均方误差
    参数：
        y_pred(np.ndarray(n_samples, 1)): 预测值
        y_true(np.ndarray(n_samples, 1)): 真实值
    Returns:
        loss(float): 均方误差
    '''
    loss = np.mean((y_pred - y_true) ** 2)
    return loss


def backward(x: np.ndarray, y_true: np.ndarray, cache: dict, params: dict):
    '''
    反向传播
    参数:
        x(np.ndarray(n_samples, 1)): 输入数据
        y_true(np.ndarray(n_samples, 1)): 真实值
        cache(dict): 缓存
            z1(np.ndarray(n_samples, hidden_dim))
            a1(np.ndarray(n_samples, hidden_dim))
            y_pred(np.ndarray(n_samples, 1))
        params(dict): 权重与偏置参数
    Returns:
        grads(dict): 梯度
    '''
    W1, W2, b1, b2 = params["W1"], params["W2"], params["b1"], params["b2"]
    z1, a1, y_pred = cache["z1"], cache["a1"], cache["y_pred"]
    dz2 = (y_pred - y_true) * 2 / y_pred.shape[0]
    da1 = dz2 @ W2.T
    dz1 = da1 * (1 - a1 ** 2)
    dw2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True) # axis = 0 代表按行求和
    dw1 = x.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)
    grads = {
        "W1": dw1,
        "b1": db1,
        "W2": dw2,
        "b2": db2
    }
    return grads


def sgd_step(params: dict, grads: dict, lr: float):
    '''
    梯度下降
    参数:
        params(dict): 权重与偏置参数
        grads(dict): 梯度
        lr(float): 学习率
    Returns:
        params(dict): 更新后的权重与偏置参数
    '''
    W1, W2, b1, b2 = params["W1"], params["W2"], params["b1"], params["b2"]
    W1 -= lr * grads["W1"]
    b1 -= lr * grads["b1"]
    W2 -= lr * grads["W2"]
    b2 -= lr * grads["b2"]
    params = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    return params


def train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    hidden_dim: int = 32,
    lr: float = 1e-3,
    epochs: int = 2000,
    epoch_print: int = 100
):
    '''
    训练
    参数:
        x_train(np.ndarray(n_samples, 1)): 训练数据
        y_train(np.ndarray(n_samples, 1)): 训练标签
        hidden_dim(int): 隐藏层维度
        lr(float): 学习率
        epochs(int): 训练轮数
    Returns:
        params(dict): 权重与偏置参数
    '''
    input_dim = x_train.shape[1]
    output_dim = y_train.shape[1]
    params = init_params(input_dim, hidden_dim, output_dim)
    for epoch in range(epochs):
        cache = forward(x_train, params)
        y_pred = cache["y_pred"]
        loss = mse_loss(y_pred, y_train)
        grads = backward(x_train, y_train, cache, params)
        params = sgd_step(params, grads, lr)
        if (epoch + 1) % epoch_print == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")
    return params


def predict(x: np.ndarray, params: dict):
    '''
    预测
    参数:
        x(np.ndarray(n_samples, 1)): 输入数据
        params(dict): 权重与偏置参数
    Returns:
        y_pred(np.ndarray(n_samples, 1)): 预测值
    '''
    cache = forward(x, params)
    y_pred = cache["y_pred"]
    return y_pred


def plot_fit(x: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray):
    '''
    绘制拟合曲线
    参数:
        x(np.ndarray(n_samples, 1)): 输入数据
        y_true(np.ndarray(n_samples, 1)): 真实值
        y_pred(np.ndarray(n_samples, 1)): 预测值
    '''
    plt.plot(x, y_true, label="True")
    plt.plot(x, y_pred, label="Pred")
    plt.legend()
    plt.show()
    print("Finish plotting.")

def main():
    set_seed(2026)
    x, y = generate_data()
    params = train(x_train=x, y_train=y, hidden_dim=32, lr=2e-2, epochs=200000, epoch_print=1000)
    y_pred = predict(x, params)
    plot_fit(x, y, y_pred)


if __name__ == "__main__":
    main()
