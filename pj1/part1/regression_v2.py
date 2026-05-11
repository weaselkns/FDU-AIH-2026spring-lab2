'''
多隐层BP回归模型
'''

import os
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


def init_params(layer_dims: list[int]):
    '''
    初始化参数放入字典中, 权重使用正态分布初始化, 偏置使用0初始化
    参数：
        layer_dims(list[int]): 各层维度
    Returns:
        params(dict): 权重与偏置参数(其中b[i]需要保持一维数组形式)
            W[i](np.ndarray(layer_dims[i], layer_dims[i+1])): 第i层到第i+1层的权重
            b[i](np.ndarray(1, layer_dims[i+1])): 第i层的偏置
    '''
    params = {}
    for i in range(len(layer_dims) - 1):
        W = np.random.randn(layer_dims[i], layer_dims[i+1]) * np.sqrt(1 / layer_dims[i])
        b = np.zeros((1, layer_dims[i+1]))
        params[f"W{i}"] = W
        params[f"b{i}"] = b
    return params


def forward(x: np.ndarray, layer_dims: list[int], params: dict):
    '''
    前向传播
    参数：
        x(np.ndarray(n_samples, 1)): 输入数据
        params(dict): 权重与偏置参数
    Returns:
        cache(dict): 中间变量
            z[i](np.ndarray(n_samples, layer_dims[i+1])): 第i层线性输出(最后一层即 y_pred, 无 tanh)
            a[i](np.ndarray(n_samples, layer_dims[i+1])): 第i层 tanh 激活（仅隐层有）
    '''
    cache = {}
    a = x
    L = len(layer_dims) - 1 # 层数
    for i in range(L):
        W = params[f"W{i}"]
        b = params[f"b{i}"]
        z = a @ W + b
        cache[f"z{i}"] = z
        if i < L - 1:
            a = np.tanh(z)
            cache[f"a{i}"] = a
        else:
            cache["y_pred"] = z
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


def backward(x: np.ndarray, y_true: np.ndarray, layer_dims: list[int], cache: dict, params: dict):
    '''
    反向传播
    参数:
        x(np.ndarray(n_samples, input_dim)): 输入数据
        y_true(np.ndarray(n_samples, output_dim)): 真实值
        layer_dims: 各层宽度，长度为 L+1
        cache: forward 写入的 z[i]、a[i]、y_pred
        params: W{i}, b{i}
    Returns:
        grads(dict): 梯度
    '''
    L = len(layer_dims) - 1
    y_pred = cache[f"z{L - 1}"]
    n = y_pred.shape[0]
    dz = (y_pred - y_true) * (2.0 / n)
    grads = {}
    for i in range(L - 1, -1, -1):
        if i == 0:
            a_prev = x
        else:
            a_prev = cache[f"a{i - 1}"]
        W = params[f"W{i}"]
        grads[f"W{i}"] = a_prev.T @ dz
        grads[f"b{i}"] = np.sum(dz, axis=0, keepdims=True)
        if i > 0:
            dz = (dz @ W.T) * (1.0 - cache[f"a{i - 1}"] ** 2)
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
    i = 0
    while f"W{i}" in params:
        params[f"W{i}"] -= lr * grads[f"W{i}"]
        params[f"b{i}"] -= lr * grads[f"b{i}"]
        i += 1
    return params


def train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    layer_dims: list[int] = [1, 32, 1],
    lr: float = 1e-3,
    epochs: int = 2000,
    epoch_print: int = 100,
    return_loss_history: bool = False,
):
    '''
    训练
    参数:
        x_train(np.ndarray(n_samples, 1)): 训练数据
        y_train(np.ndarray(n_samples, 1)): 训练标签
        layer_dims(list[int]): 各层维度
        lr(float): 学习率
        epochs(int): 训练轮数
        epoch_print(int): 每多少轮打印一次
        return_loss_history: 为True时返回每步的MSE列表
    Returns:
        params(dict) 或 (params, loss_history)
    '''
    params = init_params(layer_dims)
    loss_history = []
    for epoch in range(epochs):
        cache = forward(x_train, layer_dims, params)
        y_pred = cache["y_pred"]
        loss = mse_loss(y_pred, y_train)
        if return_loss_history:
            loss_history.append(loss)
        grads = backward(x_train, y_train, layer_dims, cache, params)
        params = sgd_step(params, grads, lr)
        if (epoch + 1) % epoch_print == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")
    if return_loss_history:
        return params, loss_history
    return params


def predict(x: np.ndarray, layer_dims: list[int], params: dict):
    '''
    预测
    参数:
        x(np.ndarray(n_samples, 1)): 输入数据
        layer_dims: 各层维度
        params(dict): 权重与偏置参数
    Returns:
        y_pred(np.ndarray(n_samples, 1)): 预测值
    '''
    cache = forward(x, layer_dims, params)
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
    out_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "regression_fit.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print("Saved:", out_path)
    plt.show()
    print("Finish plotting.")


def main():
    set_seed(2026)
    x, y = generate_data()
    layer_dims = [1, 64, 1]
    params = train(x_train=x, y_train=y, layer_dims=layer_dims, lr=2e-2, epochs=20000, epoch_print=1000)
    y_pred = predict(x, layer_dims, params)
    plot_fit(x, y, y_pred)


if __name__ == "__main__":
    main()
