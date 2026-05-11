import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


def set_seed(seed: int = 42):
    '''
    设置随机种子
    参数：
        seed: 随机种子
    '''
    np.random.seed(seed)


def load_data(
    train_dir: str | None = None,
    val_ratio: float = 0.1,
):
    '''
    读取数据并划分训练集/验证集(先打乱再切分)
    参数：
        train_dir: 训练数据根目录
        val_ratio: 验证集占比
    Returns:
        X_train(np.ndarray(N_train, 784)), X_val(np.ndarray(N_val, 784))
        Y_train(np.ndarray(N_train, 12)), Y_val(np.ndarray(N_val, 12))
    '''
    print("Loading data...")
    # 读取训练数据目录
    if train_dir is None:
        train_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "train"))
    else:
        train_dir = os.path.abspath(train_dir)
    # 读取类别名称
    class_names = []
    for entry in os.listdir(train_dir):
        full = os.path.join(train_dir, entry)
        if os.path.isdir(full):
            class_names.append(entry)
    class_names.sort(key=int) # 按数字排序
    num_classes = len(class_names)
    X_list = []
    Y_list = []
    # 读取每个类别的图像
    for i in range(num_classes):
        class_dir = os.path.join(train_dir, class_names[i])
        # 读取类别目录下的所有图像
        file_names = []
        for name in os.listdir(class_dir):
            file_names.append(name)
        file_names.sort()
        # 读取每个图像
        for name in file_names:
            path = os.path.join(class_dir, name)
            img = Image.open(path)
            pixel = np.asarray(img, dtype=np.float64).reshape(-1) # 将图像展平
            label = np.zeros(num_classes, dtype=np.float64) # one-hot编码
            label[i] = 1.0
            # 将图像和标签添加到列表中
            X_list.append(pixel)
            Y_list.append(label)
    # 将样本列表转换为二维数组
    X = np.stack(X_list, axis=0) # (N, 784)
    Y = np.stack(Y_list, axis=0) # (N, 12)
    # 打乱数据
    n = X.shape[0]
    rng = np.random.default_rng(seed=2026) # 随机数生成器
    perm = rng.permutation(n)
    X, Y = X[perm], Y[perm]

    # 划分验证集
    if val_ratio == 0:
        return X, Y, None, None
    n_val = int(round(n * val_ratio))
    n_val = max(1, min(n_val, n - 1))
    X_val, Y_val = X[:n_val], Y[:n_val]
    X_train, Y_train = X[n_val:], Y[n_val:]
    print("Loading data done.")
    return X_train, Y_train, X_val, Y_val


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


def softmax(z: np.ndarray):
    '''
     softmax 函数
    参数：
        z(np.ndarray(N, O)): 线性输出
    Returns:
        probs(np.ndarray(N, O)): 概率
    '''
    probs = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
    return probs


def forward(X: np.ndarray, layer_dims: list[int], params: dict):
    '''
    前向传播
    参数：
        X(np.ndarray(N, D)): 输入数据
        params(dict): 权重与偏置参数
    Returns:
        cache(dict): 中间变量
            z[i](np.ndarray(N, layer_dims[i+1])): 第i层线性输出(最后一层即 Y_pred, 无 tanh)
            a[i](np.ndarray(N, layer_dims[i+1])): 第i层 tanh 激活（仅隐层有）
    '''
    cache = {}
    a = X
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
            cache["logits"] = z # 最后一层线性输出(无激活函数)
            cache["probs"] = softmax(z)
    return cache
    

def cross_entropy_loss(Y_pred: np.ndarray, Y_true: np.ndarray):
    '''
    交叉熵损失
    参数：
        Y_pred(np.ndarray(N, O)): softmax概率
        Y_true(np.ndarray(N, O)): 真实值one-hot
    Returns:
        loss(float): 交叉熵损失
    '''
    loss = -np.sum(Y_true * np.log(Y_pred)) / Y_pred.shape[0]
    return loss


def backward(x: np.ndarray, y_true: np.ndarray, layer_dims: list[int], cache: dict, params: dict):
    '''
    反向传播
    参数:
        x(np.ndarray(n_samples, input_dim)): 输入数据
        y_true(np.ndarray(n_samples, output_dim)): one-hot真实标签
        layer_dims: 各层宽度, 长度为L+1
        cache: z[i], a[i], logits, probs
        params: W{i}, b{i}
    Returns:
        grads(dict): 梯度
    '''
    L = len(layer_dims) - 1
    probs = cache["probs"]
    n = probs.shape[0]
    dz = (probs - y_true) / n
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
    x_val: np.ndarray,
    y_val: np.ndarray,
    layer_dims: list[int] = [1, 32, 1],
    lr: float = 1e-2,
    epochs: int = 2000,
    epoch_print: int = 100
):
    '''
    训练
    参数:
        x_train(np.ndarray(N, D)): 训练数据
        y_train(np.ndarray(N, O)): 训练标签one-hot
        layer_dims(list[int]): 各层维度
        lr(float): 学习率
        epochs(int): 训练轮数
        epoch_print(int): 每多少轮打印一次
    Returns:
        params(dict): 权重与偏置参数
    '''
    params = init_params(layer_dims)
    for epoch in range(epochs):
        cache = forward(x_train, layer_dims, params)
        loss = cross_entropy_loss(cache["probs"], y_train)
        grads = backward(x_train, y_train, layer_dims, cache, params)
        params = sgd_step(params, grads, lr)
        if (epoch + 1) % epoch_print == 0:
            y_val_pred = predict(x_val, layer_dims, params)
            y_val_acc = np.argmax(y_val, axis=1)
            acc_val = np.mean(y_val_pred == y_val_acc)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}, Val Accuracy: {acc_val}")
    return params


def predict(x: np.ndarray, layer_dims: list[int], params: dict):
    '''
    预测
    参数:
        x(np.ndarray(N, D)): 输入数据
        layer_dims: 各层维度
        params(dict): 权重与偏置参数
    Returns:
        labels(np.ndarray(N,)): 每个样本概率最大的类别下标(0 .. num_classes-1)
    '''
    cache = forward(x, layer_dims, params)
    return np.argmax(cache["probs"], axis=1)


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


def save_params(params: dict, layer_dims: list[int], path: str = 'classification_params.npz'):
    '''保存模型参数到文件'''
    save_dict = {k: v for k, v in params.items()}
    save_dict['_layer_dims'] = np.array(layer_dims)
    np.savez(path, **save_dict)
    print(f'参数已保存到 {path}')


def load_params(path: str = 'classification_params.npz') -> tuple[dict, list[int]]:
    '''从文件加载模型参数, 返回 (params, layer_dims)'''
    data = np.load(path)
    layer_dims = data['_layer_dims'].tolist()
    params = {k: data[k] for k in data.files if not k.startswith('_')}
    print(f'参数已从 {path} 加载')
    return params, layer_dims


def interview(eval_dir: str, layer_dims: list[int], params: dict):
    '''
    对指定目录下的测试集进行批量预测并返回准确率
    参数：
        eval_dir: 测试数据根目录(与 train/ 目录结构一致)
        layer_dims: 训练时使用的各层维度
        params: 训练好的参数
    Returns:
        accuracy: 准确率 (0~100)
    '''
    if eval_dir is None:
        print("eval_dir is None")
        return
    X, Y, _, _ = load_data(train_dir=eval_dir, val_ratio=0)
    preds = predict(X, layer_dims, params)
    true_labels = np.argmax(Y, axis=1)
    accuracy = float(np.mean(preds == true_labels)) * 100
    print(f'测试准确率: {accuracy:.2f}%')
    return accuracy


def main():
    set_seed(2026)
    X_train, Y_train, X_val, Y_val = load_data()
    print(
        "train:", X_train.shape, Y_train.shape,
        "val:", X_val.shape, Y_val.shape
    )
    layer_dims = [784, 64, 32, 12]
    params = train(x_train=X_train, y_train=Y_train, x_val=X_val, y_val=Y_val, layer_dims=layer_dims, lr=2e-3, epochs=40000, epoch_print=250)
    save_params(params, layer_dims)


if __name__ == "__main__":
    main()
