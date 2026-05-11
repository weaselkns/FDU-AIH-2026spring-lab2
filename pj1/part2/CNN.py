'''
数据布局: 图像张量为 (N, C, H, W); 全连接输入为 (N, D)。
其中: N 为 batch 大小, C 为通道数, H 为图像高度, W 为图像宽度, D 为全连接输入特征维度。
'''

import os
import numpy as np
from PIL import Image


def set_seed(seed: int = 42):
    '''
    设置随机种子
    参数：
        seed: 随机种子
    '''
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# im2col / col2im（卷积前向/反向的向量化实现）
# ---------------------------------------------------------------------------

def images_flat_to_nchw(X: np.ndarray, h: int = 28, w: int = 28) -> np.ndarray:
    '''
    将展平的图像 (N, H*W) 转为 (N, 1, H, W)
    参数：
        X(np.ndarray(N, H*W)): 展平灰度图
        h, w: 高、宽
    Returns:
        X4(np.ndarray(N, 1, H, W))
    '''
    n = X.shape[0]
    return X.reshape(n, 1, h, w)


def im2col(
    X: np.ndarray,
    kH: int,
    kW: int,
    stride: int,
    pad: int,
) -> tuple[np.ndarray, int, int, np.ndarray]:
    '''
    将 (N,C,H,W) 展开为列矩阵, 便于矩阵乘实现卷积。
    参数：
        X(np.ndarray(N, C, H, W))
        kH, kW: 卷积核高、宽
        stride: 步长
        pad: 四周零填充像素数
    Returns:
        col(np.ndarray(C*kH*kW, N*out_h*out_w))
        out_h, out_w: 输出特征图高、宽
        X_pad(np.ndarray(N, C, H+2pad, W+2pad)): 填充后的输入, 供反向传播使用
    '''
    N, C, H, W = X.shape
    # 在输入图像的四周填充零, 补完后的形状为 (N, C, H+2pad, W+2pad)
    X_pad = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    # 计算输出特征图的形状
    out_h = (H + 2 * pad - kH) // stride + 1
    out_w = (W + 2 * pad - kW) // stride + 1
    # 初始化输出列矩阵, 行数为一个 patch 所有数拉直后的长度, 列数为 patch 的个数
    col = np.zeros((C * kH * kW, N * out_h * out_w), dtype=X.dtype)
    # 遍历输出特征图的每个位置, 计算每个位置的 patch, 并将其拉直后存储到 col 中
    for oh in range(out_h):
        for ow in range(out_w):
            # 计算每个位置的 patch, 形状为 (N, C, kH, kW), 即 N 张图在这个位置的整块区域
            patch = X_pad[:, :, oh * stride : oh * stride + kH, ow * stride : ow * stride + kW]
            # patch.reshape(N, -1).T 将 patch 拉直后转置, 得到一个形状为 (N, C*kH*kW) 的矩阵
            # 将这个矩阵存储到 col 中, 存储的位置为 (oh * out_w + ow) * N : (oh * out_w + ow + 1) * N
            # 储存顺序由内到外为 N -> ow -> oh
            col[:, (oh * out_w + ow) * N : (oh * out_w + ow + 1) * N] = patch.reshape(N, -1).T
    return col, out_h, out_w, X_pad


def col2im(
    col: np.ndarray,
    x_shape: tuple[int, int, int, int],
    kH: int,
    kW: int,
    stride: int,
    pad: int,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    '''
    将梯度累加回 (N,C,H+2pad,W+2pad), 调用方再去掉 pad。
    参数：
        col: 与 im2col 输出同形状, 形状为 (C*kH*kW, N*out_h*out_w), 是对 im2col 的 col 的梯度
        x_shape: 原始 X 的 (N, C, H, W)
    Returns:
        dX_pad(np.ndarray(N, C, H+2pad, W+2pad))
    '''
    N, C, H, W = x_shape
    Hp = H + 2 * pad
    Wp = W + 2 * pad
    # 初始化梯度累加的输入, 形状为 (N, C, H+2pad, W+2pad)
    dX_pad = np.zeros((N, C, Hp, Wp), dtype=col.dtype)
    for oh in range(out_h):
        for ow in range(out_w):
            # 取出 col 中对应位置的 patch, 形状为 (C*kH*kW, N), 即 N 张图在这个位置的整块区域
            piece = col[:, (oh * out_w + ow) * N : (oh * out_w + ow + 1) * N]
            # 直接将 piece 重塑为 (N, C, kH, kW) 可能会导致梯度错位
            d_patch = piece.reshape(C, kH, kW, N).transpose(3, 0, 1, 2)
            # 将 d_patch 累加到 dX_pad 中
            dX_pad[:, :, oh * stride : oh * stride + kH, ow * stride : ow * stride + kW] += d_patch
    return dX_pad


# ---------------------------------------------------------------------------
# Conv2d
# ---------------------------------------------------------------------------

def init_conv2d_weight(C_in: int, C_out: int, kH: int, kW: int) -> np.ndarray:
    '''
    He 初始化卷积核
    参数：
        C_in, C_out: 输入/输出通道数
        kH, kW: 核大小
    Returns:
        W(np.ndarray(C_out, C_in, kH, kW))
    '''
    # 上一层神经元个数
    fan_in = C_in * kH * kW
    # 使用 He 初始化, 方差为 2/fan_in
    return np.random.randn(C_out, C_in, kH, kW) * np.sqrt(2.0 / fan_in)


def init_bias_conv(C_out: int) -> np.ndarray:
    '''
    卷积偏置, 形状便于广播到 (N, C_out, out_h, out_w)
    Returns:
        b(np.ndarray(1, C_out, 1, 1))
    '''
    # 每个输出通道一个标量，对所有空间位置和 batch 样本使用相同的偏置
    return np.zeros((1, C_out, 1, 1), dtype=np.float64)


def conv2d_forward(
    X: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    stride: int = 1,
    pad: int = 0,
) -> tuple[np.ndarray, dict]:
    '''
    卷积层前向传播: out = conv(X, W) + b
    参数：
        X(np.ndarray(N, C_in, H, W))
        W(np.ndarray(C_out, C_in, kH, kW))
        b(np.ndarray(1, C_out, 1, 1)) 或与 C_out 可广播
        stride, pad: 步长与四周零填充
    Returns:
        out(np.ndarray(N, C_out, out_h, out_w))
        cache: 反向传播所需的缓存
    '''
    N = X.shape[0]
    C_out, C_in, kH, kW = W.shape
    # 将输入图像展平为列矩阵, 并返回输出特征图的形状
    col, out_h, out_w, X_pad = im2col(X, kH, kW, stride, pad)
    # 将卷积核展平为行矩阵, 形状为 (C_out, C_in * kH * kW)
    W_row = W.reshape(C_out, C_in * kH * kW)
    # 将卷积核与输入图像展平后的列矩阵相乘, 得到输出特征图展平后的列矩阵
    # col 的形状为 (C_in*kH*kW, N*out_h*out_w), W_row 的形状为 (C_out, C_in*kH*kW)
    out_col = W_row @ col
    # 将偏置加到输出特征图展平后的列矩阵中
    out_col = out_col + b.reshape(C_out, 1)
    # 将输出特征图展平后的列矩阵重塑为 (N, C_out, out_h, out_w)
    out = out_col.reshape(C_out, out_h, out_w, N).transpose(3, 0, 1, 2)
    # 缓存输入图像的形状, 卷积核, 输入图像展平后的列矩阵, 步长, 填充, 输出特征图的高度和宽度
    cache = {
        'X_shape': X.shape,
        'W': W,
        'col': col,
        'stride': stride,
        'pad': pad,
        'out_h': out_h,
        'out_w': out_w,
    }
    return out, cache


def conv2d_backward(d_out: np.ndarray, cache: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    卷积层反向传播
    把 d_out 先变成和 out_col 一样的排法, 再对 W_row、col、b 用矩阵乘法的求导公式, 最后 dcol 用 col2im 变回 dX 并去掉 pad
    中间变量：
        d_out_col: 将 d_out 排成 (C_out, N*out_h*out_w), 与前向 out_col 对齐
        W_row: W 展平为 (C_out, C_in*kH*kW), 与 conv2d_forward 中一致
        dW_row: 对 W_row 的梯度, d_out_col @ col.T
        dcol: 对 im2col 结果 col 的梯度, W_row.T @ d_out_col
        dX_pad: 对补零后输入的梯度 (N,C_in,H+2pad,W+2pad), 经 col2im 累加得到
    参数：
        d_out(np.ndarray(N, C_out, out_h, out_w)): 损失对卷积输出的梯度
        cache: conv2d_forward 的 cache(含 X_shape, W, col, stride, pad, out_h, out_w)
    Returns:
        dX(np.ndarray(N, C_in, H, W)): 损失对输入 X 的梯度
        dW: 损失对 W 的梯度, 与 W 同形
        db: 损失对 b 的梯度, 与 b 同形
    '''
    X_shape = cache['X_shape']
    W = cache['W']
    col = cache['col']
    stride = cache['stride']
    pad = cache['pad']
    out_h = cache['out_h']
    out_w = cache['out_w']
    N = X_shape[0]
    C_out, C_in, kH, kW = W.shape
    # 将 d_out 展平为 (C_out, N*out_h*out_w), 与 out_col 对齐
    d_out_col = d_out.transpose(1, 2, 3, 0).reshape(C_out, N * out_h * out_w)
    W_row = W.reshape(C_out, C_in * kH * kW)
    #d_out_col的形状为 (C_out, N*out_h*out_w), col.T的形状为 (N*out_h*out_w, C_in*kH*kW)
    dW_row = d_out_col @ col.T
    # 将 dW_row 重塑为 (C_out, C_in, kH, kW)
    dW = dW_row.reshape(W.shape)
    # 将 d_out_col 沿着第 1 维求和, 得到形状为 (C_out, 1, 1, 1) 的 db
    # 损失对 b[i] 的梯度
    # 特征图的维度是 (N, C_out, H, W), 所以要 reshape
    db = np.sum(d_out_col, axis=1).reshape(1, C_out, 1, 1)
    # 将 d_out_col 与 W_row 转置相乘, 得到形状为 (C_in*kH*kW, N*out_h*out_w) 的 dcol
    dcol = W_row.T @ d_out_col
    dX_pad = col2im(dcol, X_shape, kH, kW, stride, pad, out_h, out_w)
    # 如果有 padding, 删去
    if pad > 0:
        dX = dX_pad[:, :, pad:-pad, pad:-pad]
    else:
        dX = dX_pad
    return dX, dW, db


# ---------------------------------------------------------------------------
# ReLU
# ---------------------------------------------------------------------------

def relu_forward(X: np.ndarray) -> tuple[np.ndarray, dict]:
    '''
    ReLU 前向传播
    Returns:
        out, cache(mask)
    '''
    mask = X > 0
    out = X * mask
    cache = {'mask': mask}
    return out, cache


def relu_backward(d_out: np.ndarray, cache: dict) -> np.ndarray:
    '''
    ReLU 反向传播
    '''
    return d_out * cache['mask']


# ---------------------------------------------------------------------------
# MaxPool2d（非重叠: stride 与 pool 相等）
# ---------------------------------------------------------------------------

def maxpool2d_forward(
    X: np.ndarray,
    pool_h: int,
    pool_w: int,
    stride: int | None = None,
) -> tuple[np.ndarray, dict]:
    '''
    最大池化前向传播(非重叠池化)
    参数：
        X(np.ndarray(N, C, H, W))
        pool_h, pool_w: 池化窗口
        stride: 默认等于 pool_h(与 pool_w 一致)
    Returns:
        out, cache
    '''
    # 默认步长等于池化核高和宽
    if stride is None:
        stride = pool_h
    if not (stride == pool_h == pool_w):
        raise ValueError('当前实现仅支持正方形窗口，且 stride 应等于池化核大小(非重叠池化)')
    N, C, H, W = X.shape
    # 检查高和宽是否能被池化核整除
    if H % pool_h != 0 or W % pool_w != 0:
        raise ValueError('H、W 必须能被池化核整除')
    # 输出特征图的形状(须为整数, 供 reshape 使用)
    out_h, out_w = H // pool_h, W // pool_w
    # 将输入图像 reshape 为 (N, C, 第几个块的行, 块中第几行, 第几个块的列, 块中第几列)
    x_view = X.reshape(N, C, out_h, pool_h, out_w, pool_w)
    # 沿着第 3 维和第 5 维取最大值, 得到形状为 (N, C, out_h, out_w) 的输出特征图
    out = x_view.max(axis=3).max(axis=4)
    # 将输出特征图 reshape 为 (N, C, out_h, 1, out_w, 1)
    out_expanded = out[:, :, :, np.newaxis, :, np.newaxis]
    # 后向传播时，梯度应该只传递给当时最大值所在的位置
    # 所以要创建一个掩码，其形状为 (N, C, out_h, 1, out_w, 1)，值为 1 的位置为最大值所在的位置
    mask = (x_view == out_expanded).astype(np.float64)
    # 如果某个位置的最大值出现了多次，则需要将梯度平均分配到这些位置
    s = mask.sum(axis=(3, 5), keepdims=True)
    s[s == 0] = 1.0
    mask /= s
    cache = {
        'mask': mask,
        'x_view_shape': x_view.shape,
        'in_shape': X.shape,
    }
    return out, cache


def maxpool2d_backward(d_out: np.ndarray, cache: dict) -> np.ndarray:
    '''
    最大池化反向传播
    参数: 
        d_out(np.ndarray(N, C, out_h, out_w)): 上游传递回来的梯度
    '''
    mask = cache['mask']
    in_shape = cache['in_shape']
    # 将梯度与掩码相乘，得到形状为 (N, C, out_h, pool_h, out_w, pool_w) 的梯度
    d_view = mask * d_out[:, :, :, np.newaxis, :, np.newaxis]
    # 将梯度 reshape 为 (N, C, H, W), 也就是输入的梯度 dX 
    return d_view.reshape(in_shape)


# ---------------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------------

def dropout_forward(X: np.ndarray, drop_prob: float, training: bool) -> tuple[np.ndarray, dict]:
    '''
    Dropout 前向传播
    参数：
        X: 任意形状的输入
        drop_prob: 丢弃概率 (0 = 不丢弃, 0.5 = 丢弃一半)
        training: 训练模式为 True, 推理模式为 False
    Returns:
        out, cache
    '''
    if not training or drop_prob == 0.0:
        return X, {'mask': None, 'drop_prob': drop_prob}
    # 生成和 X 同形状的随机 0/1 掩码
    mask = (np.random.rand(*X.shape) > drop_prob).astype(X.dtype)
    # 计算缩放因子, 使得期望不变
    scale = 1.0 / (1.0 - drop_prob)
    out = X * mask * scale
    cache = {
        'mask': mask, 
        'drop_prob': drop_prob, 
        'scale': scale,
    }
    return out, cache


def dropout_backward(d_out: np.ndarray, cache: dict) -> np.ndarray:
    '''
    Dropout 反向传播
    '''
    if cache['mask'] is None:
        return d_out
    return d_out * cache['mask'] * cache['scale']


# ---------------------------------------------------------------------------
# Flatten + Linear（全连接）
# ---------------------------------------------------------------------------

def flatten_forward(X: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    '''
    将 (N, C, H, W) 展平为 (N, C*H*W)
    Returns:
        out, in_shape
    '''
    in_shape = X.shape
    return X.reshape(X.shape[0], -1), in_shape


def flatten_backward(d_out: np.ndarray, in_shape: tuple[int, int, int, int]) -> np.ndarray:
    '''
    Flatten 反向
    '''
    return d_out.reshape(in_shape)


def init_linear_weight(fan_in: int, fan_out: int) -> tuple[np.ndarray, np.ndarray]:
    '''
    全连接参数初始化(sqrt(1/fan_in))
    Returns:
        W(np.ndarray(fan_in, fan_out)), b(np.ndarray(1, fan_out))
    '''
    W = np.random.randn(fan_in, fan_out) * np.sqrt(1.0 / fan_in)
    b = np.zeros((1, fan_out), dtype=np.float64)
    return W, b


def linear_forward(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, dict]:
    '''
    全连接前向传播: Z = X @ W + b
    参数：
        X(np.ndarray(N, fan_in))
        W(np.ndarray(fan_in, fan_out))
        b(np.ndarray(1, fan_out))
    '''
    Z = X @ W + b
    cache = {'X': X, 'W': W}
    return Z, cache


def linear_backward(dZ: np.ndarray, cache: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    全连接反向传播
    Returns:
        dX(np.ndarray(N, fan_in)), dW(np.ndarray(fan_in, fan_out)), db(np.ndarray(1, fan_out))
    '''
    X = cache['X']
    W = cache['W']
    dW = X.T @ dZ
    db = np.sum(dZ, axis=0, keepdims=True)
    dX = dZ @ W.T
    return dX, dW, db


# ---------------------------------------------------------------------------
# Softmax + 交叉熵
# ---------------------------------------------------------------------------

def softmax(z: np.ndarray) -> np.ndarray:
    '''
    softmax 函数
    参数：
        z(np.ndarray(N, O)): 线性输出 logits
    Returns:
        probs(np.ndarray(N, O))
    '''
    # 防止指数爆炸
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def cross_entropy_loss(probs: np.ndarray, Y_true: np.ndarray) -> float:
    '''
    交叉熵损失（对 batch 平均）
    参数：
        probs(np.ndarray(N, O)): softmax 概率
        Y_true(np.ndarray(N, O)): one-hot
    '''
    n = probs.shape[0]
    return float(-np.sum(Y_true * np.log(probs + 1e-12)) / n)


def softmax_cross_entropy_grad_from_logits(logits: np.ndarray, Y_true: np.ndarray) -> np.ndarray:
    '''
    联合 Softmax + 交叉熵 对 logits 的梯度: (probs - Y) / N
    '''
    probs = softmax(logits)
    n = logits.shape[0]
    # L 对 z 的偏导就是 probs - Y_true, 所以 L 对 z 的梯度就是 (probs - Y_true) / n
    return (probs - Y_true) / n


# ---------------------------------------------------------------------------
# 按字典更新参数（卷积与全连接可共用）
# ---------------------------------------------------------------------------

def sgd_step(params: dict, grads: dict, lr: float, weight_decay: float = 0.0) -> dict:
    '''
    SGD 更新参数: param -= lr * (grad + weight_decay * param)
    参数：
        params, grads: 同名键一一对应
        weight_decay: L2 正则化系数, 为 0 时退化为普通 SGD
    '''
    for k in grads:
        if weight_decay > 0:
            # 在梯度上加上权重衰减 lambda*W, 相当于在 loss 上增加 1/2*lambda*||W||^2
            grads[k] = grads[k] + weight_decay * params[k]
        params[k] -= lr * grads[k]
    return params


# ---------------------------------------------------------------------------
# 数据读取
# ---------------------------------------------------------------------------

def load_data(
    train_dir: str | None = None,
    val_ratio: float = 0.1
):
    '''
    读取训练目录下手写汉字图像, 展平为 (N, 784), 标签 one-hot (N, 12), 先打乱再划分训练集和验证集
    参数：
        train_dir: 数据根目录, 默认本仓库 train/
        val_ratio: 验证集占比, 为 0 时返回 (X, Y, None, None)
    Returns:
        X_train, Y_train, X_val, Y_val
    '''
    print('Loading data...')
    if train_dir is None:
        train_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'train'))
    else:
        train_dir = os.path.abspath(train_dir)
    class_names = []
    for entry in os.listdir(train_dir):
        full = os.path.join(train_dir, entry)
        if os.path.isdir(full):
            class_names.append(entry)
    class_names.sort(key=int)
    num_classes = len(class_names)
    X_list = []
    Y_list = []
    for i in range(num_classes):
        class_dir = os.path.join(train_dir, class_names[i])
        file_names = sorted(os.listdir(class_dir))
        for name in file_names:
            path = os.path.join(class_dir, name)
            img = Image.open(path)
            pixel = np.asarray(img, dtype=np.float64).reshape(-1)
            label = np.zeros(num_classes, dtype=np.float64)
            label[i] = 1.0
            X_list.append(pixel)
            Y_list.append(label)
    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)
    n = X.shape[0]
    rng = np.random.default_rng(seed=2026)
    perm = rng.permutation(n)
    X, Y = X[perm], Y[perm]
    if val_ratio == 0:
        print('Loading data done.')
        return X, Y, None, None
    n_val = int(round(n * val_ratio))
    n_val = max(1, min(n_val, n - 1))
    X_val, Y_val = X[:n_val], Y[:n_val]
    X_train, Y_train = X[n_val:], Y[n_val:]
    print('Loading data done.')
    return X_train, Y_train, X_val, Y_val


# ---------------------------------------------------------------------------
# CNN: Conv -> ReLU -> MaxPool -> Flatten -> Linear -> logits
# ---------------------------------------------------------------------------

def init_cnn_params(
    num_classes: int = 12,
    c_in: int = 1,
    img_h: int = 28,
    img_w: int = 28,
    c_hidden: int = 8,
    k: int = 3,
    pad: int = 1,
    conv_stride: int = 1,
    pool_size: int = 2,
) -> dict:
    '''
    初始化一层卷积 + 一层全连接的分类网络参数
    默认: 28x28 灰度,卷积保持空间尺寸(pad=1,k=3,s=1), 2x2 非重叠池化后 14x14, 通道 c_hidden
    参数：
        num_classes: 类别数 O
        c_in: 输入通道(灰度 1)
        img_h, img_w: 输入高宽
        c_hidden: 卷积输出通道数
        k: 卷积核边长
        pad: 卷积填充
        conv_stride: 卷积步长
        pool_size: 最大池化核边长
    Returns:
        params: W_conv, b_conv, W_fc, b_fc
    '''
    # 初始化卷积核和偏置
    W_conv = init_conv2d_weight(c_in, c_hidden, k, k)
    b_conv = init_bias_conv(c_hidden)
    # 计算卷积后的高和宽
    h1 = (img_h + 2 * pad - k) // conv_stride + 1
    w1 = (img_w + 2 * pad - k) // conv_stride + 1
    if h1 % pool_size != 0 or w1 % pool_size != 0:
        raise ValueError('池化后尺寸须为整数: 请检查 img/pad/k/stride/pool_size')
    # 计算池化后的高和宽
    h2 = h1 // pool_size
    w2 = w1 // pool_size
    fan_in = c_hidden * h2 * w2
    # 初始化全连接核和偏置
    W_fc, b_fc = init_linear_weight(fan_in, num_classes)
    # 缓存训练参数以及元数据
    cache = {
        'W_conv': W_conv,
        'b_conv': b_conv,
        'W_fc': W_fc,
        'b_fc': b_fc,
        'meta': {
            'pad': pad,
            'conv_stride': conv_stride,
            'pool_size': pool_size,
        },
    }
    return cache


def cnn_forward(X: np.ndarray, params: dict, training: bool = True) -> tuple[np.ndarray, dict]:
    '''
    CNN 前向传播: (N,C,H,W) -> logits(N,O)
    结构: Conv2d -> ReLU -> MaxPool2d -> Flatten -> [Dropout] -> Linear
    参数：
        X(np.ndarray(N, C, H, W)): 输入特征图
        params: init_cnn_params 返回的字典(需含 meta)
        training: 是否为训练模式(影响 Dropout 行为)
    Returns:
        logits(np.ndarray(N, O))
        cache: 各子层 cache, 供 cnn_backward 使用
    '''
    meta = params['meta']
    pad = meta['pad']
    conv_stride = meta['conv_stride']
    pool_size = meta['pool_size']
    drop_prob = meta.get('drop_prob', 0.0)

    out_c, cache_c = conv2d_forward(X, params['W_conv'], params['b_conv'], stride=conv_stride, pad=pad)
    out_r, cache_r = relu_forward(out_c)
    out_p, cache_p = maxpool2d_forward(out_r, pool_size, pool_size)
    out_f, shape_f = flatten_forward(out_p)
    out_d, cache_d = dropout_forward(out_f, drop_prob, training)
    logits, cache_fc = linear_forward(out_d, params['W_fc'], params['b_fc'])

    cache = {
        'conv': cache_c,
        'relu': cache_r,
        'pool': cache_p,
        'flat_shape': shape_f,
        'dropout': cache_d,
        'fc': cache_fc,
    }
    return logits, cache


def cnn_backward(d_logits: np.ndarray, cache: dict) -> dict:
    '''
    CNN 反向传播, 仅负责卷积与全连接部分(不含 softmax)
    参数：
        d_logits(np.ndarray(N, O)): 损失对 logits 的梯度
        cache: cnn_forward 返回的 cache
    Returns:
        grads: W_conv, b_conv, W_fc, b_fc 的梯度
    '''
    grads: dict = {}
    d_flat, grads['W_fc'], grads['b_fc'] = linear_backward(d_logits, cache['fc'])
    d_flat = dropout_backward(d_flat, cache['dropout'])
    d_pool = flatten_backward(d_flat, cache['flat_shape'])
    d_relu = maxpool2d_backward(d_pool, cache['pool'])
    d_conv = relu_backward(d_relu, cache['relu'])
    _, grads['W_conv'], grads['b_conv'] = conv2d_backward(d_conv, cache['conv'])
    return grads


def predict_cnn(X_flat: np.ndarray, params: dict) -> np.ndarray:
    '''
    对展平图像 (N,784) 预测类别 (从 0 到 num_classes-1)
    参数：
        X_flat: 与 load_data 相同格式的输入
        params: 训练得到的参数
    Returns:
        labels(np.ndarray(N,), int)
    '''
    X = images_flat_to_nchw(X_flat)
    logits, _ = cnn_forward(X, params, training=False)
    return np.argmax(logits, axis=1)


def accuracy_from_onehot(labels_pred: np.ndarray, Y_true: np.ndarray) -> float:
    '''
    准确率
    参数：
        labels_pred(np.ndarray(N,)): 预测类别
        Y_true(np.ndarray(N, O)): one-hot 标签
    Returns:
        accuracy(float): 准确率
    '''
    y = np.argmax(Y_true, axis=1)
    return float(np.mean(labels_pred == y))


def save_params(params: dict, path: str = 'cnn_params.npz'):
    '''保存 CNN 模型参数到文件'''
    meta = params['meta']
    save_dict = {}
    for k, v in params.items():
        if isinstance(v, np.ndarray):
            save_dict[k] = v
    for mk, mv in meta.items():
        save_dict[f'_meta_{mk}'] = np.array(mv)
    np.savez(path, **save_dict)
    print(f'参数已保存到 {path}')


def load_params(path: str = 'cnn_params.npz') -> dict:
    '''从文件加载 CNN 模型参数'''
    data = np.load(path)
    params = {}
    meta = {}
    for k in data.files:
        if k.startswith('_meta_'):
            val = data[k].item()
            meta[k[6:]] = int(val) if float(val) == int(val) else float(val)
        else:
            params[k] = data[k]
    params['meta'] = meta
    print(f'参数已从 {path} 加载')
    return params


def interview(eval_dir: str, params: dict) -> float:
    '''
    对指定目录下的测试集进行批量预测并返回准确率
    参数：
        eval_dir: 测试数据根目录(与 train/ 目录结构一致)
        params: 训练好的参数
    Returns:
        accuracy: 准确率 (0~100)
    '''
    X, Y, _, _ = load_data(train_dir=eval_dir, val_ratio=0)
    preds = predict_cnn(X, params)
    true_labels = np.argmax(Y, axis=1)
    accuracy = float(np.mean(preds == true_labels)) * 100
    print(f'测试准确率: {accuracy:.2f}%')
    return accuracy


# 初始化 CNN 参数时接受的键
_CNN_INIT_KEYS = frozenset({
    'num_classes',
    'c_in',
    'img_h',
    'img_w',
    'c_hidden',
    'k',
    'pad',
    'conv_stride',
    'pool_size',
})


def train_cnn(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray | None,
    Y_val: np.ndarray | None,
    lr: float = 0.05,
    epochs: int = 50,
    batch_size: int = 64,
    epoch_print: int = 1,
    weight_decay: float = 0.0,
    drop_prob: float = 0.0,
    patience: int = 0,
    return_history: bool = False,
    **kwargs,
) -> dict | tuple[dict, dict]:
    '''
    按 mini-batch SGD 训练 CNN
    参数：
        X_train(np.ndarray(N,784)), Y_train(N,12): 训练集
        X_val, Y_val: 验证集, 可为 None 则只打印训练 loss
        lr, epochs, batch_size: 学习率、轮数、批大小
        epoch_print: 每隔多少 epoch 打印一次
        weight_decay: L2 正则化系数
        drop_prob: Dropout 丢弃概率
        patience: Early Stopping 耐心值, 为 0 表示不启用
        return_history: 为 True 时额外返回训练历史 {train_loss, val_acc}
        **kwargs: 仅 init_cnn_params 接受的键会传入(如 c_hidden、k等), 其余忽略
    Returns:
        params: 训练后的参数字典
        history (仅 return_history=True): {train_loss: list, val_acc: list}
    '''
    # 初始化
    init_kw = {k: v for k, v in kwargs.items() if k in _CNN_INIT_KEYS}
    init_kw.setdefault('num_classes', int(Y_train.shape[1]))
    params = init_cnn_params(**init_kw)
    params['meta']['drop_prob'] = drop_prob
    n = X_train.shape[0]

    best_val_acc = -1.0
    best_params = None
    wait = 0
    history: dict = {'train_loss': [], 'val_acc': []}

    # 训练
    for epoch in range(epochs):
        # 打乱训练集, 避免学到顺序依赖
        perm = np.random.permutation(n)
        loss_sum = 0.0
        n_batches = 0
        # 遍历训练集, 每个 batch 计算损失和梯度, 更新参数
        for s in range(0, n, batch_size):
            idx = perm[s : s + batch_size]
            Xb = images_flat_to_nchw(X_train[idx])
            Yb = Y_train[idx]
            logits, cache = cnn_forward(Xb, params, training=True)
            probs = softmax(logits)
            loss_sum += cross_entropy_loss(probs, Yb)
            n_batches += 1
            d_logits = softmax_cross_entropy_grad_from_logits(logits, Yb)
            grads = cnn_backward(d_logits, cache)
            sgd_step(params, grads, lr, weight_decay)
        avg_loss = loss_sum / max(n_batches, 1)
        history['train_loss'].append(avg_loss)
        val_acc_epoch = None
        # 打印训练信息
        if (epoch + 1) % epoch_print == 0:
            msg = f'Epoch {epoch + 1}/{epochs}, train loss: {avg_loss:.4f}'
            if X_val is not None and Y_val is not None:
                pred = predict_cnn(X_val, params)
                val_acc_epoch = accuracy_from_onehot(pred, Y_val)
                msg += f', val acc: {val_acc_epoch:.4f}'
                if val_acc_epoch > best_val_acc:
                    best_val_acc = val_acc_epoch
                    # 保存最佳参数
                    best_params = {}
                    for k, v in params.items():
                        if isinstance(v, np.ndarray):
                            # ndarray 深拷贝，得到独立的内存副本, 防止修改原始参数后最佳参数也被修改
                            best_params[k] = v.copy()   
                        else:
                            # 非 ndarray 直接引用, 训练过程中不会被修改
                            best_params[k] = v
                    wait = 0
                else:
                    wait += 1
                # 早停策略防止过拟合
                if patience > 0 and wait >= patience:
                    msg += f'  [early stop, best val acc: {best_val_acc:.4f}]'
                    print(msg)
                    break
            print(msg)
        if val_acc_epoch is not None:
            history['val_acc'].append(val_acc_epoch)

    result = best_params if best_params is not None else params
    if return_history:
        return result, history
    return result


def main():
    set_seed(2026)
    X_train, Y_train, X_val, Y_val = load_data()
    print('train:', X_train.shape, Y_train.shape, 'val:', X_val.shape, Y_val.shape)
    params = train_cnn(
        X_train,
        Y_train,
        X_val,
        Y_val,
        lr=0.01,
        epochs=30,
        batch_size=32,
        epoch_print=1,
        c_hidden=8,
        weight_decay=1e-3,
        drop_prob=0.3,
        patience=5
    )
    save_params(params)


if __name__ == '__main__':
    main()
