import pickle
import numpy as np
import matplotlib.pylab as plt
from tqdm import tqdm


def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """MNISTデータセットの読み込み




    Parameters
    ----------
    normalize : 画像のピクセル値を0.0~1.0に正規化する
    one_hot_label :
        one_hot_labelがTrueの場合、ラベルはone-hot配列として返す
        one-hot配列とは、たとえば[0,0,1,0,0,0,0,0,0,0]のような配列
    flatten : 画像を一次元配列に平にするかどうか

    Returns
    -------
    (訓練画像, 訓練ラベル), (テスト画像, テストラベル)
    """

    with open('mnist_dataset/mnist.pkl', 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label:
        def _change_one_hot_label(X):
            T = np.zeros((X.size, 10))
            for idx, row in enumerate(T):
                row[X[idx]] = 1
            return T

        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])

    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), \
           (dataset['test_img'], dataset['test_label'])


_h = 1e-4


# noinspection PyTypeChecker
def step_function(x):
    return np.array(0 < x, dtype=np.int)


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


def identity_function(x):
    return x


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))


def numerical_diff(f, x):
    return (f(x + _h) - f(x - _h)) / (2 * _h)


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t)**2)


def cross_entropy_error(y, t):
    """
    :type y: numpy.ndarray
    :type t: numpy.ndarray
    :rtype: float
    """
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    bach_size = y.shape[0]
    return -np.sum(t * np.log(y)) / bach_size


def _numerical_gradient_no_batch(f, x):
    grad = np.zeros_like(x)

    for index in range(x.size):
        tmp_x = x[index]
        x[index] = float(tmp_x) + _h
        fxh1 = f(x)

        x[index] = float(tmp_x) - _h
        fxh2 = f(x)

        grad[index] = (fxh1 - fxh2) / (2 * _h)
        x[index] = tmp_x

    return grad


def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)
        for idx, x in tqdm(enumerate(X)):
            grad[idx] = _numerical_gradient_no_batch(f, x)

        return grad


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        x -= lr * numerical_gradient(f, x)
    return x


def main():
    x = np.arange(-2, 2, 0.05)
    y1 = step_function(x)
    y2 = relu(x)
    y3 = sigmoid(x)
    plt.plot(x, y1, label='step function')
    plt.plot(x, y2, label='relu', linestyle='--')
    plt.plot(x, y3, label='sigmoid')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('activation functions')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
