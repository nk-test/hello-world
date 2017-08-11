import numpy as np
import common_funcs as funcs


class AddLayer:
    @staticmethod
    def forward(x, y):
        return x + y

    @staticmethod
    def backward(delta):
        return delta, delta


class MultiLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        return self.x * self.y

    def backward(self, delta):
        dx = delta * self.y
        dy = delta * self.x
        return dx, dy


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, delta):
        delta[self.mask] = 0
        return delta


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = funcs.sigmoid(x)
        return self.out

    def backward(self, delta):
        return delta * self.out * (1 - self.out)


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, delta):
        dx = np.dot(delta, self.W.T)
        self.dW = np.dot(self.x.T, delta)
        self.db = np.sum(delta, axis=0)
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = funcs.softmax(x)
        self.loss = funcs.cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, delta=1):
        # TODO: clarify why the result needs to be divided by the batch size
        batch_size = self.t.shape[0]
        return (self.y - self.t) / batch_size


class BatchNormalization:
    """
    http://arxiv.org/abs/1502.03167
    TODO: Understand implementation of this class
    """

    def __init__(self, gamma=1, beta=0, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None  # Conv層の場合は4次元、全結合層の場合は2次元

        # テスト時に使用する平均と分散
        self.running_mean = running_mean
        self.running_var = running_var

        # backward時に使用する中間データ
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)

        return out.reshape(*self.input_shape)

    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc ** 2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))

        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx


class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)


    def backward(self, delta):
        return delta * self.mask


def main():
    s_layer = SoftmaxWithLoss()
    x = np.array([10, 1, 3])
    t1 = np.array([1, 0, 0])
    t2 = np.array([0, 1, 0])
    s_layer.forward(x, t1)
    print('y', s_layer.y)
    print('loss', s_layer.loss)
    print('back', s_layer.backward())
    s_layer.forward(x, t2)
    print('y', s_layer.y)
    print('loss', s_layer.loss)
    print('back', s_layer.backward())


if __name__ == '__main__':
    main()
