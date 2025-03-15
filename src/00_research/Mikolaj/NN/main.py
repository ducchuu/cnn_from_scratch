from dense import Dense
from tanh import Tanh
from losses import mean_squared_error, mean_squared_error_prime
import numpy as np

X = np.reshape([[0,0], [0,1], [1,0], [1,1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

network = [Dense(2,3), Tanh(), Dense(3,1), Tanh()]

epochs = 10000
learning_rate = 0.1

for e in range(epochs):
    error = 0
    for x, y in zip(X, Y):
        output = X
        for layer in network:
            output = layer.forward(output)

        error += mean_squared_error(y, output)

        grad = mean_squared_error_prime(y, output)
        for layer in reversed(network):
            grad = layer.backwad(grad, learning_rate)

    error /= len(x)
    print("%d/%d, error=%f" % (e + 1, epochs, error))