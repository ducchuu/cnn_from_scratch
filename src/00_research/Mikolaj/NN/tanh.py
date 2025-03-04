import numpy as np
from activation import Activation

class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tahn(x)
        tanh_prime = lambda x: 1 - np.tanh(x) ** 2
        super().__init__(tanh, tanh_prime)