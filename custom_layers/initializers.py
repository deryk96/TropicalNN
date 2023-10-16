from tensorflow import constant
from tensorflow.keras import initializers
import numpy as np


class BimodalBinaryInitializer(initializers.Initializer):
    def __init__(self, high=2, low=-2, seed=None):
        self.seed = seed
        self.high = high
        self.low = low

    def __call__(self, shape, dtype=None):
        if self.seed is not None:
            np.random.seed(self.seed)
        weights = np.random.choice([-self.low, self.high], size=shape, replace=True)
        return constant(weights, dtype=dtype)


class BimodalNormalInitializer(initializers.Initializer):
    def __init__(self, stddev=1,high=2, low=-2,seed=None):
        self.seed = seed
        self.stddev = stddev
        self.high = high
        self.low = low

    def __call__(self, shape, dtype=None):
        if self.seed is not None:
            np.random.seed(self.seed)
        num_values = np.prod(shape)
        num_values_half = num_values // 2
        values1 = np.random.normal(loc=self.low, scale=self.stddev, size=num_values_half)
        values2 = np.random.normal(loc=self.high, scale=self.stddev, size=num_values - num_values_half)
        weights = np.concatenate((values1, values2))
        np.random.shuffle(weights)
        return constant(weights.reshape(shape), dtype=dtype)
    

class MultimodalNormalInitializer(initializers.Initializer):
    def __init__(self, stddev=1,modes = [-4.5, 5.5],seed=None):
        self.seed = seed
        self.stddev = stddev
        self.modes = modes

    def __call__(self, shape, dtype=None):
        if self.seed is not None:
            np.random.seed(self.seed)
        num_vals = np.prod(shape)
        num_modes = len(self.modes)
        num_vals_section = num_vals // num_modes
        num_vals_remainder = num_vals % num_modes
        values = [np.random.normal(loc=m, scale=self.stddev, size=num_vals_section) for m in self.modes]
        if num_vals_remainder != 0:
            values.append(np.random.normal(loc=np.random.choice(self.modes), scale=self.stddev, size=num_vals_remainder))
        weights = np.concatenate(values)
        np.random.shuffle(weights)
        return constant(weights.reshape(shape), dtype=dtype)
    

class Triangular(initializers.Initializer):
    def __init__(self, left=0,mode=0.5, right=1,seed=None):
        self.seed = seed
        self.left = left
        self.mode = mode
        self.right = right

    def __call__(self, shape, dtype=None):
        if self.seed is not None:
            np.random.seed(self.seed)
        num_values = np.prod(shape)
        weights = np.random.triangular(left=self.left, mode=self.mode, right=self.right, size=num_values)
        np.random.shuffle(weights)
        return constant(weights.reshape(shape), dtype=dtype)