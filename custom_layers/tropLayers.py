from tensorflow import reshape, constant, expand_dims, reduce_max ,reduce_min
from tensorflow.math import top_k, reduce_sum
from tensorflow.keras.layers import Layer
from tensorflow.keras.backend import repeat_elements
from tensorflow.keras import initializers, regularizers
from tensorflow.image import extract_patches
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class TropReg(regularizers.Regularizer):
    def __init__(self, lam=1.0):
        self.lam = lam

    def __call__(self, weight_matrix):
        values, indices = top_k(weight_matrix, 1)
        values2, indices2 = top_k(-weight_matrix, 1)
        return self.lam * reduce_sum(values[:, 0] + values2[:, 0])

    def get_config(self):
        return {'lam': float(self.lam)}


class TropEmbedTop2(Layer):
    def __init__(self , units=2, input_dim=3):
        super(TropEmbedTop2, self).__init__()
        self.w = self.add_weight(shape=(units, input_dim), initializer="random_normal", regularizer=TropReg(lam=0.01), trainable=True)
        self.units = units
        self.input_dim = input_dim

    def call(self, inputs):
        input_reshaped = reshape(inputs ,[-1, 1, self.input_dim])
        input_for_broadcast = repeat_elements(input_reshaped, self.units, 1)
        values, indices = top_k(input_for_broadcast + self.w, 2)
        return values[:,:,0] - values [:,:,1] # symmetric tropical distance
    
    
class TropEmbedMaxMin(Layer):
    def __init__(self, units = 2, input_dim = 3, initializer_w = initializers.random_normal, lam = 0.01):
        super(TropEmbedMaxMin, self).__init__()
        self.w = self.add_weight(shape=(units, input_dim), 
                                 initializer=initializer_w,
                                 regularizer=TropReg(lam=lam),
                                 trainable=True)
        self.units = units
        self.input_dim = input_dim

    def call(self, x):
        x_reshaped = reshape(x,[-1, 1, self.input_dim])
        x_for_broadcast = repeat_elements(x_reshaped, self.units, 1)
        valMax, indices = top_k(x_for_broadcast + self.w, 1)
        valMin, indices = top_k(-(x_for_broadcast + self.w), 1)
        return valMax[:,:,0] + valMin[:,:,0]
    
    
    def get_config(self):
        config = {
            'units': self.units,
            'input_dim': self.input_dim,
            'initializer_w': initializers.serialize(self.initializer_w)
        }
        base_config = super(TropEmbedMaxMin, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    

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
        np.random.shuffle(weights)  # Shuffle the order
        
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
        
        np.random.shuffle(weights)  # Shuffle the order
        
        return constant(weights.reshape(shape), dtype=dtype)


class TropConv2D(Layer):
    def __init__(self, filters = 64, 
                 window_size = [1, 3, 3, 1], 
                 strides = [1, 1, 1, 1],
                 rates = [1, 1, 1, 1],
                 padding = 'VALID',
                 channels = 3,
                 initializer_w = initializers.random_normal, 
                 lam = 0.01):
        super(TropConv2D, self).__init__()
        self.w = self.add_weight(shape=(1, 1, 1, window_size[1]*window_size[2]*channels, filters), 
                                 initializer=initializer_w,
                                 regularizer=TropReg(lam=lam),
                                 trainable=True)
        self.filters = filters
        self.window_size = window_size
        self.strides = strides
        self.rates = rates
        self.padding = padding
        self.channels = channels

    def call(self, x):
        x_patches = extract_patches(images=x, sizes=self.window_size, strides=self.strides, rates=self.rates, padding=self.padding)
        result_addition = expand_dims(x_patches, axis=-1) + self.w
        return reduce_max(result_addition, axis=(3)) - reduce_min(result_addition, axis=(3))
    
    def get_config(self):
        config = {
            'filters': self.filters,
            'window_size': self.window_size,
            'initializer_w': initializers.serialize(self.initializer_w)
        }
        base_config = super(TropConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))