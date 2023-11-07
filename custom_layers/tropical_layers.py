from tensorflow import reshape, expand_dims, reduce_max ,reduce_min, float32
from tensorflow.math import top_k, reduce_sum, exp
from tensorflow.keras.layers import Layer
from tensorflow.keras.backend import repeat_elements
from tensorflow.keras import initializers, regularizers
from tensorflow.image import extract_patches


class TropReg(regularizers.Regularizer):
    def __init__(self, lam=1.0):
        self.lam = lam

    def __call__(self, weight_matrix):
        values, indices = top_k(weight_matrix, 1)
        values2, indices2 = top_k(-weight_matrix, 1)
        return self.lam * reduce_sum(values[:, 0] + values2[:, 0])


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
    def __init__(self, units=2, initializer_w=initializers.random_normal, lam=0.0, **kwargs):
        super(TropEmbedMaxMin, self).__init__(**kwargs)
        self.units = units
        self.initializer_w = initializer_w
        self.lam = lam

    def build(self, input_shape):
        input_dim = input_shape[-1]  # Extract the last dimension from input_shape
        self.w = self.add_weight(name='tropical_fw', 
                                 shape=(self.units, input_dim),
                                 initializer=self.initializer_w,
                                 regularizer=TropReg(lam=self.lam),
                                 trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=(self.units,),
                                    initializer="zeros", 
                                    trainable=True)
        super(TropEmbedMaxMin, self).build(input_shape)

    def call(self, x):
        x_reshaped = reshape(x,[-1, 1, self.w.shape[-1]])
        x_for_broadcast = repeat_elements(x_reshaped, self.units, 1)
        result_addition = x_for_broadcast + self.w
        return reduce_max(result_addition, axis=(1)) - reduce_min(result_addition, axis=(1)) + self.bias
    
    
    def get_config(self):
        config = {
            'units': self.units,
            'initializer_w': initializers.serialize(self.initializer_w),
            'lam': self.lam
        }
        base_config = super(TropEmbedMaxMin, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)  
    

class TropConv2D(Layer):
    def __init__(self, filters = 64, 
                 window_size = [1, 3, 3, 1], 
                 strides = [1, 1, 1, 1],
                 rates = [1, 1, 1, 1],
                 padding = 'VALID',
                 initializer_w = initializers.random_normal, 
                 lam = 0.0,
                 **kwargs):
        super(TropConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.initializer_w = initializer_w
        self.window_size = window_size
        self.strides = strides
        self.rates = rates
        self.padding = padding
        self.lam = lam

    def build(self, input_shape):
        channels = input_shape[-1]  # Extract the last dimension from input_shape
        self.w = self.add_weight(shape=(1, 1, 1, self.window_size[1] * self.window_size[2] * channels, self.filters), 
                                 initializer=self.initializer_w,
                                 regularizer=TropReg(lam=self.lam),
                                 trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=(self.filters,),
                                    initializer="zeros", 
                                    trainable=True)
        super(TropConv2D, self).build(input_shape)

    def call(self, x):
        x_patches = extract_patches(images=x, sizes=self.window_size, strides=self.strides, rates=self.rates, padding=self.padding)
        result_addition = expand_dims(x_patches, axis=-1) + self.w
        return reduce_max(result_addition, axis=(3)) - reduce_min(result_addition, axis=(3)) + self.bias
    
    def get_config(self):
        config = {
            'filters': self.filters,
            'window_size': self.window_size,
            'strides': self.strides,
            'rates': self.rates,
            'padding': self.padding,
            'initializer_w': initializers.serialize(self.initializer_w),
            'lam': self.lam
        }
        base_config = super(TropConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)  


class TropConv2DMax(Layer):
    def __init__(self, filters = 64, 
                 window_size = [1, 3, 3, 1], 
                 strides = [1, 1, 1, 1],
                 rates = [1, 1, 1, 1],
                 padding = 'VALID',
                 initializer_w = initializers.random_normal, 
                 lam = 0.0):
        super(TropConv2DMax, self).__init__()
        self.filters = filters
        self.initializer_w = initializer_w
        self.window_size = window_size
        self.strides = strides
        self.rates = rates
        self.padding = padding
        self.lam = lam

    def build(self, input_shape):
        channels = input_shape[-1]  # Extract the last dimension from input_shape
        self.w = self.add_weight(shape=(1, 1, 1, self.window_size[1] * self.window_size[2] * channels, self.filters), 
                                 initializer=self.initializer_w,
                                 regularizer=TropReg(lam=self.lam),
                                 trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=(self.filters,),
                                    initializer="zeros", 
                                    trainable=True)
        super(TropConv2DMax, self).build(input_shape)

    def call(self, x):
        x_patches = extract_patches(images=x, sizes=self.window_size, strides=self.strides, rates=self.rates, padding=self.padding)
        result_addition = expand_dims(x_patches, axis=-1) + self.w
        return reduce_max(result_addition, axis=(3)) + self.bias
    
    def get_config(self):
        config = {
            'filters': self.filters,
            'window_size': self.window_size,
            'strides': self.strides,
            'rates': self.rates,
            'padding': self.padding,
            'initializer_w': initializers.serialize(self.initializer_w),
            'lam': self.lam
        }
        base_config = super(TropConv2DMax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)  
    

class TropEmbedMaxMinLogits(Layer):
    def __init__(self, units=2, initializer_w=initializers.random_normal, lam=0.0, **kwargs):
        super(TropEmbedMaxMinLogits, self).__init__(**kwargs)
        self.units = units
        self.initializer_w = initializer_w
        self.lam = lam

    def build(self, input_shape):
        input_dim = input_shape[-1]  # Extract the last dimension from input_shape
        self.w = self.add_weight(name='tropical_fw', 
                                 shape=(self.units, input_dim),
                                 initializer=self.initializer_w,
                                 regularizer=TropReg(lam=self.lam),
                                 trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=(self.units,),
                                    initializer="zeros", 
                                    trainable=True)
        super(TropEmbedMaxMinLogits, self).build(input_shape)

    def call(self, x):
        x_reshaped = reshape(x,[-1, 1, self.w.shape[-1]])
        x_for_broadcast = repeat_elements(x_reshaped, self.units, 1)
        result_addition = x_for_broadcast + self.w
        trop_distance = reduce_max(result_addition, axis=(2)) - reduce_min(result_addition, axis=(2)) + self.bias
        negative_exponents = exp(-trop_distance)
        softmin_values = negative_exponents / reduce_sum(negative_exponents, axis=-1, keepdims=True)
        return softmin_values 
    
    
    def get_config(self):
        config = {
            'units': self.units,
            'initializer_w': initializers.serialize(self.initializer_w),
            'lam': self.lam
        }
        base_config = super(TropEmbedMaxMinLogits, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    @classmethod
    def from_config(cls, config):
        return cls(**config) 