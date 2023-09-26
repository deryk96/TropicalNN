from tensorflow import reshape
from tensorflow.math import top_k
from tensorflow.keras.layers import Layer
from tensorflow.keras.backend import repeat_elements
from tensorflow.keras import initializers


class TropEmbedTop2(Layer):
    def __init__(self , units=2, input_dim=3):
        super(TropEmbedTop2, self).__init__()
        self.w = self.add_weight(shape=(units, input_dim), initializer="random_normal", trainable=True)
        self.units = units
        self.input_dim = input_dim

    def call(self, inputs):
        input_reshaped = reshape(inputs ,[-1, 1, self.input_dim])
        input_for_broadcast = repeat_elements(input_reshaped, self.units, 1)
        values, indices = top_k(input_for_broadcast + self.w, 2)
        return values[:,:,0] - values [:,:,1] # symmetric tropical distance
    
    
class TropEmbedMaxMin(Layer):
    def __init__(self, units = 2, input_dim = 3, initializer_w = initializers.random_normal):
        super(TropEmbedMaxMin, self).__init__()
        self.w = self.add_weight(shape=(units, input_dim), 
                                 initializer=initializer_w, 
                                 trainable=True)
        self.units = units
        self.input_dim = input_dim

    def call(self, x):
        x_reshaped = reshape(x,[-1, 1, self.input_dim])
        x_for_broadcast = repeat_elements(x_reshaped, self.units, 1)
        valMax, indices = top_k(x_for_broadcast + self.w, 1)
        valMin, indices = top_k(-(x_for_broadcast + self.w), 1)
        return valMax[:,:,0] - valMin[:,:,0] # symmetric tropical distance