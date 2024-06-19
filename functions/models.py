# models.py
# Description: This file contains a collection of Tensorflow models that utilize many customized tropical layers that we use for our experiments.
# Author: Kurt Pasque
# Date: November 29, 2023

'''
Module: models.py

This file contains a collection of Tensorflow models that utilize many customized tropical layers that we use for our experiments.

Functions:
- functional_conv
- pre_model
- post_model
- functional_build_model
- trop_conv3layer_logits
- trop_conv3layer_manyMaxLogits
'''

from custom_layers.tropical_layers import TropEmbedMaxMin, ChangeSignLayer, TropAsymmetricMax, TropAsymmetricMin
from tensorflow import reduce_max, reshape, shape, concat
from tensorflow.keras import Sequential, Model, initializers
from tensorflow.keras.layers import Dense, MaxPooling2D, Flatten, Conv2D, Dropout, GlobalAveragePooling2D, Layer, AveragePooling2D, DepthwiseConv2D, ReLU, BatchNormalization
from tensorflow.keras.applications import ResNet50, VGG16, MobileNet, EfficientNetB0, VGG19
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import serialize_keras_object, deserialize_keras_object


class Maxout(Layer):
    def __init__(self, num_units, axis=-1, **kwargs):
        super(Maxout, self).__init__(**kwargs)
        self.num_units = num_units
        self.axis = axis

    def call(self, inputs):
        # Use tf.shape to get the runtime shape
        input_shape = shape(inputs)
        num_channels = input_shape[self.axis]

        # Adjust the shape for the maxout operation
        new_shape = concat([
            input_shape[:self.axis],
            [self.num_units],
            [num_channels // self.num_units]
        ], axis=0)

        # Reshape and perform max operation
        step1 = reshape(inputs, new_shape)
        return reduce_max(step1, axis=-2)


class CustomModelClass(Model):
    def __init__(self, 
                 num_classes, 
                 top, 
                 initializer=initializers.RandomNormal(mean=0., stddev=2., seed=0), 
                 num_maxout_neurons = 64, 
                 dropout_rate = 0.5,
                 lam = 0,
                 **kwargs):
        super(CustomModelClass, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.initializer = initializer
        self.dropout_rate = dropout_rate
        self.num_maxout_neurons = num_maxout_neurons
        self.lam = lam
        self.top = top
        self._select_top_layer(self.top)  # Initialize layers based on top type

    def _select_top_layer(self, top):
        if top == "relu":
            self._build_relu()
            self.top_processor = self.simple_top
        elif top == "trop":
            self._build_trop()
            self.top_processor = self.simple_top
        elif top == "maxout":
            self._build_maxout()
            self.top_processor = self.maxout_top
        elif top == "tropAsymMax":
            self._build_trop_asym_max()
            self.top_processor = self.simple_top
        elif top == "tropAsymMin":
            self._build_trop_asym_min()
            self.top_processor = self.simple_top
        else:
            raise ValueError("Invalid top layer specified")

    def _build_relu(self):
        self.top_layer = Sequential([
            Dense(256, activation="relu", name="last_fc"),
            Dense(self.num_classes)
        ])

    def _build_trop(self):
        self.top_layer = Sequential([
            Dense(3, activation="relu", name="last_fc"),
            TropEmbedMaxMin(self.num_classes, initializer_w=self.initializer, lam=self.lam, name="tropical"),
            ChangeSignLayer(),
        ])
    
    def _build_trop_asym_max(self):
        self.top_layer = Sequential([
            #Dense(64, activation="relu", name="last_fc"),
            TropAsymmetricMax(self.num_classes, initializer_w=self.initializer, lam=self.lam, name="tropical"),
            ChangeSignLayer(),
        ])

    def _build_trop_asym_min(self):
        self.top_layer = Sequential([
            #Dense(64, activation="relu", name="last_fc"),
            TropAsymmetricMin(self.num_classes, initializer_w=self.initializer, lam=self.lam, name="tropical"),
            ChangeSignLayer(),
        ])

    def _build_maxout(self):
        self.top_layer = None
        self.dense_0 = Dense(256, activation="relu", name="last_fc")
        self.dense_1 = Dense(self.num_maxout_neurons * self.num_classes, kernel_initializer=self.initializer)
        self.dense_2 = Dense(self.num_maxout_neurons * self.num_classes, kernel_initializer=self.initializer)
        self.dropout_1 = Dropout(self.dropout_rate)
        self.dropout_2 = Dropout(self.dropout_rate)
        self.maxout_1 = Maxout(num_units=self.num_classes, axis=-1)
        self.maxout_2 = Maxout(num_units=self.num_classes, axis=-1)

    def simple_top(self, x, training):
        return self.top_layer(x)

    def maxout_top(self, x, training):
        x = self.dense_0(x)

        x_1 = self.dense_1(x)
        x_1 = self.dropout_1(x_1, training=training)
        x_1 = self.maxout_1(x_1)

        x_2 = self.dense_2(x)
        x_2 = self.dropout_2(x_2, training=training)
        x_2 = self.maxout_2(x_2)
        return x_1 - x_2

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'top': self.top,
            'initializer': initializers.serialize(self.initializer),
            'num_maxout_neurons': self.num_maxout_neurons,
            'dropout_rate': self.dropout_rate,
            'lam': self.lam
        })
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)

class AlexNetModel(CustomModelClass):
    def __init__(self, 
                 num_classes, 
                 top, 
                 initializer=initializers.HeNormal(), 
                 num_maxout_neurons=100, 
                 dropout_rate=0.5, 
                 input_shape=(32, 32, 3), 
                 **kwargs):
        super(AlexNetModel, self).__init__(num_classes=num_classes,
                                           top=top,
                                           initializer=initializer,
                                           num_maxout_neurons=num_maxout_neurons,
                                           dropout_rate=dropout_rate,
                                           **kwargs)
        self.input_shape = input_shape
        self._build_base()

    def _build_base(self):
        self.base_layers = Sequential([
            Conv2D(96, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=self.input_shape, padding='same', kernel_initializer=self.initializer),
            MaxPooling2D(pool_size=(3, 3), strides=(1, 1)),
            Conv2D(256, kernel_size=(5, 5), activation='relu', padding='same', kernel_initializer=self.initializer),
            MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            Conv2D(384, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer=self.initializer),
            Conv2D(384, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer=self.initializer),
            Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer=self.initializer),
            MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            Flatten(),
            Dense(4096, activation='relu', kernel_initializer=self.initializer),
            Dropout(0.5),
            Dense(4096, activation='relu', kernel_initializer=self.initializer),
            Dropout(0.5),
        ])

    def call(self, inputs, training=True):
        x = self.base_layers(inputs)
        return self.top_processor(x, training)

    def get_config(self):
        config = super().get_config()
        config.update({
            'input_shape': self.input_shape
        })
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)



class VGG16Model(CustomModelClass):
    def __init__(self, 
                 num_classes, 
                 top,
                 initializer=initializers.RandomNormal(mean=0, stddev=0.1, seed=0), 
                 num_maxout_neurons = 100, 
                 dropout_rate = 0.5,
                 input_shape = (32, 32, 3),
                 **kwargs,
                 ):
        super(VGG16Model, self).__init__(num_classes = num_classes, 
                                    top = top, 
                                    initializer=initializer, 
                                    num_maxout_neurons = num_maxout_neurons, 
                                    dropout_rate = dropout_rate,
                                    **kwargs,
                                    )
        self.input_shape = input_shape
        self._build_base()

    def _build_base(self):
        self.base_layers = Sequential([
            VGG16(weights=None, include_top=False, input_shape=self.input_shape),
            Flatten(),
            #BatchNormalization(),
            #Dropout(0.5),
            Dense(512, activation="relu", name="fc1"),#, kernel_initializer=self.initializer),
            Dropout(0.5),
            #Dense(256, activation="relu", name="fc2"),#, kernel_initializer=self.initializer),
            #Dropout(0.4),
        ])

    def call(self, inputs, training=True):
        x = self.base_layers(inputs)
        return self.top_processor(x, training)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'input_shape': self.input_shape
        })
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)

class ModifiedLeNet5(CustomModelClass):
    def __init__(self, 
                 num_classes, 
                 top,
                 initializer=initializers.RandomNormal(mean=0, stddev=1., seed=0), 
                 num_maxout_neurons = 100, 
                 dropout_rate = 0.5,
                 **kwargs):
        super(ModifiedLeNet5, self).__init__(num_classes = num_classes, 
                                    top = top, 
                                    initializer=initializer, 
                                    num_maxout_neurons = num_maxout_neurons, 
                                    dropout_rate = dropout_rate,
                                    **kwargs)
        self._build_base()

    def _build_base(self):
        self.base_layers = Sequential([
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            Flatten(),
            Dense(64, activation='relu'),
        ])

    def call(self, inputs, training=True):
        x = self.base_layers(inputs)
        return self.top_processor(x, training)
    
    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)


class LeNet5(CustomModelClass):
    def __init__(self, 
                 num_classes, 
                 top,
                 initializer=initializers.RandomNormal(mean=0, stddev=1., seed=0), 
                 num_maxout_neurons = 100, 
                 dropout_rate = 0.5,
                 **kwargs):
        super(LeNet5, self).__init__(num_classes = num_classes, 
                                    top = top, 
                                    initializer=initializer, 
                                    num_maxout_neurons = num_maxout_neurons, 
                                    dropout_rate = dropout_rate,
                                    **kwargs)
        self._build_base()

    def _build_base(self):
        self.base_layers = Sequential([
            Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='same'),
            AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
            Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'),
            AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
            Conv2D(filters=120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'),
            Flatten(),
            Dense(units=84, activation='tanh'),

        ])

    def call(self, inputs, training=True):
        x = self.base_layers(inputs)
        return self.top_processor(x, training)
    
    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)


class MobileNetModel(CustomModelClass):
    def __init__(self, 
                 num_classes, 
                 top,
                 initializer=initializers.RandomNormal(mean=0, stddev=1., seed=0), 
                 num_maxout_neurons = 100, 
                 dropout_rate = 0.5,
                 input_shape = (32, 32, 3),
                 **kwargs
                 ):
        super(MobileNetModel, self).__init__(num_classes = num_classes, 
                                    top = top, 
                                    initializer=initializer, 
                                    num_maxout_neurons = num_maxout_neurons, 
                                    dropout_rate = dropout_rate,
                                    **kwargs)
        self.input_shape = input_shape
        self._build_base()

    def _build_base(self):
        self.base_layers = Sequential([
            MobileNet(weights=None, include_top=False, input_shape=self.input_shape),
            GlobalAveragePooling2D(),
        ])

    def call(self, inputs, training=True):
        x = self.base_layers(inputs)
        return self.top_processor(x, training)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'input_shape': self.input_shape
        })
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)


class EfficientNetB4Model(CustomModelClass):
    def __init__(self, 
                 num_classes, 
                 top,
                 initializer=initializers.RandomNormal(mean=0, stddev=0.001, seed=0), 
                 num_maxout_neurons = 100, 
                 dropout_rate = 0.5,
                 input_shape = (32, 32, 3),
                 **kwargs,
                 ):
        super(EfficientNetB4Model, self).__init__(num_classes = num_classes, 
                                    top = top, 
                                    initializer=initializer, 
                                    num_maxout_neurons = num_maxout_neurons, 
                                    dropout_rate = dropout_rate,
                                    **kwargs,
                                    )
        self.input_shape = input_shape
        self._build_base()

    def _build_base(self):
        self.base_layers = Sequential([
            EfficientNetB0(weights=None, include_top=False, input_shape=self.input_shape),
            GlobalAveragePooling2D(),
        ])

    def call(self, inputs, training=True):
        x = self.base_layers(inputs)
        return self.top_processor(x, training)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'input_shape': self.input_shape
        })
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)


class ResNet50Model(CustomModelClass):
    def __init__(self, 
                 num_classes, 
                 top,
                 initializer=initializers.RandomNormal(mean=0, stddev=0.01, seed=0),
                 num_maxout_neurons=100,  
                 dropout_rate = 0.5,
                 input_shape = (32, 32, 3),
                 **kwargs,
                 ):
        super(ResNet50Model, self).__init__(num_classes = num_classes, 
                                    top = top, 
                                    initializer=initializer, 
                                    num_maxout_neurons = num_maxout_neurons, 
                                    dropout_rate = dropout_rate,
                                    **kwargs,
                                    )
        self.input_shape = input_shape
        self._build_base()

    def _build_base(self):
        self.base_layers = Sequential([
            ResNet50(weights=None, include_top=False, input_shape=self.input_shape),
        GlobalAveragePooling2D(),
        ])

    def call(self, inputs, training=True):
        x = self.base_layers(inputs)
        return self.top_processor(x, training)   
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'input_shape': self.input_shape
        })
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)


class MMRModel(Model):
    def __init__(self, num_classes, initializer=initializers.RandomNormal(mean=0.5, stddev=1., seed=0),
                 **kwargs):
        super(MMRModel, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.initializer = initializer
        self._build_model()

    def _build_model(self):
        self.conv_layer1 = Conv2D(64, (3,3), activation='relu')
        self.max_layer1 = MaxPooling2D((2, 2))
        self.conv_layer2 = Conv2D(64, (3, 3), activation='relu')
        self.max_layer2 = MaxPooling2D((2, 2))
        self.conv_layer3 = Conv2D(64, (3, 3), activation='relu')
        self.flatten = Flatten()  

        self.dense_layer1 = Dense(64, activation='relu')
        self.dense_layer2 = Dense(64, activation='relu')
        self.final_layer = Dense(self.num_classes, kernel_initializer=self.initializer)

    def call(self, inputs, training=True, return_feature_maps=False):
        feature_maps = []

        x = self.conv_layer1(inputs)
        if return_feature_maps:
            feature_maps.append(x)

        x = self.max_layer1(x)

        x = self.conv_layer2(x)
        if return_feature_maps:
            feature_maps.append(x)

        x = self.max_layer2(x)

        x = self.conv_layer3(x)
        if return_feature_maps:
            feature_maps.append(x)

        x = self.flatten(x)

        x = self.dense_layer1(x)
        if return_feature_maps:
            feature_maps.append(x)

        x = self.dense_layer2(x)
        if return_feature_maps:
            feature_maps.append(x)

        logits = self.final_layer(x)
        if return_feature_maps:
            feature_maps.append(logits)
            return logits, feature_maps
        return logits
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'initializer': initializers.serialize(self.initializer),
        })
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)
