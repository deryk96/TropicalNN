'''
File name: ./functions/models.py
Author: Kurt Pasque
Created: 2023-11-29
Description:
    This file contains a collection of PyTorch models that utilize many customized
    tropical layers used in our experiments.
'''

# Local file imports
from custom_layers.tropical_layers import TropEmbed, ChangeSignLayer

# TODO: Delete Tensorflow imports
# from tensorflow import reduce_max, reshape, shape, concat
# from tensorflow.keras import Sequential, Model, initializers
# from tensorflow.keras.layers import Dense, MaxPooling2D, Flatten, Conv2D, Dropout, GlobalAveragePooling2D, Layer, AveragePooling2D
# from tensorflow.keras.applications import ResNet50, VGG16, MobileNet, EfficientNetB0
# from tensorflow.keras.models import Sequential, Model

# PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.models as models

class Maxout(nn.Module):
    def __init__(self, num_units, axis=-1, **kwargs):
        super(Maxout, self).__init__(**kwargs)
        self.num_units = num_units
        self.axis = axis

    # def call(self, inputs):
    #     # Use tf.shape to get the runtime shape
    #     input_shape = shape(inputs)
    #     num_channels = input_shape[self.axis]
    #
    #     # Adjust the shape for the maxout operation
    #     new_shape = concat([
    #         input_shape[:self.axis],
    #         [self.num_units],
    #         [num_channels // self.num_units]
    #     ], axis=0)
    #
    #     # Reshape and perform max operation
    #     step1 = reshape(inputs, new_shape)
    #     return reduce_max(step1, axis=-2)

    def forward(self, inputs):
        shape = list(inputs.size())
        num_channels = shape[self.axis]

        # Assert that the number of channels is divisible by num_units
        assert num_channels % self.num_units == 0

        new_shape = shape[:self.axis] + [self.num_units, num_channels // self.num_units]
        inputs = inputs.view(*new_shape)
        return torch.max(inputs, dim=-2)[0]

class CustomModelClass(nn.Module):
    def __init__(self, 
                 num_classes, 
                 top, 
                 # initializer=initializers.RandomNormal(mean=0., stddev=2., seed=0),
                 initializer=None,
                 num_maxout_neurons=64,
                 dropout_rate=0.5,
                 lam=0,
                 **kwargs):
        super(CustomModelClass, self).__init__(**kwargs)
        self.num_classes = num_classes
        # self.initializer = initializer
        self.dropout_rate = dropout_rate
        self.num_maxout_neurons = num_maxout_neurons
        self.lam = lam
        self.top = top
        self.top_layer = None  # Added to explicitly initialize variable
        self._select_top_layer(top)  # Initialize layers based on top type

        # Set initializer
        if initializer is None:
            self.initializer = lambda w: nn.init.normal_(w, mean=0., std=2.0)
        else:
            self.initializer = initializer

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
            raise ValueError("Invalid top layer specified.")

    def _build_relu(self):
        # self.top_layer = Sequential([
        #     Dense(256, activation="relu", name="last_fc"),
        #     Dense(self.num_classes)
        # ])

        self.top_layer = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_classes)
        )

    def _build_trop(self):
        # self.top_layer = Sequential([
        #     Dense(64, activation="relu", name="last_fc"),
        #     TropEmbed(self.num_classes, initializer_w=self.initializer, lam=self.lam, distance_metric = "sym", name="tropical"),
        #     ChangeSignLayer(),
        # ])

        self.top_layer = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            TropEmbed(self.num_classes, initializer_w=self.initializer, lam=self.lam,
                      distance_metric="sym", name="tropical"),
            ChangeSignLayer()
        )
    
    def _build_trop_asym_max(self):
        # self.top_layer = Sequential([
        #     Dense(64, activation="relu", name="last_fc"),
        #     TropEmbed(self.num_classes, initializer_w=self.initializer, lam=self.lam, distance_metric = "asym_max", name="tropical"),
        #     ChangeSignLayer(),
        # ])

        self.top_layer = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            TropEmbed(self.num_classes, initializer_w=self.initializer, lam=self.lam,
                      distance_metric = "asym_max", name="tropical"),
            ChangeSignLayer()
        )

    def _build_trop_asym_min(self):
        # self.top_layer = Sequential([
        #     Dense(64, activation="relu", name="last_fc"),
        #     TropEmbed(self.num_classes, initializer_w=self.initializer, lam=self.lam, distance_metric = "asym_min", name="tropical"),
        #     ChangeSignLayer(),
        # ])

        self.top_layer = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            TropEmbed(self.num_classes, initializer_w=self.initializer, lam=self.lam,
                      distance_metric = "asym_min", name="tropical"),
            ChangeSignLayer()
        )

    def _build_maxout(self):
        # self.top_layer = None
        # self.dense_0 = Dense(256, activation="relu", name="last_fc")
        # self.dense_1 = Dense(self.num_maxout_neurons * self.num_classes, kernel_initializer=self.initializer)
        # self.dense_2 = Dense(self.num_maxout_neurons * self.num_classes, kernel_initializer=self.initializer)
        # self.dropout_1 = Dropout(self.dropout_rate)
        # self.dropout_2 = Dropout(self.dropout_rate)
        # self.maxout_1 = Maxout(num_units=self.num_classes, axis=-1)
        # self.maxout_2 = Maxout(num_units=self.num_classes, axis=-1)

        self.top_layer = None
        self.dense_0 = nn.Linear(256, 256)  # ReLU implemented below in maxout_top function

        self.dense_1 = nn.Linear(self.num_maxout_neurons * self.num_classes,
                                 self.num_maxout_neurons * self.num_classes)
        self.dense_2 = nn.Linear(self.num_maxout_neurons * self.num_classes,
                                 self.num_maxout_neurons * self.num_classes)

        # Apply the initializer to these layers' weights
        self.initializer(self.dense_1.weight)
        self.initializer(self.dense_2.weight)

        self.dropout_1 = nn.Dropout(self.dropout_rate)
        self.dropout_2 = nn.Dropout(self.dropout_rate)
        self.maxout_1 = Maxout(num_units=self.num_classes)
        self.maxout_2 = Maxout(num_units=self.num_classes)

    def simple_top(self, x):
        return self.top_layer(x)

    def maxout_top(self, x, training):
        # x = self.dense_0(x)
        #
        # x_1 = self.dense_1(x)
        # x_1 = self.dropout_1(x_1, training=training)
        # x_1 = self.maxout_1(x_1)
        #
        # x_2 = self.dense_2(x)
        # x_2 = self.dropout_2(x_2, training=training)
        # x_2 = self.maxout_2(x_2)

        x = F.relu(self.dense_0(x))

        x_1 = self.dense_1(x)
        x_1 = self.maxout_1(self.dropout_1(x))

        x_2 = self.dense_2(x)
        x_2 = self.maxout_2(self.dropout_2(x))

        return x_1 - x_2

    def forward(self, x):
        if self.top_layer is not None:
            return self.simple_top(x)
        else:
            return self.maxout_top(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'top': self.top,
            # 'initializer': initializers.serialize(self.initializer),
            'initializer': str(self.initializer),
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
                 # initializer=initializers.HeNormal(),
                 initializer=nn.init.kaiming_normal_,  # Should be the same as HeNormal()
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
        # self.base_layers = Sequential([
        #     Conv2D(96, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=self.input_shape, padding='same', kernel_initializer=self.initializer),
        #     MaxPooling2D(pool_size=(3, 3), strides=(1, 1)),
        #     Conv2D(256, kernel_size=(5, 5), activation='relu', padding='same', kernel_initializer=self.initializer),
        #     MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        #     Conv2D(384, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer=self.initializer),
        #     Conv2D(384, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer=self.initializer),
        #     Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer=self.initializer),
        #     MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        #     Flatten(),
        #     Dense(4096, activation='relu', kernel_initializer=self.initializer),
        #     Dropout(0.5),
        #     Dense(4096, activation='relu', kernel_initializer=self.initializer),
        #     Dropout(0.5),
        # ])

        self.base_layers = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(9216, 4096),  # Pretty sure this is the correct # of in_features
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Apply the initializer to each layer that needs it
        torch.manual_seed(0)
        for layer in self.base_layers:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                self.initializer(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)  # Initialize bias to zero

    # def call(self, inputs, training=True):
    def forward(self, inputs, training=True):
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
                 # initializer=initializers.RandomNormal(mean=0, stddev=0.1, seed=0),
                 initializer=None,  # TODO: Figure out how to initialize mean/sd (if possible)
                 num_maxout_neurons = 100, 
                 dropout_rate = 0.5,
                 input_shape = (32, 32, 3),
                 **kwargs):
        super(VGG16Model, self).__init__(num_classes = num_classes, 
                                    top = top, 
                                    initializer=initializer, 
                                    num_maxout_neurons = num_maxout_neurons, 
                                    dropout_rate = dropout_rate,
                                    **kwargs)
        self.input_shape = input_shape
        self._build_base()

    def _build_base(self):
        # self.base_layers = Sequential([
        #     VGG16(weights=None, include_top=False, input_shape=self.input_shape),
        #     Flatten(),
        #     #BatchNormalization(),
        #     #Dropout(0.5),
        #     Dense(512, activation="relu", name="fc1"),#, kernel_initializer=self.initializer),
        #     Dropout(0.5),
        #     #Dense(256, activation="relu", name="fc2"),#, kernel_initializer=self.initializer),
        #     #Dropout(0.4),
        # ])

        # Load VGG16 model
        vgg16 = models.vgg16(weights=None)

        # Remove classifier
        vgg16.classifier = nn.Identity()

        self.base_layers = nn.Sequential([
            vgg.features,
            nn.Flatten(),
            # nn.BatchNorm1d(),
            # nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
            # nn.Linear(256, 256),
            # nn.ReLU(),
            # nn.Dropout(0.4)
        ])

        # Initialize layer weights
        torch.manual_seed(0)
        for layer in self.base_layers:
            if isinstance(layer, nn.Linear):
                if self.initializer is None:
                    nn.init.normal_(layer.weight, mean=0., std=2.0)
                else:
                    self.initializer(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    # def call(self, inputs, training=True):
    def forward(self, inputs, training=True):
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
                 # initializer=initializers.RandomNormal(mean=0, stddev=1., seed=0),
                 initializer=None,
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
        # self.base_layers = Sequential([
        #     Conv2D(64, (3,3), activation='relu'),
        #     MaxPooling2D((2, 2)),
        #     Conv2D(64, (3, 3), activation='relu'),
        #     MaxPooling2D((2, 2)),
        #     Conv2D(64, (3, 3), activation='relu'),
        #     Flatten(),
        #     Dense(64, activation='relu'),
        # ])

        self.base_layers = nn.Sequential([
            nn.Conv2D(3, 64),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2),
            Conv2D(64, (3, 3), activation='relu'),
            nn.Conv2d(3, 64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(3, 64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU()
        ])

    # def call(self, inputs, training=True):
    def forward(self, inputs, training=True):
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
                 # initializer=initializers.RandomNormal(mean=0, stddev=1., seed=0),
                 initializer=None,
                 num_maxout_neurons = 100, 
                 dropout_rate = 0.5,
                 **kwargs):
        super(LeNet5, self).__init__(num_classes=num_classes,
                                    top=top,
                                    initializer=initializer, 
                                    num_maxout_neurons=num_maxout_neurons,
                                    dropout_rate=dropout_rate,
                                    **kwargs)
        self._build_base()

    def _build_base(self):
        # self.base_layers = Sequential([
        #     Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='same'),
        #     AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
        #     Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'),
        #     AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
        #     Conv2D(filters=120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'),
        #     Flatten(),
        #     Dense(units=84, activation='tanh'),
        #
        # ])

        self.base_layers = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding='same'),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(6, 15, kernel_size=5, stride=1, padding='valid'),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding='valid'),
            nn.Tanh(),
            nn.Flatten(),

            nn.Linear(in_features=480, out_features=84),
            nn.Tanh()
        )

    # def call(self, inputs, training=True):
    def forward(self, inputs, training=True):
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
                 # initializer=initializers.RandomNormal(mean=0, stddev=1., seed=0),
                 initializer=None,
                 num_maxout_neurons = 100, 
                 dropout_rate = 0.5,
                 input_shape = (32, 32, 3),
                 **kwargs):
        super(MobileNetModel, self).__init__(num_classes = num_classes, 
                                    top = top, 
                                    initializer=initializer, 
                                    num_maxout_neurons = num_maxout_neurons, 
                                    dropout_rate = dropout_rate,
                                    **kwargs)
        self.input_shape = input_shape
        self._build_base()

    def _build_base(self):
        # self.base_layers = Sequential([
        #     MobileNet(weights=None, include_top=False, input_shape=self.input_shape),
        #     GlobalAveragePooling2D(),
        # ])

        # Load MobileNet model
        mobilenet = models.mobilenet_v2(weights=None)  # TODO: Is mobilenet v2 ok with below mods?

        # Remove classifier
        mobilenet.classifier = nn.Identity()

        self.base_layers = nn.Sequential(
            mobilenet.features,          # Extract features only
            nn.AdaptiveAvgPool2d((1,1))  # Performs global average pooling
        )


    # def call(self, inputs, training=True):
    def forward(self, inputs, training=True):
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
                 # initializer=initializers.RandomNormal(mean=0, stddev=0.001, seed=0),
                 initializer=None,
                 num_maxout_neurons = 100, 
                 dropout_rate = 0.5,
                 input_shape = (32, 32, 3),
                 **kwargs):
        super(EfficientNetB4Model, self).__init__(num_classes=num_classes,
                                    top = top, 
                                    initializer=initializer, 
                                    num_maxout_neurons=num_maxout_neurons,
                                    dropout_rate=dropout_rate,
                                    **kwargs)
        self.input_shape = input_shape
        self._build_base()

    def _build_base(self):
        # self.base_layers = Sequential([
        #     EfficientNetB0(weights=None, include_top=False, input_shape=self.input_shape),
        #     GlobalAveragePooling2D(),
        # ])

        # Get EfficientNet-B0 model
        eff = models.efficientnet_b0(weights=None)

        # Remove classifier
        eff.classifier = nn.Identity()

        self.base_layers = nn.Sequential(
            eff.features,
            nn.AdaptiveAvgPool2d((1,1))
        )

    # def call(self, inputs, training=True):
    def forward(self, inputs, training=True):
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
                 # initializer=initializers.RandomNormal(mean=0, stddev=0.01, seed=0),
                 initializer=None,
                 num_maxout_neurons=100,  
                 dropout_rate = 0.5,
                 input_shape = (32, 32, 3),
                 **kwargs):
        super(ResNet50Model, self).__init__(num_classes = num_classes, 
                                    top = top, 
                                    initializer=initializer, 
                                    num_maxout_neurons = num_maxout_neurons, 
                                    dropout_rate = dropout_rate,
                                    **kwargs)
        self.input_shape = input_shape
        self._build_base()

    def _build_base(self):
        # self.base_layers = Sequential([
        #     ResNet50(weights=None, include_top=False, input_shape=self.input_shape),
        # GlobalAveragePooling2D(),
        # ])

        # Load ResNet50 model
        resnet50 = models.resnet50(weights=None)

        # Remove classifier
        resnet50.fc = nn.Identity()

        self.base_layers = resnet50

    # def call(self, inputs, training=True):
    def forward(self, inputs, training=True):
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
    def __init__(self,
                 num_classes,
                 # initializer=initializers.RandomNormal(mean=0.5, stddev=1., seed=0),
                 initializer=None,
                 **kwargs):
        super(MMRModel, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.initializer = initializer
        self._build_model()

    def _build_model(self):
        # self.conv_layer1 = Conv2D(64, (3,3), activation='relu')
        # self.max_layer1 = MaxPooling2D((2, 2))
        # self.conv_layer2 = Conv2D(64, (3, 3), activation='relu')
        # self.max_layer2 = MaxPooling2D((2, 2))
        # self.conv_layer3 = Conv2D(64, (3, 3), activation='relu')
        # self.flatten = Flatten()
        #
        # self.dense_layer1 = Dense(64, activation='relu')
        # self.dense_layer2 = Dense(64, activation='relu')
        # self.final_layer = Dense(self.num_classes, kernel_initializer=self.initializer)

        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.max_layer1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layer2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.max_layer2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layer3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)

        self.flatten = nn.Flatten()

        self.dense_layer1 = nn.Linear(in_features=1024, out_features=64)  # ReLU happens in forward pass
        self.dense_layer2 = nn.Linear(64, 64)

        self.final_layer = nn.Linear(64, self.num_classes)

        # Initialize final layer weights
        torch.manual_seed(0)
        if self.initializer is None:
            nn.init.normal_(self.final_layer.weight, mean=0.5, std=1.)


    # def call(self, inputs, training=True, return_feature_maps=False):
    def forward(self, inputs, training=True, return_feature_maps=False):
        feature_maps = []

        # x = self.conv_layer1(inputs)
        x = F.relu(self.conv_layer1(inputs))
        if return_feature_maps:
            feature_maps.append(x)

        x = self.max_layer1(x)

        # x = self.conv_layer2(x)
        x = F.relu(self.conv_layer2(x))
        if return_feature_maps:
            feature_maps.append(x)

        x = self.max_layer2(x)

        # x = self.conv_layer3(x)
        x = F.relu(self.conv_layer3(x))
        if return_feature_maps:
            feature_maps.append(x)

        x = self.flatten(x)

        # x = self.dense_layer1(x)
        x = F.relu(self.dense_layer1(x))
        if return_feature_maps:
            feature_maps.append(x)

        # x = self.dense_layer2(x)
        x = F.relu(self.dense_layer2(x))
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
            # 'initializer': initializers.serialize(self.initializer),
            'initializer': str(self.initializer),
        })
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)
