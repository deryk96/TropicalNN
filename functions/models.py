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

from custom_layers.tropical_layers import TropEmbedMaxMin, TropConv2D, TropConv2DMax, TropEmbedMaxMinLogits, SoftminLayer, ChangeSignLayer, SoftmaxLayer
from custom_layers.initializers import BimodalNormalInitializer
from tensorflow.keras import Sequential, Model, initializers, models
from tensorflow.keras.layers import Dense, Concatenate, MaxPooling2D, MaxPooling1D, AveragePooling1D, Activation, Flatten, Conv2D, Dropout, Input, BatchNormalization, ReLU, Add, AveragePooling2D, Reshape, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow import reduce_sum, reduce_max, subtract, reshape, convert_to_tensor, shape, cond, cast, concat
import time


from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras import Model
from tensorflow.keras.applications import ResNet50

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Layer
from tensorflow.keras.models import Sequential, Model
import time

class Maxout(Layer):
    def __init__(self, num_units, axis=-1, **kwargs):
        super(Maxout, self).__init__(**kwargs)
        self.num_units = num_units
        self.axis = axis

    def call(self, inputs):
        '''dynamic_shape = inputs.get_shape().as_list()
        # -1 for the last dimension if axis is -1
        num_channels = dynamic_shape[self.axis] if self.axis != -1 else dynamic_shape[-1]
        if num_channels % self.num_units:
            raise ValueError('number of features({}) is not a multiple of num_units({})'.format(num_channels, self.num_units))
        dynamic_shape[self.axis] = self.num_units
        dynamic_shape  += [num_channels // self.num_units]
        step1 = reshape(inputs, dynamic_shape)
        outputs = reduce_max(step1, -1)
        return outputs'''
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


class CH_MaxoutConv3Layer(Model):
    def __init__(self, num_classes, num_maxout_neurons = 10):
        super(CH_MaxoutConv3Layer, self).__init__()
        self.num_classes = num_classes
        self.num_maxout_neurons = num_maxout_neurons
        self._build_model()

    def _build_model(self):
        self.conv_layers = Sequential([
            Conv2D(64, (3,3), padding='same'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), padding='same'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), padding='same'),
            Flatten()
        ])

        self.dense_layer = Dense(64)
        self.dense_1 = Dense(self.num_maxout_neurons * self.num_classes)
        self.dense_2 = Dense(self.num_maxout_neurons * self.num_classes)
        self.maxout_1 = Maxout(num_units=self.num_classes, axis=-1)
        self.maxout_2 = Maxout(num_units=self.num_classes, axis=-1)

    def call(self, inputs, training=False):
        x = self.conv_layers(inputs)
        x = self.dense_layer(x)
        
        x_1 = self.dense_1(x)
        x_1 = self.maxout_1(x_1)

        x_2 = self.dense_2(x)
        x_2 = self.maxout_2(x_2)

        logits = x_1 - x_2
        return logits

    def compile_model(self, loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']):
        self.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit_model(self, x_train, y_train, num_epochs=10, batch_size=64, verbose=1):
        start_time = time.time()
        self.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=verbose)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Model built. Elapsed time: {elapsed_time:.2f} seconds | {elapsed_time / 60:.2f} minutes.")
        return self

class CH_MaxOut_ResNet50(Model):
    def __init__(self, num_classes, num_maxout_neurons=10, input_shape = (32, 32, 3)):
        super(CH_MaxOut_ResNet50, self).__init__()
        self.num_classes = num_classes
        self.num_maxout_neurons = num_maxout_neurons

        # Initialize ResNet50 base model
        self.resnet50_base = ResNet50(weights=None, include_top=False, input_shape=input_shape)

        # Define additional layers
        self.global_avg_pooling = GlobalAveragePooling2D()
        self.dense_layer = Dense(64, activation='relu')
        self.dense_1 = Dense(self.num_maxout_neurons * self.num_classes)
        self.dense_2 = Dense(self.num_maxout_neurons * self.num_classes)
        self.maxout_1 = Maxout(num_units=self.num_classes, axis=-1)
        self.maxout_2 = Maxout(num_units=self.num_classes, axis=-1)

    def call(self, inputs, training=False):
        x = self.resnet50_base(inputs, training=training)
        x = self.global_avg_pooling(x)
        x = self.dense_layer(x)
        x_1 = self.dense_1(x)
        x_1 = self.maxout_1(x_1)

        x_2 = self.dense_2(x)
        x_2 = self.maxout_2(x_2)

        logits = x_1 - x_2
        return logits
    
class CH_ReLU_ResNet50(Model):
    def __init__(self, num_classes, input_shape = (32, 32, 3)):
        super(CH_ReLU_ResNet50, self).__init__()
        self.num_classes = num_classes

        # Initialize ResNet50 base model
        self.resnet50_base = ResNet50(weights=None, include_top=False, input_shape=input_shape)

        # Define additional layers
        self.global_avg_pooling = GlobalAveragePooling2D()
        self.dense_layer = Dense(64, activation='relu')
        self.final_layer = Dense(num_classes)

    def call(self, inputs, training=False):
        x = self.resnet50_base(inputs, training=training)
        x = self.global_avg_pooling(x)
        x = self.dense_layer(x)
        return self.final_layer(x)


class CH_Trop_ResNet50(Model):
    def __init__(self, num_classes, initializer_w=initializers.RandomNormal(0.5, 1), lam=1, input_shape = (32, 32, 3)):
        super(CH_Trop_ResNet50, self).__init__()
        self.num_classes = num_classes

        self.initializer_w = initializer_w
        self.lam = lam
        # Initialize ResNet50 base model
        self.resnet50_base = ResNet50(weights=None, include_top=False, input_shape=input_shape)

        # Define additional layers
        self.global_avg_pooling = GlobalAveragePooling2D()
        self.dense_layer = Dense(64, activation='relu')
        self.trop_act = TropEmbedMaxMin(self.num_classes, initializer_w=self.initializer_w, lam=self.lam)
        self.change_sign = ChangeSignLayer()
        #self.final_layer = Activation('softmax')

    def call(self, inputs, training=False):
        x = self.resnet50_base(inputs, training=training)
        x = self.global_avg_pooling(x)
        x = self.dense_layer(x)
        x = self.trop_act(x)
        logits = self.change_sign(x)
        #logits = self.final_layer(x)
        return logits


class CH_ReluConv3Layer(Model):
    def __init__(self, num_classes, initializer_relu=initializers.RandomNormal(mean=0.5, stddev=1., seed=0)):
        super(CH_ReluConv3Layer, self).__init__()
        self.num_classes = num_classes
        self.initializer_relu = initializer_relu
        self._build_model()

    def _build_model(self):
        self.conv_layers = Sequential([
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            Flatten()
        ])

        self.dense_layer = Dense(64, activation='relu')
        self.final_layer = Dense(self.num_classes, kernel_initializer=self.initializer_relu)

    def call(self, inputs, training=False):
        x = self.conv_layers(inputs)
        x = self.dense_layer(x)
        logits = self.final_layer(x)
        return logits

    def compile_model(self, loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']):
        self.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit_model(self, x_train, y_train, num_epochs=10, batch_size=64, verbose=1):
        start_time = time.time()
        self.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=verbose)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Model built. Elapsed time: {elapsed_time:.2f} seconds | {elapsed_time / 60:.2f} minutes.")
        return self


class CH_TropConv3LayerLogits(Model):
    def __init__(self, num_classes, initializer_w=initializers.RandomNormal(0.5, 1), lam=1):
        super(CH_TropConv3LayerLogits, self).__init__()
        self.num_classes = num_classes
        self.initializer_w = initializer_w
        self.lam = lam

        self._build_model()

    def _build_model(self):
        self.conv_layers = Sequential([
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            Flatten()
        ])

        self.dense_layer = Dense(64, activation='relu', name='dense')
        self.final_layer = TropEmbedMaxMinLogits(self.num_classes, initializer_w=self.initializer_w, lam=self.lam)

    def call(self, inputs, training=False):
        x = self.conv_layers(inputs)
        x = self.dense_layer(x)
        logits = self.final_layer(x)
        return logits

    def compile_model(self, loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']):
        self.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit_model(self, x_train, y_train, num_epochs=10, batch_size=64, verbose=1):
        start_time = time.time()
        self.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=verbose)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Model built. Elapsed time: {elapsed_time:.2f} seconds | {elapsed_time / 60:.2f} minutes.")

        return self
    

def functional_conv(num_first_filters, window_first_conv, inputs):
    first_half = Sequential([Conv2D(num_first_filters, window_first_conv, activation='relu'),
                             MaxPooling2D((2, 2)), 
                             Conv2D(64, (3, 3), activation='relu'), 
                             MaxPooling2D((2, 2)),
                             Conv2D(64, (3, 3), activation='relu'),
                             Flatten()])(inputs)
    return first_half


def pre_model(x_train, y_train):
    start_time = time.time()
    num_classes = y_train.shape[1]
    inputs = Input(shape = x_train.shape[1:])
    return start_time, num_classes, inputs


def post_model(start_time, name):
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{name} model built. Elapsed time: {elapsed_time:.2f} seconds | {elapsed_time/60:.2f} minutes.")


def functional_build_model(x_train, y_train, inputs, back_half, training_loss, num_epochs, batch_size, verbose):
    model = Model(inputs=inputs, outputs=back_half)
    print(model.summary())
    model.compile(optimizer='adam', loss=training_loss, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=num_epochs,batch_size=batch_size, verbose=verbose)
    return model


def trop_conv3layer_logits(x_train, 
                       y_train, 
                       num_epochs = 10,
                       batch_size = 64, 
                       verbose = 1,
                       training_loss = 'categorical_crossentropy',
                       num_first_filters = 32,
                       window_first_conv = (3,3),
                       initializer_w = initializers.RandomNormal(0, 0.05),
                       lam=0,
                       num_neurons_b4_logits = 64):
    start_time, num_classes, inputs = pre_model(x_train, y_train)

    first_half = functional_conv(num_first_filters, window_first_conv, inputs)
    dense_layer = Dense(num_neurons_b4_logits, activation='relu', name='dense')(first_half) 
    logit_layer = TropEmbedMaxMinLogits(num_classes, initializer_w=initializer_w, lam=lam)(dense_layer)

    model = functional_build_model(x_train, y_train, inputs, logit_layer, training_loss, num_epochs, batch_size, verbose)
    post_model(start_time, 'trop_conv3layer_logits')
    return model


def maxout_network(x_train, 
                       y_train, 
                       num_epochs = 10,
                       batch_size = 64, 
                       verbose = 1,
                       initializer_relu = initializers.RandomNormal(mean=0.5, stddev=1., seed=0), 
                       training_loss = 'categorical_crossentropy',
                       num_first_filters = 32,
                       window_first_conv = (3,3),
                       p=3,
                       num_neurons_b4_logits=64):
    start_time, num_classes, inputs = pre_model(x_train, y_train)

    first_half = functional_conv(num_first_filters, window_first_conv, inputs)
    dense_layer = Dense(num_neurons_b4_logits, activation='relu', name='dense')(first_half) 
    z_layer = [[Dense(p, initializer_w=initializer_relu, name=f'class_{i}_add')(dense_layer),Dense(p, initializer_w=initializer_relu, name=f'class_{i}_sub')(dense_layer)] for i in range(num_classes)]
    h_layer = [[reduce_max(z_layer[i][0], axis=0), reduce_max(z_layer[i][1], axis=0)] for i in range(num_classes)]
    g_layer = [subtract(h_layer[i][0], h_layer[i][1]) for i in range(num_classes)]
    merge_layer = Concatenate(axis=-1, name = 'merge_layer')(g_layer)      
    logits = SoftmaxLayer()(merge_layer)
    model = functional_build_model(x_train, y_train, inputs, logits, training_loss, num_epochs, batch_size, verbose)
    post_model(start_time, 'maxout_network')
    return model


def trop_conv3layer_manyMaxLogits(x_train, 
                       y_train, 
                       num_epochs = 10,
                       batch_size = 64, 
                       verbose = 1,
                       p = 3,
                       lam=1.0,
                       training_loss = 'categorical_crossentropy',
                       num_first_filters = 32,
                       window_first_conv = (3,3),
                       initializer_w = initializers.RandomNormal(0, 0.05),
                       num_neurons_b4_logits=64):
    start_time, num_classes, inputs = pre_model(x_train, y_train)

    first_half = functional_conv(num_first_filters, window_first_conv, inputs)
    dense_layer = Dense(num_neurons_b4_logits, activation='relu', name='dense')(first_half) 
    class_work = [TropEmbedMaxMin(p, initializer_w=initializer_w, lam=lam, name=f'class_{i}')(dense_layer) for i in range(num_classes)]
    merge_layer = Concatenate(axis=-1, name='merge_layer')(class_work) 
    back_half = Sequential([Reshape((p*num_classes,1)),
                        ChangeSignLayer(),
                        MaxPooling1D(pool_size=p),
                        Flatten(),
                        SoftmaxLayer()])(merge_layer)
    
    model = functional_build_model(x_train, y_train, inputs, back_half, training_loss, num_epochs, batch_size, verbose)
    post_model(start_time, 'trop_conv3layer_manyMaxLogits')
    return model


def trop_conv3layer_manyAverageLogits(x_train, 
                       y_train, 
                       num_epochs = 10,
                       batch_size = 32, 
                       verbose = 1,
                       p = 3,
                       training_loss = 'categorical_crossentropy',
                       num_first_filters = 32,
                       window_first_conv = (3,3),
                       initializer_w = initializers.RandomNormal(0, 0.1),
                       lam=0,
                       num_neurons_b4_logits = 64):
    start_time, num_classes, inputs = pre_model(x_train, y_train)

    first_half = functional_conv(num_first_filters, window_first_conv, inputs)
    dense_layer = Dense(num_neurons_b4_logits, activation='relu', name='dense')(first_half) 
    class_work = [TropEmbedMaxMin(p, initializer_w=initializer_w, lam=lam, name=f'class_{i}')(dense_layer) for i in range(num_classes)]
    merge_layer = Concatenate(axis=-1, name = 'merge_layer')(class_work)              
    back_half = Sequential([Reshape((p*num_classes,1)),
                        AveragePooling1D(pool_size=p),
                        Flatten(),
                        SoftminLayer()])(merge_layer)
    
    model = functional_build_model(x_train, y_train, inputs, back_half, training_loss, num_epochs, batch_size, verbose)
    post_model(start_time, 'trop_conv3layer_manyAverageLogits')
    return model


def relu_binary(x_train, 
                    y_train, 
                    num_epochs = 10,
                    first_layer_size = 100, 
                    verbose = 0, 
                    initializer_w = initializers.RandomNormal(mean=0.5, stddev=1., seed=0),
                    second_layer_size = 1,
                    second_layer_activation = 'sigmoid', 
                    clipnorm = None,
                    clipvalue = None,
                    training_loss = 'binary_crossentropy'):
    start_time = time.time()
    num_predictors = x_train.shape[1]
    initializer_out = initializers.RandomNormal(mean=0.5, stddev=1., seed=0)
    model = Sequential([Dense(first_layer_size, input_shape=(num_predictors,), activation="relu",  kernel_initializer=initializer_w),
                        Dense(second_layer_size, activation=second_layer_activation,  kernel_initializer=initializer_out)])
    model.compile(optimizer=Adam(0.1, clipnorm=clipnorm, clipvalue=clipvalue),loss=training_loss, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=num_epochs, verbose=verbose)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"relu_binary model built. Elapsed time: {elapsed_time:.2f} seconds | {elapsed_time/60:.2f} minutes.")
    return model


def trop_binary(x_train, 
                       y_train, 
                       num_epochs = 10,
                       first_layer_size = 100, 
                       verbose = 0, 
                       initializer_w = initializers.random_normal,
                       second_layer_size = 1,
                       second_layer_activation = 'sigmoid', 
                       clipnorm = None,
                       clipvalue = None,
                       training_loss = 'binary_crossentropy',
                       lam = 0.01,
                       boo_dropout = False,
                       p_dropout = 0.5):
    start_time = time.time() 
    initializer_out = initializers.RandomNormal(mean=0.5, stddev=1., seed=0)
    if boo_dropout:
        model = Sequential([TropEmbedMaxMin(first_layer_size, initializer_w = initializer_w, lam=lam),
                            Dropout(p_dropout),
                            Dense(second_layer_size, activation=second_layer_activation,  kernel_initializer=initializer_out)])
    else:
        model = Sequential([TropEmbedMaxMin(first_layer_size, initializer_w = initializer_w, lam=lam),
                        Dense(second_layer_size, activation=second_layer_activation,  kernel_initializer=initializer_out)])
    model.compile(optimizer=Adam(0.1, clipnorm=clipnorm, clipvalue=clipvalue), loss=training_loss, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=num_epochs, verbose=verbose)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"trop_binary model built. Elapsed time: {elapsed_time:.2f} seconds | {elapsed_time/60:.2f} minutes.")
    return model


def relu_conv2layer(x_train, 
                       y_train, 
                       num_epochs = 10,
                       batch_size = 64, 
                       verbose = 1,
                       initializer_relu = initializers.RandomNormal(mean=0.5, stddev=1., seed=0),
                       final_layer_activation = 'softmax', 
                       training_loss = 'categorical_crossentropy',
                       num_first_filters = 32):
    start_time = time.time()
    model = Sequential([Conv2D(num_first_filters, (3, 3), activation='relu'),
                        MaxPooling2D((2, 2)),                            
                        Conv2D(32, (3, 3), activation='relu'),
                        Flatten(),
                        Dense(10, activation=final_layer_activation, kernel_initializer = initializer_relu)])
    model.compile(optimizer='adam', loss=training_loss, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=num_epochs,batch_size=batch_size, verbose=verbose)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"relu_conv2layer model built. Elapsed time: {elapsed_time:.2f} seconds | {elapsed_time/60:.2f} minutes.")
    return model


def trop_conv2layer_logits(x_train, 
                       y_train, 
                       num_epochs = 10,
                       batch_size = 64, 
                       verbose = 1,
                       initializer_relu = initializers.RandomNormal(mean=0.5, stddev=1., seed=0),
                       final_layer_activation = 'softmax', 
                       training_loss = 'categorical_crossentropy',
                       num_first_filters = 32):
    start_time = time.time()
    model = Sequential([Conv2D(num_first_filters, (3, 3), activation='relu'),
                        MaxPooling2D((2, 2)),                            
                        Conv2D(32, (3, 3), activation='relu'),
                        Flatten(),
                        TropEmbedMaxMinLogits(10)])
    model.compile(optimizer='adam', loss=training_loss, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=num_epochs,batch_size=batch_size, verbose=verbose)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"trop_conv2layer_logits model built. Elapsed time: {elapsed_time:.2f} seconds | {elapsed_time/60:.2f} minutes.")
    return model


def relu_conv3layer_manyAvgerageLogits(x_train, 
                       y_train, 
                       num_epochs = 10,
                       batch_size = 64, 
                       verbose = 1,
                       initializer_relu = initializers.RandomNormal(mean=0.5, stddev=1., seed=0), 
                       training_loss = 'categorical_crossentropy',
                       num_first_filters = 32,
                       window_first_conv = (3,3),
                       p=3,
                       num_neurons_b4_logits=64):
    start_time, num_classes, inputs = pre_model(x_train, y_train)

    first_half = functional_conv(num_first_filters, window_first_conv, inputs)
    dense_layer = Dense(num_neurons_b4_logits, activation='relu')(first_half) 
    #class_work = [Dense(p, kernel_initializer=initializer_relu, name=f'class_{i}')(dense_layer) for i in range(num_classes)]
    #merge_layer = Concatenate(axis=-1, name = 'merge_layer')(class_work)              
    merge_layer = Dense(p*num_classes, kernel_initializer=initializer_relu)(dense_layer)
    back_half = Sequential([Reshape((p*num_classes,1)),
                        AveragePooling1D(pool_size=p),
                        Flatten(),
                        SoftmaxLayer()])(merge_layer)
    
    model = functional_build_model(x_train, y_train, inputs, back_half, training_loss, num_epochs, batch_size, verbose)
    post_model(start_time, 'relu_conv3layer_manyAvgerageLogits')
    return model


def relu_conv3layer_manyMaxLogits(x_train, 
                       y_train, 
                       num_epochs = 10,
                       batch_size = 64, 
                       verbose = 1,
                       initializer_relu = initializers.RandomNormal(mean=0, stddev=0.2, seed=0),
                       final_layer_activation = 'softmax', 
                       training_loss = 'categorical_crossentropy',
                       num_first_filters = 32,
                       window_first_conv = (3,3),
                       p=3):
    start_time = time.time()
    model = Sequential([Conv2D(num_first_filters, window_first_conv, activation='relu'),
                        MaxPooling2D((2, 2)),                            
                        Conv2D(64, (3, 3), activation='relu'),
                        MaxPooling2D((2, 2)),                            
                        Conv2D(64, (3, 3), activation='relu'),
                        Flatten(),
                        Dense(64, activation='relu'),
                        Dense(p*10, activation=final_layer_activation, kernel_initializer = initializer_relu),
                        Reshape((p*10,1)),
                        MaxPooling1D(pool_size=p),
                        Flatten()])
                        #Activation('softmax')])
    model.compile(optimizer='adam', loss=training_loss, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=num_epochs,batch_size=batch_size, verbose=verbose)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"relu_conv3layer model built. Elapsed time: {elapsed_time:.2f} seconds | {elapsed_time/60:.2f} minutes.")
    return model


def relu_conv3layer(x_train, 
                       y_train, 
                       num_epochs = 10,
                       batch_size = 64, 
                       verbose = 1,
                       initializer_relu = initializers.RandomNormal(mean=0.5, stddev=1., seed=0),
                       final_layer_activation = 'softmax', 
                       training_loss = 'categorical_crossentropy',
                       num_first_filters = 32,
                       window_first_conv = (3,3),
                       num_neurons_b4_logits = 64):
    start_time, num_classes, inputs = pre_model(x_train, y_train)

    first_half = functional_conv(num_first_filters, window_first_conv, inputs)
    dense_layer = Dense(num_neurons_b4_logits, activation='relu')(first_half) 
    logit_layer = Dense(num_classes, activation=final_layer_activation, kernel_initializer = initializer_relu)(dense_layer)
    model = functional_build_model(x_train, y_train, inputs, logit_layer, training_loss, num_epochs, batch_size, verbose)
    post_model(start_time, 'trop_conv3layer_logits')
    return model


def trop_conv2layer(x_train, 
                       y_train, 
                       num_epochs = 10,
                       batch_size = 64, 
                       verbose = 1,
                       initializer_relu = initializers.RandomNormal(mean=0.5, stddev=1., seed=0),
                       final_layer_activation = 'softmax', 
                       training_loss = 'categorical_crossentropy',
                       initializer_w = initializers.RandomNormal(mean=0, stddev=0.05, seed=0),
                       lam=0.0,
                       num_first_filters = 32):
    start_time = time.time() 
    model = Sequential([TropConv2D(filters=num_first_filters, initializer_w=initializer_w, lam=lam),
                        MaxPooling2D((2, 2)),                            
                        Conv2D(32, (3, 3), activation='relu'),
                        Flatten(),
                        Dense(10, activation=final_layer_activation, kernel_initializer = initializer_relu)])
    model.compile(optimizer='adam', loss=training_loss, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=num_epochs,batch_size=batch_size, verbose=verbose)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"trop_conv2layer model built. Elapsed time: {elapsed_time:.2f} seconds | {elapsed_time/60:.2f} minutes.")
    return model


def trop_conv3layer(x_train, 
                       y_train, 
                       num_epochs = 10,
                       batch_size = 64, 
                       verbose = 1,
                       initializer_relu = initializers.RandomNormal(mean=0.5, stddev=1., seed=0),
                       final_layer_activation = 'softmax', 
                       training_loss = 'categorical_crossentropy',
                       initializer_w = initializers.random_normal,
                       lam=0.0,
                       num_first_filters = 32,
                       window_first_conv = (3,3)):
    start_time = time.time()
    model = Sequential([TropConv2D(filters=num_first_filters, window_size= [1,window_first_conv[0],window_first_conv[1],1], initializer_w=initializer_w, lam=lam),
                        MaxPooling2D((2, 2), strides=(1, 1)),                            
                        Conv2D(64, (1, 1), activation='relu'),
                        MaxPooling2D((2, 2), strides=(1, 1)),                            
                        Conv2D(64, (1, 1), activation='relu'),
                        Flatten(),
                        Dense(64, activation='relu'),
                        Dense(10, activation=final_layer_activation, kernel_initializer = initializer_relu)])
    model.compile(optimizer='adam', loss=training_loss, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=num_epochs,batch_size=batch_size, verbose=verbose)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"trop_conv3layer model built. Elapsed time: {elapsed_time:.2f} seconds | {elapsed_time/60:.2f} minutes.")
    return model