from custom_layers.tropical_layers import TropEmbedMaxMin, TropConv2D
from custom_layers.initializers import BimodalNormalInitializer
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, MaxPooling2D, Flatten, Conv2D, Dropout, Input, BatchNormalization, ReLU, Add, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers
import time


def simple_relu_model(x_train, 
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
    print(f"ReLU model built. Elapsed time: {elapsed_time:.2f} seconds | {elapsed_time/60:.2f} minutes.")
    return model


def simple_tropical_model(x_train, 
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
    print(f"Tropical model built. Elapsed time: {elapsed_time:.2f} seconds | {elapsed_time/60:.2f} minutes.")
    return model


def simple_conv_relu_model(x_train, 
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
                        MaxPooling2D((2, 2), strides=(1, 1)),                            
                        Conv2D(32, (3, 3), activation='relu'),
                        Flatten(),
                        Dense(10, activation=final_layer_activation, kernel_initializer = initializer_relu)])
    model.compile(optimizer='adam', loss=training_loss, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=num_epochs,batch_size=batch_size, verbose=verbose)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"ReLU convolution model built. Elapsed time: {elapsed_time:.2f} seconds | {elapsed_time/60:.2f} minutes.")
    return model


def simple_3layer_conv_relu_model(x_train, 
                       y_train, 
                       num_epochs = 10,
                       batch_size = 64, 
                       verbose = 1,
                       initializer_relu = initializers.RandomNormal(mean=0.5, stddev=1., seed=0),
                       final_layer_activation = 'softmax', 
                       training_loss = 'categorical_crossentropy',
                       num_first_filters = 32,
                       window_first_conv = (3,3)):
    start_time = time.time()
    model = Sequential([Conv2D(num_first_filters, window_first_conv, activation='relu'),
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
    print(f"ReLU convolution model built. Elapsed time: {elapsed_time:.2f} seconds | {elapsed_time/60:.2f} minutes.")
    return model


def simple_conv_trop_model(x_train, 
                       y_train, 
                       num_epochs = 10,
                       batch_size = 64, 
                       verbose = 1,
                       initializer_relu = initializers.RandomNormal(mean=0.5, stddev=1., seed=0),
                       final_layer_activation = 'softmax', 
                       training_loss = 'categorical_crossentropy',
                       initializer_w = initializers.random_normal,
                       lam=0.0,
                       num_first_filters = 32):
    start_time = time.time() 
    model = Sequential([TropConv2D(filters=num_first_filters, initializer_w=initializer_w, lam=lam),
                        MaxPooling2D((2, 2), strides=(1, 1)),                            
                        Conv2D(32, (3, 3), activation='relu'),
                        Flatten(),
                        Dense(10, activation=final_layer_activation, kernel_initializer = initializer_relu)])
    model.compile(optimizer='adam', loss=training_loss, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=num_epochs,batch_size=batch_size, verbose=verbose)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Tropical convolution model built. Elapsed time: {elapsed_time:.2f} seconds | {elapsed_time/60:.2f} minutes.")
    return model


def simple_3layer_conv_trop_model(x_train, 
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
    print(f"ReLU convolution model built. Elapsed time: {elapsed_time:.2f} seconds | {elapsed_time/60:.2f} minutes.")
    return model


def residual_block(x, filters, kernel_size=3, stride=1):
    # Shortcut path
    shortcut = x
    
    # First convolution layer
    x = Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Second convolution layer
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    
    # If the shortcut and main paths have different shapes, add a 1x1 convolution
    if x.shape[-1] != shortcut.shape[-1] or stride != 1:
        shortcut = Conv2D(filters, 1, strides=stride)(shortcut)
    
    # Add the shortcut to the main path
    x = Add()([x, shortcut])
    x = ReLU()(x)
    
    return x

# Define the ResNet-50 model
def ReLUResNet50(input_shape=(224, 224, 3), num_classes=1000):
    input_tensor = Input(shape=input_shape)
    
    x = Conv2D(64, 7, strides=2, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Residual blocks
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    
    x = residual_block(x, 512, stride=2)
    x = residual_block(x, 512)
    x = residual_block(x, 512)
    
    x = GlobalAveragePooling2D()(x)
    
    output = Dense(num_classes, activation='softmax')(x)
    
    return Model(input_tensor, output)

def TropicalResNet50(input_shape=(224, 224, 3), num_classes=1000):
    input_tensor = Input(shape=input_shape)

    x = TropConv2D(64, window_size = [1, 7, 7, 1], strides = [1, 2, 2, 1], padding = 'SAME')(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Residual blocks
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    
    x = residual_block(x, 512, stride=2)
    x = residual_block(x, 512)
    x = residual_block(x, 512)
    
    x = GlobalAveragePooling2D()(x)
    
    output = Dense(num_classes, activation='softmax')(x)
    
    return Model(input_tensor, output)