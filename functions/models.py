from custom_layers.tropical_layers import TropEmbedMaxMin, TropConv2D
from custom_layers.initializers import BimodalNormalInitializer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, MaxPooling2D, Flatten, Conv2D, Dropout
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
    num_predictors = x_train.shape[1] #assumes flattened array input
    initializer_out = initializers.RandomNormal(mean=0.5, stddev=1., seed=0)
    if boo_dropout:
        model = Sequential([TropEmbedMaxMin(first_layer_size, num_predictors, initializer_w = initializer_w, lam=lam),
                            Dropout(p_dropout),
                            Dense(second_layer_size, activation=second_layer_activation,  kernel_initializer=initializer_out)])
    else:
        model = Sequential([TropEmbedMaxMin(first_layer_size, num_predictors, initializer_w = initializer_w, lam=lam),
                        Dense(second_layer_size, activation=second_layer_activation,  kernel_initializer=initializer_out)])
    model.compile(optimizer=Adam(0.1, clipnorm=clipnorm, clipvalue=clipvalue), loss=training_loss, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=num_epochs, verbose=verbose)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Tropical model built. Elapsed time: {elapsed_time:.2f} seconds | {elapsed_time/60:.2f} minutes.")
    return model


def conv_tropical_3layer_model(x_train, 
                       y_train, 
                       num_epochs = 10,
                       batch_size = 64, 
                       verbose = 1, 
                       initializer_trop = BimodalNormalInitializer(stddev=1,high=5.5, low=-4.5),
                       initializer_relu = initializers.RandomNormal(mean=0.5, stddev=1., seed=0),
                       final_layer_activation = 'softmax', 
                       training_loss = 'categorical_crossentropy',
                       lam = 0):
    start_time = time.time() 
    in_channels = x_train.shape[-1]
    model = Sequential([TropConv2D(filters=32, channels=in_channels, window_size = [1, 3, 3, 1], strides = [1, 1, 1, 1], initializer_w=initializer_trop, lam=lam),
                        MaxPooling2D((2, 2)),                            
                        TropConv2D(filters=32, channels=32, window_size = [1, 3, 3, 1], strides = [1, 1, 1, 1], initializer_w=initializer_trop, lam=lam),
                        MaxPooling2D((2, 2)),
                        TropConv2D(filters=32, channels=32, window_size = [1, 3, 3, 1], strides = [1, 1, 1, 1], initializer_w=initializer_trop, lam=lam),
                        Flatten(),
                        Dense(64, activation='relu', kernel_initializer = initializer_relu),
                        Dense(10, activation=final_layer_activation, kernel_initializer = initializer_relu)])
    model.compile(optimizer='adam', loss=training_loss, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=num_epochs,batch_size=batch_size, verbose=verbose)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Tropical model built. Elapsed time: {elapsed_time:.2f} seconds | {elapsed_time/60:.2f} minutes.")
    #model.build(input_shape=(32, 28, 28, 1))
    #model.summary()
    return model


def conv_relu_3layer_model(x_train, 
                       y_train, 
                       num_epochs = 10,
                       batch_size = 64, 
                       verbose = 1,
                       initializer_relu = initializers.RandomNormal(mean=0.5, stddev=1., seed=0),
                       final_layer_activation = 'softmax', 
                       training_loss = 'categorical_crossentropy',):
    start_time = time.time() 
    input_shape = x_train[0].shape
    model = Sequential([Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
                        MaxPooling2D((2, 2)),                            
                        Conv2D(32, (3, 3), activation='relu'),
                        MaxPooling2D((2, 2)),
                        Conv2D(32, (3, 3), activation='relu'),
                        Flatten(),
                        Dense(64, activation='relu', kernel_initializer = initializer_relu),
                        Dense(10, activation=final_layer_activation, kernel_initializer = initializer_relu)])
    model.compile(optimizer='adam', loss=training_loss, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=num_epochs,batch_size=batch_size, verbose=verbose)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Tropical model built. Elapsed time: {elapsed_time:.2f} seconds | {elapsed_time/60:.2f} minutes.")
    #model.build(input_shape=(32, 28, 28, 1))
    #model.summary()
    return model


def conv_relu_then_trop__3layer_model(x_train, 
                       y_train, 
                       num_epochs = 10,
                       batch_size = 64, 
                       verbose = 1,
                       initializer_relu = initializers.RandomNormal(mean=0.5, stddev=1., seed=0),
                       final_layer_activation = 'softmax', 
                       training_loss = 'categorical_crossentropy',
                       initializer_w = BimodalNormalInitializer(stddev=1,high=5.5, low=-4.5),
                       lam=0.0):
    start_time = time.time() 
    input_shape = x_train[0].shape
    model = Sequential([Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
                        MaxPooling2D((2, 2)),                            
                        Conv2D(32, (3, 3), activation='relu'),
                        MaxPooling2D((2, 2)),
                        Conv2D(32, (3, 3), activation='relu'),
                        Flatten(),
                        TropEmbedMaxMin(64, num_predictors, initializer_w = initializer_w, lam=lam),
                        Dense(10, activation=final_layer_activation, kernel_initializer = initializer_relu)])
    model.compile(optimizer='adam', loss=training_loss, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=num_epochs,batch_size=batch_size, verbose=verbose)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Tropical model built. Elapsed time: {elapsed_time:.2f} seconds | {elapsed_time/60:.2f} minutes.")
    #model.build(input_shape=(32, 28, 28, 1))
    #model.summary()
    return model