

from custom_layers.tropical_layers import TropEmbedMaxMin, TropConv2D, TropConv2DMax, TropEmbedMaxMinLogits, SoftminLayer, ChangeSignLayer, SoftmaxLayer
from custom_layers.initializers import BimodalNormalInitializer
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Concatenate, MaxPooling2D, MaxPooling1D, AveragePooling1D, Activation, Flatten, Conv2D, Dropout, Input, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Reshape, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers
from tensorflow import reduce_sum
import time

def trop_conv3layer_logits(x_train, 
                       y_train, 
                       num_epochs = 10,
                       batch_size = 64, 
                       verbose = 1,
                       training_loss = 'categorical_crossentropy',
                       num_first_filters = 32,
                       window_first_conv = (3,3),
                       initializer_w = initializers.RandomNormal(0, 0.05),
                       lam=0):
    start_time = time.time()
    model = Sequential([Conv2D(num_first_filters, window_first_conv, activation='relu'),
                        MaxPooling2D((2, 2)),                            
                        Conv2D(64, (3, 3), activation='relu'),
                        MaxPooling2D((2, 2)),                            
                        Conv2D(64, (3, 3), activation='relu'),
                        Flatten(),
                        Dense(3, activation='relu'),
                        TropEmbedMaxMinLogits(y_train.shape[1], initializer_w=initializer_w, lam=lam)])
    model.compile(optimizer='adam', loss=training_loss, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=num_epochs,batch_size=batch_size, verbose=verbose)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"trop_conv3layer_logits model built. Elapsed time: {elapsed_time:.2f} seconds | {elapsed_time/60:.2f} minutes.")
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
                       initializer_w = initializers.RandomNormal(0, 0.05)):
    start_time = time.time()
    num_classes = y_train.shape[1]
    inputs = Input(shape = x_train.shape[1:])
    first_half = Conv2D(num_first_filters, window_first_conv, activation='relu')(inputs)
    first_half = MaxPooling2D((2, 2))(first_half)                          
    first_half = Conv2D(64, (3, 3), activation='relu')(first_half) 
    first_half = MaxPooling2D((2, 2))(first_half)                             
    first_half = Conv2D(64, (3, 3), activation='relu')(first_half) 
    first_half = Flatten()(first_half) 
    dense_layer = Dense(3, activation='relu', name='dense')(first_half) 
    class_work = [TropEmbedMaxMin(p, initializer_w=initializer_w, lam=lam)(dense_layer) for _ in range(num_classes)]
    merge_layer = Concatenate(axis=-1, name='merge_layer')(class_work) 
    back_half = Sequential([Reshape((p*num_classes,1)),
                        ChangeSignLayer(),
                        MaxPooling1D(pool_size=p),
                        Flatten(),
                        SoftmaxLayer()])(merge_layer)
                        #Activation('softmax')])
    model = Model(inputs=inputs, outputs=back_half)
    print(model.summary())
    model.compile(optimizer='adam', loss=training_loss, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=num_epochs,batch_size=batch_size, verbose=verbose)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"trop_conv3layer_manyMaxLogits model built. Elapsed time: {elapsed_time:.2f} seconds | {elapsed_time/60:.2f} minutes.")
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
                       lam=0):
    start_time = time.time()
    num_classes = y_train.shape[1]
    inputs = Input(shape = x_train.shape[1:])
    first_half = Conv2D(num_first_filters, window_first_conv, activation='relu')(inputs)
    first_half = MaxPooling2D((2, 2))(first_half)                          
    first_half = Conv2D(64, (3, 3), activation='relu')(first_half) 
    first_half = MaxPooling2D((2, 2))(first_half)                             
    first_half = Conv2D(64, (3, 3), activation='relu')(first_half) 
    first_half = Flatten()(first_half) 
    dense_layer = Dense(3, activation='relu', name='dense')(first_half) 
    class_work = [TropEmbedMaxMin(p, initializer_w=initializer_w, lam=lam)(dense_layer) for _ in range(num_classes)]
    merge_layer = Concatenate(axis=-1, name = 'merge_layer')(class_work)              
    back_half = Sequential([Reshape((p*num_classes,1)),
                        AveragePooling1D(pool_size=p),
                        Flatten(),
                        SoftminLayer()])(merge_layer)
    model = Model(inputs=inputs, outputs=back_half)
    print(model.summary())
    model.compile(optimizer='adam', loss=training_loss, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=num_epochs,batch_size=batch_size, verbose=verbose)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"trop_conv3layer_manyAverageLogits model built. Elapsed time: {elapsed_time:.2f} seconds | {elapsed_time/60:.2f} minutes.")
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
                        AveragePooling1D(pool_size=p),
                        Flatten(),
                        Activation('softmax')])
    model.compile(optimizer='adam', loss=training_loss, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=num_epochs,batch_size=batch_size, verbose=verbose)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"relu_conv3layer model built. Elapsed time: {elapsed_time:.2f} seconds | {elapsed_time/60:.2f} minutes.")
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
                       window_first_conv = (3,3)):
    start_time = time.time()
    model = Sequential([Conv2D(num_first_filters, window_first_conv, activation='relu'),
                        MaxPooling2D((2, 2)),                            
                        Conv2D(64, (3, 3), activation='relu'),
                        MaxPooling2D((2, 2)),                            
                        Conv2D(64, (3, 3), activation='relu'),
                        Flatten(),
                        Dense(64, activation='relu'),
                        Dense(10, activation=final_layer_activation, kernel_initializer = initializer_relu)])
    model.compile(optimizer='adam', loss=training_loss, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=num_epochs,batch_size=batch_size, verbose=verbose)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"relu_conv3layer model built. Elapsed time: {elapsed_time:.2f} seconds | {elapsed_time/60:.2f} minutes.")
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