# tropical_layers.py
# Description: This file contains a collection of class object for customized tropical layers to be used in Tensorflow neural networks.
# Author: Kurt Pasque
# Date: November 22, 2023

'''
Module: tropical_layers.py

This file contains a collection of class object for customized tropical layers to be used in Tensorflow neural networks.

Functions:
- ChangeSignLayer : Takes flat inputs and multiplies by -1
- SoftminLayer : Takes flat inputs and applieds the softmin activation function to it. 
- SoftmaxLayer : Takes flat inputs and applieds the softmax activation function to it. 
- TropReg : Projected-gradient-descent attack on batch of input vectors given label, loss object, and model.
- attackTestSetBatch : Attack a whole set of data batch-by-batch given loss object, model, data, and type of attack.
'''

from tensorflow import reshape, expand_dims, reduce_max ,reduce_min,reduce_sum, float32, transpose, shape, ones, bool, exp, boolean_mask, zeros, concat, add, fill, constant
from tensorflow.math import top_k, reduce_sum, exp, logical_not, maximum, minimum, scalar_mul
from tensorflow.keras.layers import Layer
from tensorflow.keras.backend import repeat_elements
from tensorflow.keras import initializers, regularizers
from tensorflow.image import extract_patches
from tensorflow.linalg import band_part
from keras.constraints import NonNeg

class ChangeSignLayer(Layer):
    '''
    Custom TensorFlow layer to change the sign of the input tensor.
    '''

    def __init__(self, **kwargs):
        '''
        Initializes the ChangeSignLayer.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments for the Layer superclass.
        '''
        super(ChangeSignLayer, self).__init__(**kwargs)  # Initialize the Layer superclass

    def call(self, inputs):
        '''
        Performs the forward pass of the layer.

        Parameters
        ----------
        inputs : tensorflow tensor object
            Input tensor to change the sign.

        Returns
        -------
        output : tensorflow tensor object
            Output tensor with signs changed.
        '''
        return add(constant(50.0), scalar_mul(-1.0, inputs))# Change the sign of the input tensor by multiplying with -1
    

class SoftminLayer(Layer):
    '''
    Custom TensorFlow layer implementing the Softmin activation function.
    '''

    def __init__(self, **kwargs):
        '''
        Initializes the SoftminLayer.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments for the Layer superclass.
        '''
        super(SoftminLayer, self).__init__(**kwargs)  # Initialize the Layer superclass

    def call(self, inputs):
        '''
        Performs the forward pass of the SoftminLayer.

        Parameters
        ----------
        inputs : tensorflow tensor object
            Input tensor.

        Returns
        -------
        output : tensorflow tensor object
            Output tensor after applying the Softmin activation.
        '''
        negative_exponents = exp(-inputs)
        return negative_exponents / reduce_sum(negative_exponents, axis=-1, keepdims=True)


class SoftmaxLayer(Layer):
    '''
    Custom TensorFlow layer implementing the Softmax activation function.
    '''

    def __init__(self, **kwargs):
        '''
        Initializes the SoftmaxLayer.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments for the Layer superclass.
        '''
        super(SoftmaxLayer, self).__init__(**kwargs)  # Initialize the Layer superclass

    def call(self, inputs):
        '''
        Performs the forward pass of the SoftmaxLayer.

        Parameters
        ----------
        inputs : tensorflow tensor object
            Input tensor.

        Returns
        -------
        output : tensorflow tensor object
            Output tensor after applying the Softmax activation.
        '''
        exponents = exp(inputs)
        return exponents / reduce_sum(exponents, axis=-1, keepdims=True)
    

class TropReg(regularizers.Regularizer):
    '''
    Custom TensorFlow regularizer implementing a regularization on Tropical weights where the spread of the weights is penalized. 
    Serves to keep weights in a narrower band where the max weight and min weight are drawn closer together across the whole set of weights.
    '''

    def __init__(self, lam=1.0):
        '''
        Initializes the TropReg regularizer.

        Parameters
        ----------
        lam : float, optional
            Regularization parameter (default is 1.0).
        '''
        self.lam = lam

    def __call__(self, weight_matrix):
        '''
        Calculates the Tropical regularization term.

        Parameters
        ----------
        weight_matrix : tensorflow tensor object
            Weight matrix of a layer.

        Returns
        -------
        regularization_term : float
            Tropical regularization term.
        '''
        max_vals, _ = top_k(weight_matrix, 1)
        min_vals, _ = top_k(-weight_matrix, 1)
        return self.lam * reduce_sum(max_vals[:, 0] + min_vals[:, 0])


class TropRegIncreaseDistance(regularizers.Regularizer):
    '''
    Custom TensorFlow regularizer implementing Tropical regularization
    to increase distances between weights in a layer. Penalizes weights 
    that are close to one another. Serves to "spread" the weights out in 
    an attempt to create a set of points that can more robustly define 
    the decision boundaries of input data. 
    '''

    def __init__(self, lam=1.0):
        '''
        Initializes the TropRegIncreaseDistance regularizer.

        Parameters
        ----------
        lam : float, optional
            Regularization parameter (default is 1.0).
        '''
        self.lam = lam

    def __call__(self, weight_matrix):
        '''
        Calculates the Tropical regularization term to increase distances between weights.

        Parameters
        ----------
        weight_matrix : tensorflow tensor object
            Weight matrix of a layer.

        Returns
        -------
        regularization_term : float
            Tropical regularization term to increase distances between weights.
        '''
        reshaped_weights = expand_dims(weight_matrix, 1)  # Reshape weights to have an additional dimension
        result_addition = reshaped_weights + transpose(reshaped_weights, perm=[1, 0, 2])  # Add weight matrices and their transposes
        tropical_distances = reduce_max(result_addition, axis=2) - reduce_min(result_addition, axis=2)  # Calculate tropical distances
        n = shape(tropical_distances)[0]  # Get the shape of tropical distances
        mask = band_part(ones((n, n), dtype=bool), 0, -1)  # Create a mask to exclude the main diagonal
        flat_vector = boolean_mask(tropical_distances, logical_not(mask))  # Extract values not in the main diagonal
        return self.lam * exp(-reduce_min(flat_vector))  # Apply exponential and scaling to obtain regularization term


class TropRegDecreaseDistance(regularizers.Regularizer):
    '''
    Custom TensorFlow regularizer implementing Tropical regularization
    to increase distances between weights in a layer. Penalizes weights 
    that are close to one another. Serves to "spread" the weights out in 
    an attempt to create a set of points that can more robustly define 
    the decision boundaries of input data. 
    '''

    def __init__(self, lam=1.0):
        '''
        Initializes the TropRegIncreaseDistance regularizer.

        Parameters
        ----------
        lam : float, optional
            Regularization parameter (default is 1.0).
        '''
        self.lam = lam

    def __call__(self, weight_matrix):
        '''
        Calculates the Tropical regularization term to increase distances between weights.

        Parameters
        ----------
        weight_matrix : tensorflow tensor object
            Weight matrix of a layer.

        Returns
        -------
        regularization_term : float
            Tropical regularization term to increase distances between weights.
        '''
        reshaped_weights = expand_dims(weight_matrix, 1)  # Reshape weights to have an additional dimension
        result_addition = reshaped_weights + transpose(reshaped_weights, perm=[1, 0, 2])  # Add weight matrices and their transposes
        tropical_distances = reduce_max(result_addition, axis=2) - reduce_min(result_addition, axis=2)  # Calculate tropical distances
        n = shape(tropical_distances)[0]  # Get the shape of tropical distances
        mask = band_part(ones((n, n), dtype=bool), 0, -1)  # Create a mask to exclude the main diagonal
        flat_vector = boolean_mask(tropical_distances, logical_not(mask))  # Extract values not in the main diagonal
        return self.lam * reduce_max(flat_vector)  # Take max and multiple by lambda to obtain regularization term


class TropEmbedTop2(Layer):
    '''
    Custom TensorFlow layer implementing Tropical Embedding for top 2 values.
    '''

    def __init__(self, units=2, input_dim=3):
        '''
        Initializes the TropEmbedTop2 layer.

        Parameters
        ----------
        units : int, optional
            Number of output units (default is 2).
        input_dim : int, optional
            Dimension of input data (default is 3).
        '''
        super(TropEmbedTop2, self).__init__()
        self.w = self.add_weight(
            shape=(units, input_dim),
            initializer=initializers.RandomNormal(),
            regularizer=TropicalRegularizer(lam=0.01),
            trainable=True
        )
        self.units = units
        self.input_dim = input_dim

    def call(self, inputs):
        '''
        Performs the forward pass of the TropEmbedTop2 layer.

        Parameters
        ----------
        inputs : tensorflow tensor object
            Input tensor.

        Returns
        -------
        output : tensorflow tensor object
            Output tensor after applying Tropical Embedding for top 2 values.
        '''
        input_reshaped = reshape(inputs, [-1, 1, self.input_dim])  # Reshape input data
        input_for_broadcast = repeat_elements(input_reshaped, self.units, 1)  # Repeat input for broadcasting
        values, _ = top_k(input_for_broadcast + self.w, 2)  # Calculate top 2 values
        return values[:, :, 0] - values[:, :, 1]  # Compute symmetric tropical distance
 

class TropEmbedMaxMin(Layer):
    '''
    Custom TensorFlow layer implementing Tropical Embedding for max-min distances.
    '''

    def __init__(self, units=2, initializer_w=initializers.random_normal, lam=0.0, axis_for_reduction=2, **kwargs):
        '''
        Initializes the TropEmbedMaxMin layer.

        Parameters
        ----------
        units : int, optional
            Number of output units (default is 2).
        initializer_w : initializer function, optional
            Weight initializer function (default is random_normal).
        lam : float, optional
            Regularization parameter (default is 0.0).
        axis_for_reduction : int, optional
            Axis for reduction in distance calculation (default is 2).
        **kwargs : dict
            Additional keyword arguments.
        '''
        super(TropEmbedMaxMin, self).__init__(**kwargs)
        self.units = units
        self.initializer_w = initializer_w
        self.lam = lam
        self.axis_for_reduction = axis_for_reduction

    def build(self, input_shape):
        input_dim = input_shape[-1]  # Extract the last dimension from input_shape
        self.w = self.add_weight(name='tropical_fw',
                                 shape=(self.units, input_dim),
                                 initializer=self.initializer_w,
                                 regularizer=TropRegIncreaseDistance(lam=self.lam),
                                 trainable=False) ### CHANGE BACK TO TRUE SIRRRR
        self.bias = self.add_weight(name='bias',
                                    shape=(self.units,),
                                    initializer="zeros",
                                    trainable=False) ### CHANGE BACK TO TRUE SIRRRR
        super(TropEmbedMaxMin, self).build(input_shape)

    def call(self, x):
        '''
        Performs the forward pass of the TropEmbedMaxMin layer.

        Parameters
        ----------
        x : tensorflow tensor object
            Input tensor.

        Returns
        -------
        trop_distance : tensorflow tensor object
            Output tensor after applying Tropical Embedding for max-min distances.
        '''
        x_reshaped = reshape(x, [-1, 1, self.w.shape[-1]])  # Reshape input data
        x_for_broadcast = repeat_elements(x_reshaped, self.units, 1)  # Repeat input for broadcasting
        result_addition = x_for_broadcast + self.w  # Calculate addition of input and weights
        trop_distance = reduce_max(result_addition, axis=(self.axis_for_reduction)) - reduce_min(result_addition, axis=(self.axis_for_reduction)) + self.bias  # Calculate tropical distances with bias
        return trop_distance

    def get_config(self):
        '''
        Gets the configuration of the layer.

        Returns
        -------
        config : dict
            Configuration of the layer.
        '''
        config = {
            'units': self.units,
            'initializer_w': initializers.serialize(self.initializer_w),
            'lam': self.lam
        }
        base_config = super(TropEmbedMaxMin, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        '''
        Creates a layer from its config.

        Parameters
        ----------
        config : dict
            Configuration of the layer.

        Returns
        -------
        cls : TropEmbedMaxMin object
            Instantiated TropEmbedMaxMin object with given configuration.
        '''
        return cls(**config)
    

class TropAsymmetricMax(Layer):
    '''
    Custom TensorFlow layer implementing Tropical Embedding for max-min distances.
    '''

    def __init__(self, units=2, initializer_w=initializers.random_normal, lam=0.0, axis_for_reduction=2, **kwargs):
        '''
        Initializes the TropEmbedMaxMin layer.

        Parameters
        ----------
        units : int, optional
            Number of output units (default is 2).
        initializer_w : initializer function, optional
            Weight initializer function (default is random_normal).
        lam : float, optional
            Regularization parameter (default is 0.0).
        axis_for_reduction : int, optional
            Axis for reduction in distance calculation (default is 2).
        **kwargs : dict
            Additional keyword arguments.
        '''
        super(TropAsymmetricMax, self).__init__(**kwargs)
        self.units = units
        self.initializer_w = initializer_w
        self.lam = lam
        self.axis_for_reduction = axis_for_reduction

    def build(self, input_shape):
        input_dim = input_shape[-1]  # Extract the last dimension from input_shape
        self.w = self.add_weight(name='tropical_fw',
                                 shape=(self.units, input_dim),
                                 initializer=self.initializer_w,
                                 regularizer=TropRegIncreaseDistance(lam=self.lam),
                                 trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=(self.units,),
                                    initializer="zeros",
                                    trainable=True)
        super(TropAsymmetricMax, self).build(input_shape)

    def call(self, x):
        '''
        Performs the forward pass of the TropEmbedMaxMin layer.

        Parameters
        ----------
        x : tensorflow tensor object
            Input tensor.

        Returns
        -------
        trop_distance : tensorflow tensor object
            Output tensor after applying Tropical Embedding for max-min distances.
        '''
        x_reshaped = reshape(x, [-1, 1, self.w.shape[-1]])  # Reshape input data
        x_for_broadcast = repeat_elements(x_reshaped, self.units, 1)  # Repeat input for broadcasting
        result_addition = x_for_broadcast + self.w  # Calculate addition of input and weights
        trop_distance = self.units*reduce_max(result_addition, axis=(self.axis_for_reduction)) - reduce_sum(result_addition, axis=(self.axis_for_reduction)) + self.bias  # Calculate tropical distances with bias
        return trop_distance

    def get_config(self):
        '''
        Gets the configuration of the layer.

        Returns
        -------
        config : dict
            Configuration of the layer.
        '''
        config = {
            'units': self.units,
            'initializer_w': initializers.serialize(self.initializer_w),
            'lam': self.lam
        }
        base_config = super(TropEmbedMaxMin, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        '''
        Creates a layer from its config.

        Parameters
        ----------
        config : dict
            Configuration of the layer.

        Returns
        -------
        cls : TropEmbedMaxMin object
            Instantiated TropEmbedMaxMin object with given configuration.
        '''
        return cls(**config)
    

class TropAsymmetricMin(Layer):
    '''
    Custom TensorFlow layer implementing Tropical Embedding for max-min distances.
    '''

    def __init__(self, units=2, initializer_w=initializers.random_normal, lam=0.0, axis_for_reduction=2, **kwargs):
        '''
        Initializes the TropEmbedMaxMin layer.

        Parameters
        ----------
        units : int, optional
            Number of output units (default is 2).
        initializer_w : initializer function, optional
            Weight initializer function (default is random_normal).
        lam : float, optional
            Regularization parameter (default is 0.0).
        axis_for_reduction : int, optional
            Axis for reduction in distance calculation (default is 2).
        **kwargs : dict
            Additional keyword arguments.
        '''
        super(TropAsymmetricMin, self).__init__(**kwargs)
        self.units = units
        self.initializer_w = initializer_w
        self.lam = lam
        self.axis_for_reduction = axis_for_reduction

    def build(self, input_shape):
        input_dim = input_shape[-1]  # Extract the last dimension from input_shape
        self.w = self.add_weight(name='tropical_fw',
                                 shape=(self.units, input_dim),
                                 initializer=self.initializer_w,
                                 regularizer=TropRegIncreaseDistance(lam=self.lam),
                                 trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=(self.units,),
                                    initializer="zeros",
                                    trainable=True)
        super(TropAsymmetricMin, self).build(input_shape)

    def call(self, x):
        '''
        Performs the forward pass of the TropEmbedMaxMin layer.

        Parameters
        ----------
        x : tensorflow tensor object
            Input tensor.

        Returns
        -------
        trop_distance : tensorflow tensor object
            Output tensor after applying Tropical Embedding for max-min distances.
        '''
        x_reshaped = reshape(x, [-1, 1, self.w.shape[-1]])  # Reshape input data
        x_for_broadcast = repeat_elements(x_reshaped, self.units, 1)  # Repeat input for broadcasting
        result_addition = x_for_broadcast + self.w  # Calculate addition of input and weights
        trop_distance = reduce_sum(result_addition, axis=(self.axis_for_reduction)) - self.units*reduce_min(result_addition, axis=(self.axis_for_reduction)) + self.bias  # Calculate tropical distances with bias
        return trop_distance

    def get_config(self):
        '''
        Gets the configuration of the layer.

        Returns
        -------
        config : dict
            Configuration of the layer.
        '''
        config = {
            'units': self.units,
            'initializer_w': initializers.serialize(self.initializer_w),
            'lam': self.lam
        }
        base_config = super(TropEmbedMaxMin, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        '''
        Creates a layer from its config.

        Parameters
        ----------
        config : dict
            Configuration of the layer.

        Returns
        -------
        cls : TropEmbedMaxMin object
            Instantiated TropEmbedMaxMin object with given configuration.
        '''
        return cls(**config)


class TropConv2D(Layer):
    '''
    Custom TensorFlow layer implementing Tropical Convolution 2D.
    '''

    def __init__(self, filters=64, window_size=[1, 3, 3, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1],
                 padding='VALID', initializer_w=initializers.random_normal, lam=0.0, **kwargs):
        '''
        Initializes the TropConv2D layer.

        Parameters
        ----------
        filters : int, optional
            Number of filters (default is 64).
        window_size : list, optional
            Size of the sliding window for convolution (default is [1, 3, 3, 1]).
        strides : list, optional
            Stride of the convolution (default is [1, 1, 1, 1]).
        rates : list, optional
            Rate for dilated convolution (default is [1, 1, 1, 1]).
        padding : str, optional
            Type of padding (default is 'VALID').
        initializer_w : initializer function, optional
            Weight initializer function (default is random_normal).
        lam : float, optional
            Regularization parameter (default is 0.0).
        **kwargs : dict
            Additional keyword arguments.
        '''
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
                                 regularizer=TropicalRegularizer(lam=self.lam),
                                 trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=(self.filters,),
                                    initializer="zeros",
                                    trainable=True)
        super(TropConv2D, self).build(input_shape)

    def call(self, x):
        '''
        Performs the forward pass of the TropConv2D layer.

        Parameters
        ----------
        x : tensorflow tensor object
            Input tensor.

        Returns
        -------
        trop_conv_result : tensorflow tensor object
            Output tensor after applying Tropical Convolution 2D.
        '''
        x_patches = extract_patches(images=x, sizes=self.window_size, strides=self.strides, rates=self.rates,
                                    padding=self.padding)  # Extract patches from input
        result_addition = expand_dims(x_patches, axis=-1) + self.w  # Calculate addition of patches and weights
        trop_conv_result = reduce_max(result_addition, axis=(3)) - reduce_min(result_addition, axis=(3)) + self.bias  # Compute tropical convolution
        return trop_conv_result

    def get_config(self):
        '''
        Gets the configuration of the layer.

        Returns
        -------
        config : dict
            Configuration of the layer.
        '''
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
        '''
        Creates a layer from its config.

        Parameters
        ----------
        config : dict
            Configuration of the layer.

        Returns
        -------
        cls : TropConv2D object
            Instantiated TropConv2D object with given configuration.
        '''
        return cls(**config)


class TropConv2DMax(Layer):
    '''
    Custom TensorFlow layer implementing Tropical Convolution 2D with maximum operation.
    '''

    def __init__(self, filters=64, window_size=[1, 3, 3, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1],
                 padding='VALID', initializer_w=initializers.random_normal, lam=0.0):
        '''
        Initializes the TropConv2DMax layer.

        Parameters
        ----------
        filters : int, optional
            Number of filters (default is 64).
        window_size : list, optional
            Size of the sliding window for convolution (default is [1, 3, 3, 1]).
        strides : list, optional
            Stride of the convolution (default is [1, 1, 1, 1]).
        rates : list, optional
            Rate for dilated convolution (default is [1, 1, 1, 1]).
        padding : str, optional
            Type of padding (default is 'VALID').
        initializer_w : initializer function, optional
            Weight initializer function (default is random_normal).
        lam : float, optional
            Regularization parameter (default is 0.0).
        '''
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
        '''
        Performs the forward pass of the TropConv2DMax layer.

        Parameters
        ----------
        x : tensorflow tensor object
            Input tensor.

        Returns
        -------
        trop_conv_result : tensorflow tensor object
            Output tensor after applying Tropical Convolution 2D with maximum operation.
        '''
        x_patches = extract_patches(images=x, sizes=self.window_size, strides=self.strides, rates=self.rates,
                                    padding=self.padding)  # Extract patches from input
        result_addition = expand_dims(x_patches, axis=-1) + self.w  # Calculate addition of patches and weights
        trop_conv_result = reduce_max(result_addition, axis=(3)) + self.bias  # Compute tropical convolution with maximum operation
        return trop_conv_result

    def get_config(self):
        '''
        Gets the configuration of the layer.

        Returns
        -------
        config : dict
            Configuration of the layer.
        '''
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
        '''
        Creates a layer from its config.

        Parameters
        ----------
        config : dict
            Configuration of the layer.

        Returns
        -------
        cls : TropConv2DMax object
            Instantiated TropConv2DMax object with given configuration.
        '''
        return cls(**config)


class TropEmbedMaxMinLogits(Layer):
    '''
    Custom TensorFlow layer implementing Tropical Embedding for max-min logits.
    '''

    def __init__(self, units=2, initializer_w=initializers.random_normal, lam=1.0, **kwargs):
        '''
        Initializes the TropEmbedMaxMinLogits layer.

        Parameters
        ----------
        units : int, optional
            Number of output units (default is 2).
        initializer_w : initializer function, optional
            Weight initializer function (default is random_normal).
        lam : float, optional
            Regularization parameter (default is 1.0).
        **kwargs : dict
            Additional keyword arguments.
        '''
        super(TropEmbedMaxMinLogits, self).__init__(**kwargs)
        self.units = units
        self.initializer_w = initializer_w
        self.lam = lam

    def build(self, input_shape):
        input_dim = input_shape[-1]  # Extract the last dimension from input_shape
        self.w = self.add_weight(name='tropical_fw',
                                 shape=(self.units, input_dim),
                                 initializer=self.initializer_w,
                                 regularizer=TropRegIncreaseDistance(lam=self.lam),
                                 trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=(self.units,),
                                    initializer="zeros",
                                    trainable=True)
        super(TropEmbedMaxMinLogits, self).build(input_shape)

    def call(self, x):
        '''
        Performs the forward pass of the TropEmbedMaxMinLogits layer.

        Parameters
        ----------
        x : tensorflow tensor object
            Input tensor.

        Returns
        -------
        softmin_values : tensorflow tensor object
            Output tensor after applying Tropical Embedding for max-min logits.
        '''
        x_reshaped = reshape(x, [-1, 1, self.w.shape[-1]])  # Reshape input data
        x_for_broadcast = repeat_elements(x_reshaped, self.units, 1)  # Repeat input for broadcasting
        result_addition = x_for_broadcast + self.w  # Calculate addition of input and weights
        axis_for_reduction = 2  # Define axis for reduction
        trop_distance = reduce_max(result_addition, axis=(axis_for_reduction)) - reduce_min(result_addition, axis=(axis_for_reduction)) + self.bias  # Compute tropical distance with bias
        negative_exponents = exp(-trop_distance)  # Compute negative exponents
        softmin_values = negative_exponents / reduce_sum(negative_exponents, axis=-1, keepdims=True)  # Compute softmin values
        return softmin_values

    def get_config(self):
        '''
        Gets the configuration of the layer.

        Returns
        -------
        config : dict
            Configuration of the layer.
        '''
        config = {
            'units': self.units,
            'initializer_w': initializers.serialize(self.initializer_w),
            'lam': self.lam
        }
        base_config = super(TropEmbedMaxMinLogits, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        '''
        Creates a layer from its config.

        Parameters
        ----------
        config : dict
            Configuration of the layer.

        Returns
        -------
        cls : TropEmbedMaxMinLogits object
            Instantiated TropEmbedMaxMinLogits object with given configuration.
        '''
        return cls(**config)
