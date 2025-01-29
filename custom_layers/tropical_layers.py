# tropical_layers.py
# Description: This file contains a collection of class object for customized tropical layers to be used in Tensorflow neural networks.
# Author: Kurt Pasque
# Initial Build Date: November 22, 2023
# Last Update: June 19, 2024

'''
Module: tropical_layers.py

This file contains a collection of class object for customized tropical layers to be used in Tensorflow neural networks.

Classes:
- ChangeSignLayer : Takes flat inputs and multiplies by -1 and adds 50
- TropEmbed : Custom PyTorch layer implementing fully connected Tropical Embedding Layer.
'''

# PyTorch imports
import torch
import torch.nn as nn

class ChangeSignLayer(nn.Module):
    '''
    Custom PyTorch layer to change the sign of the input tensor.
    '''

    def __init__(self, 
                 constant_to_add = 50.0,
                 multiplier = -1.0,
                 **kwargs):
        '''
        Initializes the ChangeSignLayer.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments for the Layer superclass.
        '''
        super(ChangeSignLayer, self).__init__(**kwargs)  # Initialize the Layer superclass
        self.constand_to_add = constant_to_add
        self.multiplier = multiplier

    # def call(self, inputs):
    def forward(self, inputs):
        '''
        Performs the forward pass of the layer.

        Parameters
        ----------
        inputs : PyTorch tensor object
            Input tensor to change the sign.

        Returns
        -------
        output : PyTorch tensor object
            Output tensor with signs changed.
        '''
        # return add(constant(self.constand_to_add), scalar_mul(self.multiplier, inputs))
        return self.constand_to_add + self.multiplier * inputs  # Change the sign of the input tensor (multiply by -1)


# Converted to PyTorch by Kurt
class TropEmbed(nn.Module):
    '''
    Custom PyTorch module implementing Tropical Embedding with various distance metrics.
    This class converted to PyTorch by Kurt Pasque.
    '''

    def __init__(self, 
                 in_features,
                 out_features=256,
                 initializer_w=None,
                 lam=0.0, 
                 axis_for_reduction=2, 
                 distance_metric="sym",
                 **kwargs):
        """
        Initializes the TropEmbed layer.

        Parameters
        ----------
        in_features : int
            Size of each input sample.
        out_features : int, optional
            Number of output features (default is 2).
        initializer_w : callable, optional
            Weight initializer function (default is None, which uses nn.init.normal_).
        lam : float, optional
            Regularization parameter (default is 0.0).
        axis_for_reduction : int, optional
            Axis for reduction in distance calculation (default is 2).
        distance_metric : str, optional
            Distance metric to employ. Options are "sym", "asym_max", "asym_min" (default is "sym").
        """
        super(TropEmbed, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lam = lam
        self.axis_for_reduction = axis_for_reduction
        self.distance_metric = distance_metric

        if self.distance_metric == "sym":
            self._distance_function = self._symmetric_distance
        elif self.distance_metric == "asym_max":
            self._distance_function = self._asymmetric_max_distance
        elif self.distance_metric == "asym_min":
            self._distance_function = self._asymmetric_min_distance
        else:
            raise ValueError(
                f"{self.distance_metric} unsupported. Distance metric must be 'sym', 'asym_max', or 'asym_min'."
            )

        # Initialize weights and biases
        # self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight = nn.Parameter(torch.empty((in_features, out_features)))
        print(f'{self.weight.shape = }')
        # self.bias = nn.Parameter(torch.Tensor(out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.reset_parameters(initializer_w)

    def reset_parameters(self, initializer_w):
        """
        Initializes the weights and biases of the layer.

        Parameters
        ----------
        initializer_w : callable
            Weight initializer function.
        """
        torch.manual_seed(0)
        if initializer_w is None:
            nn.init.normal_(self.weight, mean=0.0, std=2.0)
        else:
            initializer_w(self.weight)


    def _symmetric_distance(self, result_addition):
        """
        Computes symmetric tropical distance.

        Parameters
        ----------
        result_addition : torch.Tensor
            Tensor after addition of input and weights.

        Returns
        -------
        torch.Tensor
            Computed tropical distance.
        """
        max_vals, _ = torch.max(result_addition, dim=self.axis_for_reduction)
        min_vals, _ = torch.min(result_addition, dim=self.axis_for_reduction)
        trop_distance = max_vals - min_vals #+ self.bias
        return trop_distance

    def _asymmetric_max_distance(self, result_addition):
        """
        Computes asymmetric max tropical distance.

        Parameters
        ----------
        result_addition : torch.Tensor
            Tensor after addition of input and weights.

        Returns
        -------
        torch.Tensor
            Computed tropical distance.
        """
        max_vals, _ = torch.max(result_addition, dim=self.axis_for_reduction)
        sum_vals = torch.sum(result_addition, dim=self.axis_for_reduction)
        trop_distance = self.in_features * max_vals - sum_vals #+ self.bias
        return trop_distance

    def _asymmetric_min_distance(self, result_addition):
        """
        Computes asymmetric min tropical distance.

        Parameters
        ----------
        result_addition : torch.Tensor
            Tensor after addition of input and weights.

        Returns
        -------
        torch.Tensor
            Computed tropical distance.
        """
        min_vals, _ = torch.min(result_addition, dim=self.axis_for_reduction)
        sum_vals = torch.sum(result_addition, dim=self.axis_for_reduction)
        trop_distance = sum_vals - self.in_features * min_vals #+ self.bias
        return trop_distance


    def forward(self, x):
        """
        Performs the forward pass of the TropEmbed layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_features).

        Returns
        -------
        torch.Tensor
            Output tensor after applying Tropical Embedding.
        """

        # print(f'**** {x.shape = }, {x.unsqueeze(1).shape}, {self.weight.shape = }')
        return self._distance_function(x.unsqueeze(1) + self.weight)

    def extra_repr(self):
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"distance_metric='{self.distance_metric}'")

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
            # 'initializer_w': initializers.serialize(self.initializer_w),
            'initializer': str(self.initializer),
            'lam': self.lam,
            'axis_for_reduction': self.axis_for_reduction,
            'distance_metric': self.distance_metric
        }
        base_config = super(TropEmbed, self).get_config()
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
