import torch
from torch import nn
from math import sqrt
import numpy as np

class SinAct(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.sin(x)

class CustomBatchNorm1d(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.bn = nn.BatchNorm1d(d, affine=False)

    def forward(self, x):
        return self.bn(x)

import torch
import torch.nn as nn
from math import sqrt

class CustomNormalization(nn.Module):
    def __init__(self, norm_type, mean_reduction, force_factor=None):
        """
        Initializes the CustomNormalization layer.

        Args:
            norm_type (str): Type of normalization to apply. 'bn' for batch normalization,
                             'ln' for layer normalization, 'id' for identity (no normalization).
            mean_reduction (bool): If True, subtracts the mean before normalization.
            force_factor (float, optional): A custom scaling factor. If None, it's computed based on dimensions.
        """
        super().__init__()
        self.mean_reduction = mean_reduction
        self.norm_type = norm_type
        self.force_factor = force_factor

        # Determines the dimension along which normalization is applied.
        if norm_type == 'bn':
            self.dim = 0  # Normalize across the batch size (columns).
        elif norm_type == 'ln':
            self.dim = 1  # Normalize across the feature dimension (rows).
        elif norm_type == 'id':
            self.dim = -1  # No normalization.
        else:
            raise ValueError("No such normalization.")

    def forward(self, X):
        """
        Applies the normalization to the input tensor.

        Args:
            X (Tensor): The input tensor to normalize.

        Returns:
            Tensor: The normalized tensor.
        """
        # If 'id', return the input as is (no normalization).
        if self.dim == -1:
            return X
        
        # If mean_reduction is True, subtracts the mean from the tensor along the specified dimension.
        if self.mean_reduction:
            X = X - X.mean(dim=self.dim, keepdim=True)

        # Computes the norm of the tensor along the specified dimension.
        norm = X.norm(dim=self.dim, keepdim=True)

        # Determines the scaling factor: the square root of the dimension size.
        # For batch normalization ('bn'), it's the batch size (n).
        # For layer normalization ('ln'), it's the feature dimension size (d).
        factor = sqrt(X.shape[self.dim])

        # If a custom force_factor is provided, it overrides the computed factor.
        if self.force_factor is not None:
            factor = self.force_factor

        # Normalizes the tensor by dividing each element by (norm / factor).
        X = X / (norm / factor)
        return X


class GainedActivation(nn.Module):
    def __init__(self, activation, gain):
        super().__init__()
        self.activation = activation()
        self.gain = nn.Parameter(torch.tensor([gain], requires_grad=True))

    def forward(self, x):
        return self.activation(self.gain * x)



###############################


class MLPWithBatchNorm(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, hidden_dim, norm_type, mean_reduction, activation, save_hidden, exponent, order='norm_act', force_factor=None, bias=False):
        """
        Initializes the MLPWithBatchNorm class.

        Args:
            input_dim (int): Dimension of the input features.
            output_dim (int): Dimension of the output.
            num_layers (int): Number of layers in the MLP.
            hidden_dim (int): Dimension of the hidden layers.
            norm_type (str): Type of normalization ('torch_bn' for PyTorch BatchNorm1d or other types for custom normalization).
            mean_reduction (bool): If True, normalization includes mean reduction.
            activation (callable): Activation function to be used in the network.
            save_hidden (bool): If True, saves the output of each layer.
            exponent (float): Exponent factor for layer gain adjustment.
            order (str): The order of applying normalization and activation. Either 'act_norm' or 'norm_act'.
            force_factor (float, optional): Force factor for custom normalization.
            bias (bool): If True, adds a learnable bias to the layers.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.hiddens = {}  # Dictionary to store outputs of each layer if save_hidden is True.
        self.initialized = False  # Flag to check if the model's parameters have been initialized.
        self.exponent = exponent
        self.save_hidden = save_hidden
        self.order = order

        # Validate the order of normalization and activation.
        if self.order not in ['act_norm', 'norm_act']:
            raise ValueError("Unknown order")

        # Initializing the layers of the MLP.
        self.layers = nn.ModuleDict()

        # Input layer
        self.layers[f'fc_0'] = nn.Linear(input_dim, hidden_dim, bias=bias)
        # Normalization layer
        if norm_type == 'torch_bn':
            self.layers[f'norm_0'] = nn.BatchNorm1d(hidden_dim)
        else:
            self.layers[f'norm_0'] = CustomNormalization(norm_type, mean_reduction, force_factor=force_factor)
        # Activation layer
        self.layers[f'act_0'] = activation()

        # Hidden layers
        for l in range(1, num_layers):
            self.layers[f'fc_{l}'] = nn.Linear(hidden_dim, hidden_dim, bias=bias)
            if norm_type == 'torch_bn':
                self.layers[f'norm_{l}'] = nn.BatchNorm1d(hidden_dim)
            else:
                self.layers[f'norm_{l}'] = CustomNormalization(norm_type, mean_reduction, force_factor=force_factor)
            self.layers[f'act_{l}'] = activation()

        # Output layer
        self.layers[f'fc_{num_layers}'] = nn.Linear(hidden_dim, output_dim, bias=bias)

    def forward(self, x):
        """
        Forward pass of the MLP.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after passing through the MLP.
        """
        assert self.initialized, "Model parameters not initialized."

        # Flatten the input tensor if necessary.
        x = x.view(-1, self.input_dim)

        # Pass input through each layer.
        for l in range(self.num_layers):
            # Calculate layer gain based on the exponent.
            layer_gain = ((l + 1) ** self.exponent)
            
            # Apply linear transformation.
            x = self.layers[f'fc_{l}'](x)
            if self.save_hidden:
                self.hiddens[f'fc_{l}'] = x.clone().detach()

            # Apply normalization and activation in the specified order.
            if self.order == 'norm_act':
                x = self.layers[f'norm_{l}'](x)
                if self.save_hidden:
                    self.hiddens[f'norm_{l}'] = x.clone().detach()
                x = self.layers[f'act_{l}'](x * layer_gain)
                if self.save_hidden:
                    self.hiddens[f'act_{l}'] = x.clone().detach()
            elif self.order == 'act_norm':
                x = self.layers[f'act_{l}'](x * layer_gain)
                if self.save_hidden:
                    self.hiddens[f'act_{l}'] = x.clone().detach()
                x = self.layers[f'norm_{l}'](x)
                if self.save_hidden:
                    self.hiddens[f'norm_{l}'] = x.clone().detach()

        # Final layer to produce output.
        x = self.layers[f'fc_{self.num_layers}'](x)
        if self.save_hidden:
            self.hiddens[f'fc_{self.num_layers}'] = x.clone().detach()
        return x

    def set_save_hidden(self, state):
        """
        Enables or disables saving of hidden layer outputs.

        Args:
            state (bool): If True, enables saving hidden layer outputs.
        """
        self.save_hidden = state
        if state:
            self.hiddens.clear()

    def reset_parameters(self, init_type, gain=1.0):
        """
        Resets the parameters of the network according to the specified initialization type.

        Args:
            init_type (str): Type of initialization ('xavier_normal' or 'orthogonal').
            gain (float): Gain factor for initialization.
        """
        for name, p in self.named_modules():
            if isinstance(p, nn.Linear):
                # Xavier normal initialization.
                if init_type == 'xavier_normal':
                    nn.init.xavier_normal_(p.weight, gain=gain)
                # Orthogonal initialization.
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(p.weight)
                else:
                    raise ValueError("No such initialization scheme.")
        self.initialized = True

