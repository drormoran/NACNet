import torch
from torch import nn
from utils import TensorSet


class FeaturesHeads(nn.Module):
    def __init__(self, d_in, d_out, n_layers, beta):
        super().__init__()
        self.mlp = MLP(d_in, d_out, n_layers, nn.Softplus, beta=beta)

    def forward(self, x):
        return self.mlp(x)


class NoiseHead(nn.Module):
    def __init__(self, d_in, d_out, noise_norm_function, n_layers, beta):
        super().__init__()
        self.mlp = MLP(d_in, d_out, n_layers, nn.LeakyReLU, negative_slope=0.1)
        self.noise_norm_function = noise_norm_function

    def forward(self, x):
        pred_noise = self.mlp(x)
        pred_noise = pred_noise.apply_func(self.noise_norm_function)
        return pred_noise


class ClassificationHead(nn.Module):
    def __init__(self, d_in, n_layers, beta):
        super().__init__()
        self.mlp = MLP(d_in, 2, n_layers, nn.Softplus, beta=beta)

    def forward(self, x):
        logits = self.mlp(x)
        pred_outliers = (logits[:, :, 0]).apply_layer(torch.sigmoid)
        return pred_outliers, logits


class RegressionHead(nn.Module):
    def __init__(self, d_in, d_out, n_layers, beta):
        super().__init__()
        self.mlp = MLP(d_in, d_out, n_layers, nn.Softplus, beta=beta)

    def forward(self, x):
        return self.mlp(x)


class MLP(nn.Module):
    def __init__(self, d_in, d_out, n_layers, activation, **activation_args):
        super().__init__()
        self.mlp_layers = nn.ModuleList()
        for i in range(n_layers):
            if i == n_layers - 1:
                self.mlp_layers.append(nn.Linear(d_in, d_out))
            else:
                self.mlp_layers.append(nn.Linear(d_in, d_in))
                self.mlp_layers.append(activation(**activation_args))

    def forward(self, x):
        if isinstance(x, TensorSet.TensorSet):
            for layer in self.mlp_layers:
                x = x.apply_layer(layer)
        else:
            for layer in self.mlp_layers:
                x = layer(x)
        return x


def get_binary_classification(pred_outliers):
    return pred_outliers.apply_func(torch.round)

