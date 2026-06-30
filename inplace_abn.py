"""Lightweight inference fallback for the optional inplace_abn package.

The original DML-CSR code depends on the compiled `inplace_abn` extension.
For public inference, a BatchNorm2d-based implementation keeps the same
state-dict parameter names and avoids requiring users to build a CUDA extension.
"""

import torch.nn as nn
import torch.nn.functional as F


class InPlaceABN(nn.BatchNorm2d):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        activation="leaky_relu",
        activation_param=0.01,
        **kwargs,
    ):
        super().__init__(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
        self.activation = activation
        self.activation_param = activation_param

    def forward(self, input):
        output = super().forward(input)
        if self.activation in (None, "identity", "none"):
            return output
        if self.activation == "relu":
            return F.relu(output, inplace=True)
        if self.activation == "leaky_relu":
            return F.leaky_relu(output, negative_slope=self.activation_param, inplace=True)
        if self.activation == "elu":
            return F.elu(output, alpha=self.activation_param, inplace=True)
        return output


InPlaceABNSync = InPlaceABN
