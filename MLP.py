import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.nn import functional as F
from torch.distributions.normal import Normal
import numpy as np

"""MLP model"""


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_size=(1024, 512),
        activation="relu",
        discrim=False,
        dropout=-1,
    ):
        super(MLP, self).__init__()
        dims = []
        dims.append(input_dim)
        dims.extend(hidden_size)
        dims.append(output_dim)
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "identity":
            self.activation = nn.Identity()
        elif activation == "prelu":
            self.activation = nn.PReLU()

        self.sigmoid = nn.Sigmoid() if discrim else None
        self.dropout = dropout

    def forward(self, x, mask=None):
        """
        x input has shape [batch * group_ped, 16], which includes 8 * 2 feature dim, so everything is encoded together?
        """
        # if mask is not None:
        # mask = mask[:, 0]

        for i in range(len(self.layers)):
            x = self.layers[i](x)

            if mask is not None:
                x = torch.einsum("nvc, nv->nvc", (x, mask))

            if i != len(self.layers) - 1:
                x = self.activation(x)
                if self.dropout != -1:
                    x = nn.Dropout(
                        min(0.1, self.dropout / 3) if i == 1 else self.dropout
                    )(x)
            elif self.sigmoid:
                x = self.sigmoid(x)
        return x
