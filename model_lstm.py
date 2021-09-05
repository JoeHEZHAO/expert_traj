"""
README: batch + loss_mask version of model
Author: He Zhao
Date: 14/10/2020 (dd/mm/yy)
"""
import os
import math
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.autograd import Variable

import torch.optim as optim
from gmm2d import *


class ConvTemporalGraphical(nn.Module):
    # Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        t_kernel_size=1,
        t_stride=1,
        t_padding=0,
        t_dilation=1,
        bias=True,
    ):
        super(ConvTemporalGraphical, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias,
        )

    def forward(self, x, A, mask=None):
        x = self.conv(x)
        if mask is not None:
            x = torch.einsum("nctv, ntv->nctv", (x, mask))

        # x = torch.einsum("nctv,ntvw->nctw", (x, A))
        return x.contiguous(), A


class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        use_mdn=False,
        stride=1,
        dropout=0,
        residual=True,
    ):
        super(st_gcn, self).__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.use_mdn = use_mdn

        self.bn1 = torch.nn.LayerNorm([8, out_channels])
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])
        self.bn0 = torch.nn.LayerNorm([8, out_channels])
        self.prelu = nn.PReLU()
        self.tcn = nn.Conv2d(
            out_channels,
            out_channels,
            (kernel_size[0], 1),
            (stride, 1),
            padding,
        )
        self.dropout = nn.Dropout(dropout, inplace=True)

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=(stride, 1),
            )
            self.bn = torch.nn.LayerNorm([8, out_channels])

        self.prelu = nn.PReLU()

    def forward(self, x, A, mask=None, if_bn=True):

        res = self.residual(x)
        res = torch.einsum("nctv, ntv->nctv", (res, mask))
        if if_bn:
            res = res.permute(0, 3, 2, 1)
            res = self.bn(res)
            res = res.permute(0, 3, 2, 1)

        x, A = self.gcn(x, A, mask)
        if if_bn:
            x = x.permute(0, 3, 2, 1)
            x = self.bn0(x)
            x = x.permute(0, 3, 2, 1)

        x = self.prelu(x)

        x = self.tcn(x)
        x = torch.einsum("nctv, ntv->nctv", (x, mask))

        if if_bn:
            x = x.permute(0, 3, 2, 1)
            x = self.bn(x)
            x = x.permute(0, 3, 2, 1)

        x = self.dropout(x) + res

        if not self.use_mdn:
            x = self.prelu(x)

        return x, A


class Goal_Example_Model(nn.Module):
    def __init__(
        self,
        n_stgcnn=1,
        n_txpcnn=1,
        input_feat=2,
        output_feat=5,
        seq_len=8,
        pred_seq_len=12,
        inter_feat=32,
        kernel_size=3,
    ):
        super(Goal_Example_Model, self).__init__()
        self.n_stgcnn = n_stgcnn
        self.n_txpcnn = n_txpcnn
        self.rnn_type = "LSTM"
        self.nlayers = 1
        self.pred_seq_len = pred_seq_len

        self.st_gcns = nn.ModuleList()
        self.st_gcns.append(st_gcn(input_feat, output_feat, (kernel_size, seq_len)))
        for j in range(1, self.n_stgcnn):
            self.st_gcns.append(
                st_gcn(output_feat, output_feat, (kernel_size, seq_len))
            )

        self.enc_lstm = torch.nn.LSTM(output_feat, output_feat)

        self.state_start = torch.nn.Linear(output_feat, 2)  # manully define as 2 dim;

        self.dec_lstm = torch.nn.LSTM(output_feat + 2, output_feat)

        self.out_mus = nn.Linear(output_feat, 2)
        self.out_sigma = nn.Linear(output_feat, 2)
        self.out_corr = nn.Linear(output_feat, 1)

        self.leakyrelu = torch.nn.LeakyReLU()

    def forward(self, v, a, mask=None, out_mask=None):

        # use the last observed as input, rather than inferring it with other nets;
        a_0 = v[:, :2, -1, :].clone()  # extract (x, y)

        for k in range(self.n_stgcnn):
            v, a = self.st_gcns[k](v, a, mask)
        v = v.permute(0, 2, 3, 1)  # [B, T, N, C]
        # v = self.enc_mlp(v)
        v = v.permute(0, 3, 1, 2)  # [B, C, T, N]
        B, C, T, N = v.shape

        # transform to shape [T, B*N, C]
        v = v.permute(2, 0, 3, 1).reshape(T, B * N, C)

        h_0, c_0 = self.init_hidden(B * N, C)

        out, (h_inp, c_inp) = self.enc_lstm(v, (h_0, c_0))

        """ Transform the state to start_action """
        # Should I use the last observed coords as a_0?
        # a_0 = self.state_start(h_inp)

        a_0 = a_0.permute(0, 2, 1)
        a_0 = a_0.reshape(1, B * N, 2)
        a_i = torch.zeros(a_0.shape)

        """ Init some stats """
        V_pred = []
        a_list = [a_0]

        """ Start Decoding Stage """
        for i in range(self.pred_seq_len):
            if i == 0:
                (h_t, c_t) = self.init_hidden(B * N, C)
                inp = torch.cat([h_inp, a_0], -1)
                _, (h_t, c_t) = self.dec_lstm(inp, (h_t, c_t))

            else:
                inp = torch.cat([h_inp, a_i], -1)
                _, (h_t, c_t) = self.dec_lstm(inp, (h_t, c_t))

            v_mus = self.out_mus(h_t)
            v_sigma = self.out_sigma(h_t)
            v_corr = self.out_corr(h_t)
            v = torch.cat([v_mus, v_sigma, v_corr], -1)
            v = v.reshape(B, N, 5)
            V_pred.append(v.clone())

            log_pis = torch.ones(v[..., -2:-1].shape)
            gmm2d = GMM2D(
                log_pis, v[..., 0:2], v[..., 2:4], torch.tanh(v[..., -1]).unsqueeze(-1)
            )

            a_i = gmm2d.rsample().squeeze()
            a_i = a_i.reshape(1, B * N, 2)
            a_list.append(a_i.clone())

        V_pred = torch.stack(V_pred, dim=1)
        a_pred = torch.stack(a_list, dim=1)

        return V_pred, a_pred

    def residual_block(self, index, x, out_mask=None):

        residual = x
        out = self.tpcnns[index](x)
        if out_mask is not None:
            out = torch.einsum("ntcv, ntv->ntcv", out, out_mask)
        out = self.prelus[index](out)
        out += residual

        return out

    def init_hidden(self, bsz, nhid):

        weight = next(self.parameters()).data

        if self.rnn_type == "LSTM":
            return (
                Variable(weight.new(self.nlayers, bsz, nhid).zero_()),
                Variable(weight.new(self.nlayers, bsz, nhid).zero_()),
            )
        else:
            return Variable(weight.new(self.nlayers, bsz, hid).zero_())


if __name__ == "__main__":

    inp = torch.randn(64, 2, 8, 10)
    inp_adj = torch.randn(64, 8, 10, 10)
    inp_maks = torch.randn(64, 8, 10)
    out_maks = torch.randn(64, 12, 10)

    model = social_stgcnn()

    out = model(inp, inp_adj, mask=inp_maks, out_mask=out_maks)
    print(out[0].shape)
