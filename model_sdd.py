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
from MLP import *


class Goal_example_model(nn.Module):
    def __init__(
        self,
        config,
        input_feat=2,
        output_feat=5,
        seq_len=8,
        pred_seq_len=12,
        non_local_loop=3,
    ):
        super(Goal_example_model, self).__init__()
        self.rnn_type = "LSTM"
        self.nlayers = 1
        self.pred_seq_len = pred_seq_len
        self.output_feat = output_feat
        self.input_feat = input_feat
        self.non_local_loop = non_local_loop

        # self.traj_enc = torch.nn.Linear(input_feat, output_feat)
        self.traj_enc = MLP(
            input_feat, output_feat, hidden_size=config["enc_past_size"]
        )

        self.enc_lstm = torch.nn.LSTM(output_feat, output_feat)

        self.dec_lstm = torch.nn.LSTM(output_feat + 2, output_feat)

        self.out_mus = MLP(output_feat, 2, hidden_size=[64, 32], activation="prelu")
        self.out_sigma = MLP(output_feat, 2, hidden_size=[64, 32], activation="prelu")
        self.out_corr = MLP(output_feat, 1, hidden_size=[64, 32], activation="prelu")

        self.non_local_theta = MLP(
            input_dim=output_feat,
            output_dim=output_feat,
            hidden_size=config["non_local_theta_size"],
        )
        self.non_local_phi = MLP(
            input_dim=output_feat,
            output_dim=output_feat,
            hidden_size=config["non_local_phi_size"],
        )
        self.non_local_g = MLP(
            input_dim=output_feat,
            output_dim=output_feat,
            hidden_size=config["non_local_g_size"],
        )

    def forward(self, v, mask):

        """
        v:      input traj that has shape [batch, seq, feat_dim]
        mask:   input mask that has shape [batch, batch]
        """
        B, T, C = v.shape

        a_0 = v[:, -1, :2].clone()

        v = self.traj_enc(v)

        # Perform LSTM encoder
        v = v.permute(1, 0, 2)  # [seq, batch, feat_dim]
        h_0, c_0 = self.init_hidden(B, self.output_feat)
        out, (h_inp, c_inp) = self.enc_lstm(v, (h_0, c_0))

        # Perform non_local_social_pooling
        for i in range(self.non_local_loop):
            h_inp = self.non_local_social_pooling(h_inp.squeeze(), mask)
            h_inp = h_inp.unsqueeze(0)

        """Init some stats"""
        a_0 = a_0.reshape(1, B, 2)
        a_i = torch.zeros(a_0.shape)
        V_pred = []
        a_list = [a_0]

        for i in range(self.pred_seq_len):
            if i == 0:
                (h_t, c_t) = self.init_hidden(B, self.output_feat)
                inp = torch.cat([h_inp, a_0], -1)
                _, (h_t, c_t) = self.dec_lstm(inp, (h_t, c_t))

            else:
                inp = torch.cat([h_inp, a_i], -1)
                _, (h_t, c_t) = self.dec_lstm(inp, (h_t, c_t))

            v_mus = self.out_mus(h_t)
            v_sigma = self.out_sigma(h_t)
            v_corr = self.out_corr(h_t)
            v = torch.cat([v_mus, v_sigma, v_corr], -1)
            v = v.reshape(B, 5)
            V_pred.append(v.clone())

            log_pis = torch.ones(v[..., -2:-1].shape)
            gmm2d = GMM2D(
                log_pis, v[..., 0:2], v[..., 2:4], torch.tanh(v[..., -1]).unsqueeze(-1)
            )

            a_i = gmm2d.rsample().squeeze()
            a_i = a_i.reshape(1, B, 2)
            a_list.append(a_i.clone())

        V_pred = torch.stack(V_pred, dim=1)
        a_pred = torch.stack(a_list, dim=1)

        return V_pred, a_pred

    def init_hidden(self, bsz, nhid):

        weight = next(self.parameters()).data

        if self.rnn_type == "LSTM":
            return (
                Variable(weight.new(self.nlayers, bsz, nhid).zero_()),
                Variable(weight.new(self.nlayers, bsz, nhid).zero_()),
            )
        else:
            return Variable(weight.new(self.nlayers, bsz, hid).zero_())

    def non_local_social_pooling(self, feat, mask):
        # N,C
        theta_x = self.non_local_theta(feat)

        # C,N
        phi_x = self.non_local_phi(feat).transpose(1, 0)

        # f_ij = (theta_i)^T(phi_j), (N,N)
        f = torch.matmul(theta_x, phi_x)

        # f_weights_i =  exp(f_ij)/(\sum_{j=1}^N exp(f_ij))
        f_weights = F.softmax(f, dim=-1)

        # setting weights of non neighbours to zero
        f_weights = f_weights * mask

        # rescaling row weights to 1
        f_weights = F.normalize(f_weights, p=1, dim=1)

        # ith row of all_pooled_f = \sum_{j=1}^N f_weights_i_j * g_row_j
        pooled_f = torch.matmul(f_weights, self.non_local_g(feat))

        return pooled_f + feat


if __name__ == "__main__":

    inp = torch.randn(64, 8, 2).cuda()
    mask = torch.randn(64, 64).cuda()

    model = social_stgcnn().cuda()

    out = model(inp, mask)
    print(out[0].shape)
