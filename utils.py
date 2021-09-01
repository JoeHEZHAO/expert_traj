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

import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from numpy import linalg as LA
import networkx as nx
from tqdm import tqdm
import time

MAX_NODE = 57


def sinkhorn(log_alpha, n_iters=5):
    n = log_alpha.shape[1]
    log_alpha = log_alpha.view(-1, n, n)
    for i in range(n_iters):
        log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True)).view(
            -1, n, 1
        )
        log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True)).view(
            -1, 1, n
        )
    return torch.exp(log_alpha)


def rotate_pc(coords, alpha):
    alpha = alpha * np.pi / 180
    M = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    return M @ coords


def torch_seq_to_nodes(seq_):
    seq_ = seq_.squeeze()
    batch = seq_.shape[0]
    seq_len = seq_.shape[1]
    num_ped = seq_.shape[2]

    V = torch.zeros((batch, seq_len, num_ped, 2), requires_grad=True).cuda()
    for s in range(seq_len):
        step_ = seq_[:, s, :, :]
        for h in range(num_ped):
            V[:, s, h, :] = step_[:, h]

    return V.squeeze()


def torch_nodes_rel_to_nodes_abs(nodes, init_node):
    """
    batch enable funct
    """

    nodes_ = torch.zeros_like(nodes, requires_grad=True).cuda()
    """
    nodes: [batch, seq_len, num_ped, feat]
    init : [batch, seq_len, num_ped, feat]
    """

    for s in range(nodes.shape[1]):
        for ped in range(nodes.shape[2]):
            nodes_[:, s, ped, :] = (
                torch.sum(nodes[:, : s + 1, ped, :], axis=1) + init_node[:, ped, :]
            )

    return nodes_.squeeze()


def anorm(p1, p2):
    NORM = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    if NORM == 0:
        return 0
    return 1 / (NORM)


def torch_anorm(p1, p2):
    NORM = torch.sqrt((p1[:, 0] - p2[:, 0]) ** 2 + (p1[:, 1] - p2[:, 1]) ** 2)
    rst = torch.where(NORM != 0.0, (1 / NORM).data, NORM.data)

    return rst


def seq_to_graph(
    seq_,
    seq_rel,
    norm_lap_matr=True,
    alloc=False,
):
    """
    Pytorch Version;
    For this function, input pytorch tensor:
        (seq_rel) has shape [num_ped, 2, seq_len]
    """
    norm_lap_matr = False
    # norm_lap_matr = True

    seq_ = seq_.squeeze()
    V = seq_rel.permute(2, 0, 1)
    # seq_rel = seq_rel.squeeze()
    """ Decide if use real coords for adj computation or not """
    seq_rel = seq_.clone()

    if len(seq_.shape) < 3:
        seq_ = seq_.unsqueeze(-1)
        seq_rel = seq_rel.unsqueeze(-1)

    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]

    seq_rel = (
        seq_rel.permute(2, 0, 1).unsqueeze(-2).repeat(1, 1, max_nodes, 1)
    )  # convert to [seq_len, node, nodes, feat]
    # 1. Find relative coordinates in this way;
    # 2. Reduce centroid position information?
    # 3. Trans_seq_rel = seq_rel.permute(0, 2, 1, 3) - center
    trans_seq_rel = seq_rel.permute(0, 2, 1, 3)

    # calc relative
    seq_rel_r = seq_rel - trans_seq_rel
    seq_rel_r = torch.sqrt(seq_rel_r[..., 0] ** 2 + seq_rel_r[..., 1] ** 2)

    # set threshold to the number of neighbors?
    # seq_rel_r  = torch.where(seq_rel_r > 3, torch.zeros(1), seq_rel_r)

    """ Find the inverse """
    seq_rel_r = torch.where(seq_rel_r != 0.0, (1 / seq_rel_r).data, seq_rel_r.data)

    """ How to deal with dist > 1?, which will lead to unstable? """
    """ Will this play an important factor?  """
    if seq_rel.is_cuda:
        seq_rel_r = torch.where(
            # seq_rel_r > 1.0, seq_rel_r, torch.ones(1).cuda())
            seq_rel_r > 1.0,
            torch.ones(1).cuda(),
            seq_rel_r,
        )
    else:
        seq_rel_r = torch.where(
            # seq_rel_r > 1.0, seq_rel_r, torch.ones(1))
            seq_rel_r > 1.0,
            torch.ones(1),
            seq_rel_r,
        )

    """Normalized based on the largest value in column?"""
    # max_column, _ = torch.max(seq_rel_r, -1, keepdim=True)
    # seq_rel_r /= max_column

    if seq_rel.is_cuda:
        diag_ones = torch.eye(max_nodes).cuda()
    else:
        diag_ones = torch.eye(max_nodes)
    seq_rel_r[:, :] = seq_rel_r[:, :] + diag_ones

    A = seq_rel_r

    if norm_lap_matr:
        """
        Laplacian from graph matrix, as in
        1). https://github.com/dimkastan/PyTorch-Spectral-clustering/blob/master/FiedlerVectorLaplacian.py;
        2). https://github.com/huyvd7/pytorch-deepglr/blob/master/deepglr/deepglr.py;
        3). https://github.com/huyvd7/pytorch-deepglr
        """
        A_sumed = torch.sum(A, axis=1).unsqueeze(-1)
        diag_ones_tensor = diag_ones.unsqueeze(0).repeat(seq_len, 1, 1)
        D = diag_ones_tensor * A_sumed
        DH = torch.sqrt(D)
        DH = torch.where(DH != 0, 1.0 / DH, DH)  # avoid inf values
        L = D - A
        A = torch.bmm(DH, torch.bmm(L, DH))
    # else:
    # A = torch.where(A != 0, 1.0/A, A)

    if alloc:
        # for now, adj_rel_shift only admit numpy data;
        A_alloc = adj_rel_shift(seq_rel_r.cpu().numpy())
        A_alloc = torch.from_numpy(A_alloc)

        if norm_lap_matr:
            A_sumed = torch.sum(A_alloc, axis=1).unsqueeze(-1)
            diag_ones_tensor = diag_ones.unsqueeze(0).repeat(seq_len, 1, 1)
            D = diag_ones_tensor * A_sumed
            DH = 1.0 / torch.sqrt(D)
            DH[torch.isinf(DH)] = 0.0
            L = D - A_alloc
            A_alloc = torch.bmm(DH, torch.bmm(L, DH))

        return V, A, A_alloc
    return V, A, A  # the last A acts like padding for A_alloc


def adj_rel_shift(A):
    """shift adj edges w.r.t. cloest neighbor

    # A: adj matrix, [batch, seq_len, num_ped, num_ped]
    A: adj matrix, [seq_len, num_ped, num_ped]


    Notice: the ped could be padded ?? I guess no, this has to happen in preprocessing:

    procedure:
    1). Find cloest neighbor for each nodes;
    2). Replace edges of all other neighbors, except the cloest one, with those of neighbors;
    3). See what happens; (Although, numberically difference is very small;)
    4). numpy.take_along_axis seems a answer;


    Update: use numpy instead of pytorch
    """

    seq_len, num_ped = A.shape[:2]

    # min_values, indices = torch.min(A, dim=-1)
    indices = np.argmin(A, axis=-1)
    min_values = np.min(A, axis=-1)
    # target_index = len(indices.shape)
    # tmp = A.clone()

    # replace current row with min value row, -1 works for second last axis;
    # A[:, :, :] = tmp.gather(target_index-1, indices)
    # out = tmp.gather(2, indices.unsqueeze(-1))
    # indices = indices.unsqueeze(-1).repeat(1, 1, 1, num_ped)
    # indices[:, :, :] = torch.range(
    # 0, num_ped-1, dtype=torch.int32).view(1, 1, 1, num_ped).repeat(batch, seq_len, num_ped, 1)
    # import pdb
    # pdb.set_trace()
    # out = tmp.gather(2, indices)
    indices = np.expand_dims(indices, -1)
    min_values = np.expand_dims(min_values, -1)
    out = np.take_along_axis(A, indices, 1)

    # replace diagonal with zeros, since it should be
    # idenity = torch.eye(num_ped)
    idenity = np.eye(num_ped)
    ivt_idenity = np.where(idenity != 0, 0, 1)
    # ivt_idenity = ivt_idenity.unsqueeze(
    # 0).unsqueeze(0).repeat(batch, seq_len, 1, 1)
    out = out * ivt_idenity

    # replace the anchor nodes with one:
    np.put_along_axis(out, indices, min_values, axis=-1)

    # return A
    return out


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


def read_file(_path, delim="\t"):
    data = []
    if delim == "tab":
        delim = "\t"
    elif delim == "space":
        delim = " "
    with open(_path, "r") as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(
        self,
        data_dir,
        obs_len=8,
        pred_len=8,
        skip=1,
        threshold=0.002,
        min_ped=1,
        delim="\t",
        norm_lap_matr=True,
        alloc=False,
        angles=[0],
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        global MAX_NODE

        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.norm_lap_matr = norm_lap_matr
        self.alloc = alloc

        all_files = os.listdir(self.data_dir)
        all_files = [
            os.path.join(self.data_dir, _path)
            for _path in all_files
            if _path.endswith("txt")
        ]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        seq_list_v = []
        seq_list_a = []
        seq_list_abs = []
        seq_list_norm = []

        loss_mask_list = []
        non_linear_ped = []

        # angles = np.arange(0, 360, 45) if "train" in self.data_dir else [0]
        angles = [0]  # Set angle always to [0], to cancel the augmentation;
        data_scale = 1.0

        for path in all_files:
            data_ori = read_file(path, delim)

            for angle in angles:
                # data = data_ori.copy()
                data = np.copy(data_ori) * data_scale
                # Can I perform rotation here? It seems that I can, let me give a try;
                data[:, -2:] = rotate_pc(data[:, -2:].transpose(), angle).transpose()

                # Global-wise x-mean and y-mean based on the rotated data;
                x_mean = data[:, 2].mean()
                y_mean = data[:, 3].mean()

                frames = np.unique(data[:, 0]).tolist()
                frame_data = []

                for frame in frames:
                    frame_data.append(data[frame == data[:, 0], :])
                num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))

                for idx in range(0, num_sequences * self.skip + 1, skip):

                    curr_seq_data = np.concatenate(
                        frame_data[idx : idx + self.seq_len], axis=0
                    )
                    peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                    self.max_peds_in_frame = max(
                        self.max_peds_in_frame, len(peds_in_curr_seq)
                    )
                    curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                    curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                    curr_seq_v = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                    curr_seq_a = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                    curr_loss_mask = np.zeros((len(peds_in_curr_seq), self.seq_len))
                    curr_seq_abs = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                    curr_seq_norm = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))

                    num_peds_considered = 0
                    _non_linear_ped = []

                    # Loop all pedestrians in curr sequence;
                    for _, ped_id in enumerate(peds_in_curr_seq):
                        curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]

                        curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                        pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                        pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                        if pad_end - pad_front != self.seq_len:
                            continue
                        curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])

                        # move the traj to new zero origin * 170
                        curr_ped_seq[0, :] = curr_ped_seq[0, :] - curr_ped_seq[0, 0]
                        curr_ped_seq[1, :] = curr_ped_seq[1, :] - curr_ped_seq[1, 0]
                        # curr_ped_seq *= 170

                        # keep label the abs data format
                        label_ped_seq = np.copy(curr_ped_seq)

                        # Make coordinates relative
                        rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                        # Make coordinates velocity
                        v_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                        # Make coorindates acceleration
                        a_curr_ped_seq = np.zeros(curr_ped_seq.shape)

                        # How about use some simple direct information?
                        rel_curr_ped_seq[:, 1:] = (
                            curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                        )
                        # rel_curr_ped_seq = curr_ped_seq

                        v_curr_ped_seq = np.gradient(
                            np.array(curr_ped_seq), 0.4, axis=1
                        )
                        a_curr_ped_seq = np.gradient(
                            np.array(v_curr_ped_seq), 0.4, axis=1
                        )
                        # v_curr_ped_seq = np.gradient(
                        # np.array(curr_ped_seq), 0.4, axis=1
                        # )
                        # a_curr_ped_seq = np.gradient(
                        # np.array(v_curr_ped_seq), 0.4, axis=1) / 1.0

                        """
                        Mean Substraction
                        What if I do not do this?
                        """
                        # curr_ped_seq[0, :] -= x_mean
                        # curr_ped_seq[1, :] -= y_mean

                        # re-calculate the mean and use 3 as std
                        obs_len = 8
                        x_local_mean = curr_ped_seq[0, :obs_len].mean()
                        y_local_mean = curr_ped_seq[1, :obs_len].mean()
                        # x_local_mean = curr_ped_seq[0, obs_len -1]
                        # y_local_mean = curr_ped_seq[1, obs_len -1]
                        # x_max = np.max(curr_ped_seq[0, :obs_len])
                        # y_max = np.max(curr_ped_seq[1, :obs_len])
                        x_max = np.max(curr_ped_seq[0, :])
                        y_max = np.max(curr_ped_seq[1, :])

                        x_mean_all = curr_ped_seq[0, -1].mean()
                        y_mean_all = curr_ped_seq[1, -1].mean()

                        curr_ped_seq_norm = np.copy(curr_ped_seq)
                        # if "test" not in self.data_dir:
                        curr_ped_seq_norm[0, :] = (
                            curr_ped_seq_norm[0, :] - (x_mean_all) / 1.0
                        )
                        curr_ped_seq_norm[1, :] = (
                            curr_ped_seq_norm[1, :] - (y_mean_all) / 1.0
                        )
                        # curr_ped_seq_norm[0, :] /= 3.0
                        # curr_ped_seq_norm[1, :] /= 3.0
                        # curr_ped_seq_norm[0, :] = curr_ped_seq_norm[0, ::-1] / 3.0
                        # curr_ped_seq_norm[1, :] = curr_ped_seq_norm[1, ::-1] / 3.0
                        # else:
                        # curr_ped_seq_norm[0, :] = (curr_ped_seq_norm[0, :] ) / 3.0
                        # curr_ped_seq_norm[1, :] = (curr_ped_seq_norm[1, :] ) / 3.0
                        # curr_ped_seq_norm[0, :] = curr_ped_seq_norm[0, :] - (
                        # x_mean_all
                        # )
                        # curr_ped_seq_norm[1, :] = curr_ped_seq_norm[1, :] - (
                        # y_mean_all
                        # )
                        # curr_ped_seq_norm[0, :] = curr_ped_seq_norm[0, :] - (x_mean_all * 12.0)
                        # curr_ped_seq_norm[1, :] = curr_ped_seq_norm[1, :] - (y_mean_all * 12.0)
                        # curr_ped_seq_norm[0, :] += 0.5
                        # curr_ped_seq_norm[1, :] += 0.5
                        # curr_ped_seq_norm[0, :] = curr_ped_seq_norm[0, ::-1]  / 3.0
                        # curr_ped_seq_norm[1, :] = curr_ped_seq_norm[1, ::-1]  / 3.0

                        _idx = num_peds_considered
                        curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                        curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                        curr_seq_v[_idx, :, pad_front:pad_end] = v_curr_ped_seq
                        curr_seq_a[_idx, :, pad_front:pad_end] = a_curr_ped_seq
                        curr_seq_abs[_idx, :, pad_front:pad_end] = label_ped_seq
                        curr_seq_norm[_idx, :, pad_front:pad_end] = curr_ped_seq_norm

                        # Linear vs Non-Linear Trajectory
                        _non_linear_ped.append(
                            poly_fit(curr_ped_seq, pred_len, threshold)
                        )
                        curr_loss_mask[_idx, pad_front:pad_end] = 1
                        num_peds_considered += 1

                    # if num_peds_considered > min_ped:
                    min_ped = 0
                    max_ped = 1000
                    flip = False

                    if num_peds_considered > min_ped and num_peds_considered <= max_ped:
                        non_linear_ped += _non_linear_ped
                        num_peds_in_seq.append(num_peds_considered)
                        loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                        seq_list.append(curr_seq[:num_peds_considered])
                        seq_list_rel.append(curr_seq_rel[:num_peds_considered])
                        seq_list_v.append(curr_seq_v[:num_peds_considered])
                        seq_list_a.append(curr_seq_a[:num_peds_considered])
                        seq_list_abs.append(curr_seq_abs[:num_peds_considered])
                        seq_list_norm.append(curr_seq_norm[:num_peds_considered])

                        if flip and "train" in self.data_dir:
                            non_linear_ped += _non_linear_ped
                            num_peds_in_seq.append(num_peds_considered)
                            loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                            seq_list.append(np.flip(curr_seq[:num_peds_considered], 2))
                            seq_list_rel.append(
                                np.flip(curr_seq_rel[:num_peds_considered], 2)
                            )
                            seq_list_v.append(
                                np.flip(curr_seq_v[:num_peds_considered], 2)
                            )
                            seq_list_a.append(
                                np.flip(curr_seq_a[:num_peds_considered], 2)
                            )
                            seq_list_abs.append(
                                np.flip(curr_seq_abs[:num_peds_considered], 2)
                            )
                            seq_list_norm.append(
                                np.flip(curr_seq_norm[:num_peds_considered], 2)
                            )

        # For vanilla eth dataset, there are 2785 dataset;
        # For angle augmented eth dataset, there are 24 times 2785 = 66840;
        self.num_seq = len(seq_list)
        seq_list_norm = np.concatenate(seq_list_norm, axis=0)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        seq_list_v = np.concatenate(seq_list_v, axis=0)
        seq_list_a = np.concatenate(seq_list_a, axis=0)
        seq_list_abs = np.concatenate(seq_list_abs, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.obs_traj_norm = torch.from_numpy(seq_list_norm[:, :, : self.obs_len]).type(
            torch.float
        )
        # Reverse the input sequence
        # self.obs_traj_norm = torch.flip(self.obs_traj_norm, [2])

        self.obs_traj = torch.from_numpy(seq_list[:, :, : self.obs_len]).type(
            torch.float
        )

        self.pred_traj = torch.from_numpy(
            # seq_list_abs[:, :, self.obs_len:]).type(torch.float)
            seq_list[:, :, self.obs_len :]
        ).type(torch.float)

        self.obs_traj_rel = torch.from_numpy(seq_list_rel[:, :, : self.obs_len]).type(
            torch.float
        )
        self.pred_traj_rel = torch.from_numpy(seq_list_rel[:, :, self.obs_len :]).type(
            torch.float
        )

        self.obs_traj_v = torch.from_numpy(seq_list_v[:, :, : self.obs_len]).type(
            torch.float
        )
        self.pred_traj_v = torch.from_numpy(seq_list_v[:, :, self.obs_len :]).type(
            torch.float
        )
        self.obs_traj_a = torch.from_numpy(seq_list_a[:, :, : self.obs_len]).type(
            torch.float
        )
        self.pred_traj_a = torch.from_numpy(seq_list_a[:, :, self.obs_len :]).type(
            torch.float
        )

        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]

        # Convert to Graphs
        self.v_obs = []
        self.A_obs = []
        self.v_pred = []
        self.A_pred = []

        self.velocity_obs = []
        self.velocity_pred = []
        self.acceleration_obs = []
        self.acceleration_pred = []

        if self.alloc:
            self.A_obs_alloc = []
            self.A_pred_alloc = []

        self.mask_in = []
        self.mask_out = []
        print("Processing Data .....")
        pbar = tqdm(total=len(self.seq_start_end))
        for ss in range(len(self.seq_start_end)):
            pbar.update(1)

            start, end = self.seq_start_end[ss]
            v_, a_, a_alloc_ = seq_to_graph(
                self.obs_traj[start:end, :],
                self.obs_traj_rel[start:end, :],
                self.norm_lap_matr,
                alloc=self.alloc,
            )

            """ fill-in to dummpy node form """
            dummpy_v = torch.zeros(v_.shape[0], MAX_NODE, v_.shape[2])
            dummpy_a = torch.zeros(a_.shape[0], MAX_NODE, MAX_NODE)
            dummpy_v[:, : v_.shape[1], :] = v_
            dummpy_a[:, : a_.shape[1], : a_.shape[2]] = a_

            self.v_obs.append(dummpy_v.clone())
            self.A_obs.append(dummpy_a.clone())

            if self.alloc:
                dummpy_a_alloc = torch.zeros(a_.shape[0], MAX_NODE, MAX_NODE)
                dummpy_a_alloc[:, : a_alloc_.shape[1], : a_alloc_.shape[2]] = a_alloc_
                self.A_obs_alloc.append(dummpy_a_alloc.clone())

            v_, a_, a_alloc_ = seq_to_graph(
                self.pred_traj[start:end, :],
                self.pred_traj_rel[start:end, :],
                self.norm_lap_matr,
            )

            """ fill-in to dummpy node form """
            dummpy_v = torch.zeros(v_.shape[0], MAX_NODE, v_.shape[2])
            dummpy_a = torch.zeros(a_.shape[0], MAX_NODE, MAX_NODE)
            dummpy_v[:, : v_.shape[1], :] = v_
            dummpy_a[:, : a_.shape[1], : a_.shape[2]] = a_

            self.v_pred.append(dummpy_v.clone())
            self.A_pred.append(dummpy_a.clone())
            if self.alloc:
                dummpy_a_alloc = torch.zeros(a_.shape[0], MAX_NODE, MAX_NODE)
                dummpy_a_alloc[:, : a_alloc_.shape[1], : a_alloc_.shape[2]] = a_alloc_
                self.A_pred_alloc.append(dummpy_a_alloc.clone())

            # Processing velocity and accerlation
            dummy_velocity = torch.zeros(8, MAX_NODE, v_.shape[2])
            dummy_velocity[:, : v_.shape[1], :] = self.obs_traj_v[start:end, :].permute(
                2, 0, 1
            )
            self.velocity_obs.append(dummy_velocity.clone())
            dummy_acceleration = torch.zeros(8, MAX_NODE, v_.shape[2])
            dummy_acceleration[:, : v_.shape[1], :] = self.obs_traj_a[
                start:end, :
            ].permute(2, 0, 1)
            self.acceleration_obs.append(dummy_acceleration.clone())

            dummy_velocity = torch.zeros(12, MAX_NODE, v_.shape[2])
            dummy_velocity[:, : v_.shape[1], :] = self.pred_traj_v[
                start:end, :
            ].permute(2, 0, 1)
            self.velocity_pred.append(dummy_velocity.clone())
            dummy_acceleration = torch.zeros(12, MAX_NODE, v_.shape[2])
            dummy_acceleration[:, : v_.shape[1], :] = self.pred_traj_a[
                start:end, :
            ].permute(2, 0, 1)
            self.acceleration_pred.append(dummy_acceleration.clone())

        pbar.close()

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        obs_traj_norm = torch.zeros(8, MAX_NODE, 2)
        obs_traj = torch.zeros(8, MAX_NODE, 2)
        obs_traj_rel = torch.zeros(8, MAX_NODE, 2)
        pred_traj = torch.zeros(12, MAX_NODE, 2)
        pred_traj_rel = torch.zeros(12, MAX_NODE, 2)
        # non_linear_ped = torch.zeros(12, MAX_NODE, 2)

        """
        generate mask for later process
        """
        input_mask = torch.zeros(8, MAX_NODE)
        output_mask = torch.zeros(12, MAX_NODE)

        input_mask[:, : self.obs_traj[start:end, :].shape[0]] = 1.0
        output_mask[:, : self.obs_traj[start:end, :].shape[0]] = 1.0

        obs_traj_norm[
            :, : self.obs_traj[start:end, :].shape[0], :
        ] = self.obs_traj_norm[start:end, :].permute(2, 0, 1)

        obs_traj[:, : self.obs_traj[start:end, :].shape[0], :] = self.obs_traj[
            start:end, :
        ].permute(2, 0, 1)

        obs_traj_rel[
            :, : self.obs_traj_rel[start:end, :].shape[0], :
        ] = self.obs_traj_rel[start:end, :].permute(2, 0, 1)

        pred_traj[:, : self.pred_traj[start:end, :].shape[0], :] = self.pred_traj[
            start:end, :
        ].permute(2, 0, 1)

        pred_traj_rel[
            :, : self.pred_traj_rel[start:end, :].shape[0], :
        ] = self.pred_traj_rel[start:end, :].permute(2, 0, 1)

        # non_linear_ped[:, :self.non_linear_ped[start:end, :].shape[0],
        #                :] = self.non_linear_ped[start:end, :].permute(2, 0, 1)

        if self.alloc:
            out = [
                obs_traj,
                obs_traj_rel,
                pred_traj,
                pred_traj_rel,
                self.v_obs[index],
                self.A_obs[index],
                self.v_pred[index],
                self.A_pred[index],
                self.A_obs_alloc[index],
                self.A_pred_alloc[index],
                input_mask,
                output_mask,
            ]

            return out

        out = [
            obs_traj_norm,
            obs_traj,
            obs_traj_rel,
            pred_traj,
            pred_traj_rel,
            self.v_obs[index],
            self.A_obs[index],
            self.v_pred[index],
            self.A_pred[index],
            input_mask,
            output_mask,
            self.velocity_obs[index],
            self.velocity_pred[index],
            self.acceleration_obs[index],
            self.acceleration_pred[index],
        ]

        # out = [
        #     self.obs_traj[start:end, :], self.pred_traj[start:end, :],
        #     self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
        #     self.non_linear_ped[start:end], self.loss_mask[start:end, :],
        #     self.v_obs[index], self.A_obs[index],
        #     self.v_pred[index], self.A_pred[index]
        # ]
        return out


if __name__ == "__main__":

    # test_inp = torch.randn(8, 5, 2)
    # V, A = torch_batch_seq_to_graph(test_inp, test_inp)

    """
    Compare two methods, 10.29.2020:
        1). results are same as shown as below;
    """
    # test_inp = np.random.randn(5, 2, 8)
    # test_tensor_inp = torch.from_numpy(test_inp)
    # V, A = seq_to_graph_2(test_tensor_inp, test_tensor_inp)
    # V2, A2 = seq_to_graph(test_inp, test_inp)
    # print(A[0])
    # print(A2[0])
    # print(V[0])
    # print(V2[0])
    """
    tensor([[ 0.7538, -0.1823, -0.2660, -0.1170, -0.1368],
            [-0.1823,  0.8205, -0.3826, -0.1230, -0.2373],
            [-0.2660, -0.3826,  0.8232, -0.1284, -0.1591],
            [-0.1170, -0.1230, -0.1284,  0.6347, -0.1136],
            [-0.1368, -0.2373, -0.1591, -0.1136,  0.7294]], dtype=torch.float64)
    tensor([[ 0.7538, -0.1823, -0.2660, -0.1170, -0.1368],
            [-0.1823,  0.8205, -0.3826, -0.1230, -0.2373],
            [-0.2660, -0.3826,  0.8232, -0.1284, -0.1591],
            [-0.1170, -0.1230, -0.1284,  0.6347, -0.1136],
            [-0.1368, -0.2373, -0.1591, -0.1136,  0.7294]])
    """

    """
    Test func adj_rel_shift， 11.01.2020:
    """
    # test_inp = torch.randn(1, 1, 5, 5)
    # test_inp = np.random.randn(8, 5, 5)
    # test_out = adj_rel_shift(test_inp)
    # import pdb
    # pdb.set_trace()

    """
    Test 3: test the dataloader
    """
    # data_set = os.path.join("./datasets", "univ", "val")
    data_set = os.path.join("./datasets", "stanford_synthetic", "train")
    dset_train = TrajectoryDataset(
        data_set, obs_len=8, pred_len=12, skip=1, norm_lap_matr=True, alloc=False
    )

    loader_train = DataLoader(
        dset_train,
        batch_size=3,  # This is irrelative to the args batch size parameter
        shuffle=True,
        num_workers=0,
    )

    for cnt, batch in enumerate(loader_train):

        batch = [tensor.cuda() for tensor in batch]

        # obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, \
        # V_obs, A_obs, V_tr, A_tr, A_obs_alloc, A_pred_alloc, inp_mask, out_mask = batch

        # obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, \
        #     V_obs, A_obs, V_tr, A_tr, A_obs_alloc, A_pred_alloc, \
        #     inp_mask, out_mask, velocity_obs, velocity_pred, \
        #     acc_obs, acc_pred = batch

        (
            obs_traj_norm,
            obs_traj,
            pred_traj_gt,
            obs_traj_rel,
            pred_traj_gt_rel,
            V_obs,
            A_obs,
            V_tr,
            A_tr,
            inp_mask,
            out_mask,
            velocity_obs,
            velocity_pred,
            acc_obs,
            acc_pred,
        ) = batch

    #     # print(out[-1].shape)
    #     # print(out[-1])
    #     print(out_mask.shape)
    #     print(obs_traj.shape)
    #     print(A_tr.shape)
    #     print(V_tr.shape)
