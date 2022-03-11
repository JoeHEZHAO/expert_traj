import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
from torch.utils.data import DataLoader
import argparse

from social_utils import *
import yaml

from model_sdd import Goal_example_model
import numpy as np
import pdb
from gmm2d import *

from metrics import *
from utils import *

parser = argparse.ArgumentParser(description="GoalExample")

parser.add_argument("--num_workers", "-nw", type=int, default=0)
parser.add_argument("--gpu_index", "-gi", type=int, default=0)
parser.add_argument("--config_filename", "-cfn", type=str, default="optimal.yaml")
parser.add_argument("--save_file", "-sf", type=str, default="PECNET_social_model.pt")
parser.add_argument("--verbose", "-v", action="store_true")
parser.add_argument("--lr", type=float, default=0.0003, help="learning rate")
parser.add_argument("--input_feat", type=int, default=2, help="learning rate")
parser.add_argument("--output_feat", type=int, default=128, help="learning rate")
parser.add_argument(
    "--checkpoint", type=str, default="./checkpoint_sdd_abs2", help="learning rate"
)

args = parser.parse_args()
args.checkpoint = "./sdd_wo_goal"

dtype = torch.float64

torch.set_default_dtype(dtype)
device = (
    torch.device("cuda", index=args.gpu_index)
    if torch.cuda.is_available()
    else torch.device("cpu")
)

if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)
print(device)


def batch_bivariate_loss_ssd(V_pred, V_trgt):
    """
    V_pred, V_trgt:
        [Batch, Seq_len, Nodes, 5/2];

    """
    # mux, muy, sx, sy, corr
    # assert V_pred.shape == V_trgt.shape
    normx = V_trgt[..., 0] - V_pred[..., 0]
    normy = V_trgt[..., 1] - V_pred[..., 1]

    sx = torch.exp(V_pred[..., 2])  # sx
    sy = torch.exp(V_pred[..., 3])  # sy
    corr = torch.tanh(V_pred[..., 4])  # corr

    sxsy = sx * sy

    z = (normx / sx) ** 2 + (normy / sy) ** 2 - 2 * ((corr * normx * normy) / sxsy)
    negRho = 1 - corr ** 2

    # Numerator
    result = torch.exp(-z / (2 * negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20

    result = -torch.log(torch.clamp(result, min=epsilon))

    return result.mean()


def graph_loss(V_pred, V_target):
    return batch_bivariate_loss_ssd(V_pred, V_target)


with open("./config/" + args.config_filename, "r") as file:
    try:
        hyper_params = yaml.load(file, Loader=yaml.FullLoader)
    except:
        hyper_params = yaml.load(file)
file.close()
print(hyper_params)

train_dataset = SocialDataset(
    set_name="train",
    b_size=hyper_params["train_b_size"],
    t_tresh=hyper_params["time_thresh"],
    d_tresh=hyper_params["dist_thresh"],
    verbose=args.verbose,
)

test_dataset = SocialDataset(
    set_name="test",
    b_size=hyper_params["test_b_size"],
    t_tresh=hyper_params["time_thresh"],
    d_tresh=hyper_params["dist_thresh"],
    verbose=args.verbose,
)

model = Goal_example_model(
    input_feat=args.input_feat,
    output_feat=args.output_feat,
    config=hyper_params,
    non_local_loop=0,
).cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

"""Prepare some data for this batch of data"""
# shift origin and scale data
for traj in train_dataset.trajectory_batches:
    traj -= traj[:, :1, :]
    traj *= 0.2

for traj in test_dataset.trajectory_batches:
    traj -= traj[:, :1, :]
    traj *= 0.2


def test(test_dataset, best_of_n=20):
    global model, optim
    model.eval()
    ade_bigls = []
    fde_bigls = []

    for i, (traj, mask, initial_pos) in enumerate(
        zip(
            test_dataset.trajectory_batches,
            test_dataset.mask_batches,
            test_dataset.initial_pos_batches,
        )
    ):

        traj_v = np.gradient(np.transpose(traj, (0, 2, 1)), 0.4, axis=-1)
        traj_a = np.gradient(traj_v, 0.4, axis=-1)
        traj_v = torch.from_numpy(traj_v).permute(0, 2, 1)
        traj_a = torch.from_numpy(traj_a).permute(0, 2, 1)

        traj, mask, initial_pos, traj_a, traj_v = (
            torch.DoubleTensor(traj).to(device),
            torch.DoubleTensor(mask).to(device),
            torch.DoubleTensor(initial_pos).to(device),
            torch.DoubleTensor(traj_a).to(device),
            torch.DoubleTensor(traj_v).to(device),
        )

        """Pre-process data into relative coords"""
        # input_traj = traj[:, : hyper_params["past_length"], :]
        dest = traj[:, -1].unsqueeze(1).repeat(1, 8, 1)
        # dest = 0.0
        # dest = torch.mean(traj, 1).unsqueeze(1).repeat(1, 8, 1)
        # input_traj = torch.cat(
        #     [
        #         traj[:, : hyper_params["past_length"]] - (dest / 3.0),
        #         traj_v[:, : hyper_params["past_length"]],
        #         traj_a[:, : hyper_params["past_length"]],
        #     ],
        #     -1,
        # )
        # input_traj = traj[:, : hyper_params["past_length"]] - (dest / 2.0)
        input_traj = traj[:, : hyper_params["past_length"]] - (dest)
        # input_traj = traj[:, : hyper_params["past_length"]] - dest
        # input_traj = torch.cat([traj[:, : hyper_params["past_length"]], dest[:, :1]], 1)

        init_traj = traj[
            :, hyper_params["past_length"] - 1 : hyper_params["past_length"], :
        ]
        V_tr = traj[:, hyper_params["past_length"] :, :]

        V_pred, _ = model(input_traj, mask)
        V_pred = V_pred.squeeze()

        log_pis = torch.ones(V_pred[..., -2:-1].shape)
        gmm2d = GMM2D(
            log_pis,
            V_pred[..., 0:2],
            V_pred[..., 2:4],
            Func.tanh(V_pred[..., -1]).unsqueeze(-1),
        )

        ade_ls = {}
        fde_ls = {}
        for n in range(traj.shape[0]):
            ade_ls[n] = []
            fde_ls[n] = []

        for k in range(best_of_n):
            V_pred = gmm2d.rsample().squeeze()

            """Evaluate rel output

            Comment out for evaluating abs output
            """
            # V_pred = torch.cumsum(V_pred, dim=1) + init_traj.repeat(1, 12, 1)

            for n in range(traj.shape[0]):
                ade_ls[n].append(torch.norm(V_pred[n] - V_tr[n], dim=-1).mean())
                fde_ls[n].append(torch.norm(V_pred[n, -1] - V_tr[n, -1]))

        # Metrics
        for n in range(traj.shape[0]):
            ade_bigls.append(min(ade_ls[n]))
            fde_bigls.append(min(fde_ls[n]))

    ade_ = sum(ade_bigls) / len(ade_bigls)
    fde_ = sum(fde_bigls) / len(fde_bigls)
    return ade_, fde_


def train(train_dataset, epoch):
    global model, optim
    model.train()

    for i, (traj, mask, initial_pos) in enumerate(
        zip(
            train_dataset.trajectory_batches,
            train_dataset.mask_batches,
            train_dataset.initial_pos_batches,
        )
    ):

        optimizer.zero_grad()

        traj_v = np.gradient(np.transpose(traj, (0, 2, 1)), 0.4, axis=-1)
        traj_a = np.gradient(traj_v, 0.4, axis=-1)
        traj_v = torch.from_numpy(traj_v).permute(0, 2, 1)
        traj_a = torch.from_numpy(traj_a).permute(0, 2, 1)

        traj, mask, initial_pos, traj_v, traj_a = (
            torch.DoubleTensor(traj).to(device),
            torch.DoubleTensor(mask).to(device),
            torch.DoubleTensor(initial_pos).to(device),
            torch.DoubleTensor(traj_v).to(device),
            torch.DoubleTensor(traj_a).to(device),
        )

        """Pre-process data into relative coords"""
        # rel_traj = traj[:, 1:] - traj[:, :-1]
        # V_tr = rel_traj[:, -12:]
        V_tr = traj[:, hyper_params["past_length"] :]
        dest = traj[:, -1].unsqueeze(1).repeat(1, 8, 1)
        # dest = 0.0
        # dest = torch.mean(traj, 1).unsqueeze(1).repeat(1, 8, 1)

        # input_traj = torch.cat(
        # [
        # traj[:, : hyper_params["past_length"]] - (dest / 3.0),
        # traj_a[:, : hyper_params["past_length"]],
        # traj_v[:, : hyper_params["past_length"]],
        # ],
        # -1,
        # )

        # input_traj = traj[:, : hyper_params["past_length"]] - (dest / 2.0)
        input_traj = traj[:, : hyper_params["past_length"]] - (dest)
        # input_traj = traj[:, : hyper_params["past_length"]] - dest
        # input_traj = torch.cat([traj[:, : hyper_params["past_length"]], dest[:, :1]], 1)

        V_pred, _ = model(input_traj, mask)
        V_pred = V_pred.squeeze()

        loss = graph_loss(V_pred, V_tr)
        loss.backward()

        optimizer.step()

        # Metrics
        loss_batch = loss.item()
        print("TRAIN:", "\t Epoch:", epoch, "\t Loss:", loss_batch)


for epoch in range(450):

    train(train_dataset, epoch)

    if epoch > 20:
        ade_ = 99999
        fde_ = 99999
        ad, fd = test(test_dataset, 20)
        ade_new = min(ade_, ad)
        fde_new = min(fde_, fd)

        if ade_new < ade_ and fde_new < fde_:
            ade_ = ade_new
            fde_ = fde_new
            torch.save(
                model.state_dict(),
                os.path.join(
                    args.checkpoint,
                    "val_best_{}_{}_{}.pth".format(
                        epoch, ade_.item() * 5.0, fde_.item() * 5.0
                    ),
                ),
            )

        print("ADE:", ade_.item() * 5.0, " FDE:", fde_.item() * 5.0)
