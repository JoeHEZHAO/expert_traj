import os
import math
from os import error
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import glob
import torch.distributions.multivariate_normal as torchdist

from utils_expert import *
from metrics import *
from model_lstm import Goal_Example_Model
from helper_expert import *
import copy
from gmm2d import *
import pickle

# dataset_name = "univ"
# dataset_name = "eth"
dataset_name = "zara1"
# dataset_name = "zara2"
# dataset_name = "hotel"


class Data_Expert:
    def __init__(self, obs_traj_norm, velocity_obs, pred_traj_gt):
        self.obs_traj_norm = obs_traj_norm
        self.velocity_obs = velocity_obs
        self.pred_traj_gt = pred_traj_gt


def test(KSTEPS=20, dataset_name="eth", online_expert=True):
    global loader_test, model, log_file_curve, dset_train, dset_val
    model.eval()
    expert_dest = None
    saved_num_obj = 0

    if not online_expert:
        expert_dest = np.load(
            "./checkpoint_ethucy/test_{}_expert.npy".format(dataset_name)
        )
        print("Loading stored expert examples test_{}_expert.npy".format(dataset_name))

        # [num_of_objs, 8, 2]
        saved_num_obj = expert_dest.shape[0]
        print("total number of expert data point is {}".format(saved_num_obj))

    ade_bigls = []
    fde_bigls = []
    raw_data_dict = {}
    all_experts = []
    step = 0  # I can use this step to track the experts
    total_num_of_objs = 0
    all_goal_error = []

    for batch in loader_test:
        step += 1

        # Get data
        batch = [tensor.cuda() for tensor in batch]

        (
            obs_traj_norm,
            obs_traj,
            obs_traj_rel,
            pred_traj_gt,
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
            seq_start,
        ) = batch

        """
        Perform the experties matching here
        """
        num_of_objs = int(sum(inp_mask[0, 0]))

        if online_expert:
            data = Data_Expert(obs_traj_norm, velocity_obs, pred_traj_gt)
            end_error, rst = expert_find(
                data, num_of_objs, dset_train, dset_val, gamma=1.0
            )
            rst = torch.stack(rst)  # [num_of_objs, 2]
            rst = rst.reshape(1, 1, num_of_objs, 2).repeat(1, 8, 1, 1)
            end_error_list = [x.item() for x in end_error]
            all_goal_error.append(
                end_error_list
            )  # list of all lowest goal error of num_of_objs,

            all_experts.append(
                rst.squeeze(0).permute(1, 0, 2).data.cpu().numpy()
            )  # [num_of_objs, 8, 2]
            print(
                "Averaging end-point error is {}".format(sum(end_error) / num_of_objs)
            )

        else:
            rst = expert_dest[total_num_of_objs : num_of_objs + total_num_of_objs]
            rst = torch.from_numpy(rst).unsqueeze(0).cuda()
            rst = rst.permute(0, 2, 1, 3)
            total_num_of_objs += num_of_objs

        rst = rst.view(1, 8, num_of_objs, 2)
        obs_traj_norm[:, :, :num_of_objs] = obs_traj_norm[:, :, :num_of_objs] - (
            rst / 1.0
        )

        """
        Perform the regular inference process
        """
        V_obs_tmp = torch.cat([obs_traj_norm, velocity_obs, acc_obs], dim=-1)
        # V_obs_tmp = obs_traj_norm
        V_obs_tmp = V_obs_tmp.permute(0, 3, 1, 2)

        V_pred, _ = model(V_obs_tmp, A_obs, inp_mask, out_mask)

        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()
        num_of_objs = int(sum(inp_mask[0, 0]))

        # only evaluate on valid nodes;
        V_pred, V_tr, obs_traj, obs_traj_rel, V_obs, pred_traj_gt = (
            V_pred[:, :num_of_objs, :],
            V_tr[:, :num_of_objs, :],
            obs_traj.squeeze()[:, :num_of_objs, :],
            obs_traj_rel.squeeze()[:, :num_of_objs, :],
            V_obs.squeeze()[:, :num_of_objs, :],
            pred_traj_gt.squeeze()[:, :num_of_objs, :],
        )

        log_pis = torch.ones(V_pred[..., -2:-1].shape)
        gmm2d = GMM2D(
            log_pis,
            V_pred[..., 0:2],
            V_pred[..., 2:4],
            Func.tanh(V_pred[..., -1]).unsqueeze(-1),
        )

        # Now sample 20 samples
        ade_ls = {}
        fde_ls = {}

        V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())
        V_x_rel_to_abs = nodes_rel_to_nodes_abs(
            V_obs.data.cpu().numpy().copy(),
            V_x[0, :, :].copy(),
        )
        """For only one ped case"""
        if len(V_x_rel_to_abs.shape) < 3:
            V_x_rel_to_abs = np.expand_dims(V_x_rel_to_abs, 1)

        V_y = seq_to_nodes(pred_traj_gt.data.cpu().numpy().copy())
        V_y_rel_to_abs = nodes_rel_to_nodes_abs(
            V_tr.data.cpu().numpy().copy(),
            V_x[-1, :, :].copy(),
        )

        """For only one ped case"""
        if len(V_y_rel_to_abs.shape) < 3:
            V_y_rel_to_abs = np.expand_dims(V_y_rel_to_abs, 1)

        raw_data_dict[step] = {}
        raw_data_dict[step]["obs"] = copy.deepcopy(V_x_rel_to_abs)
        raw_data_dict[step]["trgt"] = copy.deepcopy(V_y_rel_to_abs)
        raw_data_dict[step]["pred"] = []

        for n in range(num_of_objs):
            ade_ls[n] = []
            fde_ls[n] = []

        for k in range(KSTEPS):
            V_pred = gmm2d.rsample()

            """Evaluate rel output"""
            V_pred_rel_to_abs = nodes_rel_to_nodes_abs(
                V_pred.data.cpu().numpy().copy(),
                V_x[-1, :, :].copy(),
            )

            """For only one ped case"""
            if len(V_pred_rel_to_abs.shape) < 3:
                V_pred_rel_to_abs = np.expand_dims(V_pred_rel_to_abs, 1)

            """Plug rst into the last position of V_y_rel_to_abs
            Thus, use the retrieved goal to do the evaluation;
            
            Comment out for refinement;
            """
            V_pred_rel_to_abs[-1] = rst[0, -1].cpu().numpy()

            raw_data_dict[step]["pred"].append(copy.deepcopy(V_pred_rel_to_abs))

            for n in range(num_of_objs):
                pred = []
                target = []
                obsrvs = []
                number_of = []
                pred.append(V_pred_rel_to_abs[:, n : n + 1, :])
                target.append(V_y_rel_to_abs[:, n : n + 1, :])
                obsrvs.append(V_x_rel_to_abs[:, n : n + 1, :])
                number_of.append(1)

                ade_ls[n].append(ade(pred, target, number_of))
                fde_ls[n].append(fde(pred, target, number_of))

        for n in range(num_of_objs):
            ade_bigls.append(min(ade_ls[n]))
            fde_bigls.append(min(fde_ls[n]))

    print(total_num_of_objs)
    ade_ = sum(ade_bigls) / len(ade_bigls)
    fde_ = sum(fde_bigls) / len(fde_bigls)
    return ade_, fde_, all_experts, all_goal_error


paths = "./checkpoint_ethucy/{}_best.pth".format(dataset_name)


KSTEPS = 20
grad_eff = 0.4
load_expert_local = True

print("*" * 50)
print("Number of samples:", KSTEPS)
print("*" * 50)
NUM_PED_ALL = 0.0
online_expert = False
print("State of online_expert : {}".format(online_expert))


ade_ls = []
fde_ls = []

print("*" * 50)

# Load args
# args_path = "./checkpoint_ethucy/" + "/{}_args.pkl".format(dataset_name)
# with open(args_path, "rb") as f:
#     args = pickle.load(f)

# stats = "./checkpoint_ethucy/" + "/{}_constant_metrics.pkl".format(dataset_name)
# with open(stats, "rb") as f:
#     cm = pickle.load(f)
# print("Stats:", cm)

# Data prep
obs_seq_len = 8
pred_seq_len = 12
data_set = "./datasets/" + dataset_name + "/"

dset_test = TrajectoryDataset(
    data_set + "test/",
    obs_len=obs_seq_len,
    pred_len=pred_seq_len,
    skip=1,
    norm_lap_matr=True,
    grad_eff=grad_eff,
)

if online_expert:
    if load_expert_local and os.path.exists(
        "/media/chris/hdd1/expert_traj/{}_expert_train_{}.pth".format(
            dataset_name, grad_eff
        )
    ):
        with open(
            "/media/chris/hdd1/expert_traj/{}_expert_train_{}.pth".format(
                dataset_name, grad_eff
            ),
            "rb",
        ) as f:
            dset_train = pickle.load(f)
    else:
        dset_train = TrajectoryDataset(
            data_set + "train/",
            obs_len=obs_seq_len,
            pred_len=pred_seq_len,
            skip=1,
            norm_lap_matr=True,
            grad_eff=grad_eff,
        )
else:
    dset_train = None

if online_expert:
    if load_expert_local and os.path.exists(
        "/media/chris/hdd1/expert_traj/{}_expert_val_{}.pth".format(
            dataset_name, grad_eff
        )
    ):
        with open(
            "/media/chris/hdd1/expert_traj/{}_expert_val_{}.pth".format(
                dataset_name, grad_eff
            ),
            "rb",
        ) as f:
            dset_val = pickle.load(f)
    else:
        dset_val = TrajectoryDataset(
            data_set + "val/",
            obs_len=obs_seq_len,
            pred_len=pred_seq_len,
            skip=1,
            norm_lap_matr=True,
            grad_eff=grad_eff,
        )
else:
    dset_val = None

loader_test = DataLoader(
    dset_test,
    batch_size=1,  # This is irrelative to the args batch size parameter
    shuffle=False,
    num_workers=1,
)

"""Save augmented expert to local """
save_expert_local = False

if save_expert_local:

    if not os.path.exists("./{}_expert_train_{}.pth".format(dataset_name, grad_eff)):
        with open(
            "./{}_expert_train_{}.pth".format(dataset_name, grad_eff),
            "wb",
        ) as f:
            pickle.dump(dset_train, f)

    if not os.path.exists("./{}_expert_val_{}.pth".format(dataset_name, grad_eff)):
        with open(
            "./{}_expert_val_{}.pth".format(dataset_name, grad_eff),
            "wb",
        ) as f:
            pickle.dump(dset_val, f)
        print("Saving two expert examples for dataset {}".format(dataset_name))

# Defining the model
model = Goal_Example_Model(
    n_stgcnn=1,
    n_txpcnn=5,
    input_feat=6,
    output_feat=128,
    seq_len=8,
    kernel_size=3,
    pred_seq_len=12,
).cuda()
model.eval()

# model_paths = glob.glob(exp_path)
model_paths = glob.glob(paths)

for num_avg in range(1):
    for model_path in model_paths:
        print("evaluating epoch {}".format(model_path))
        model.load_state_dict(torch.load(model_path))

        ade_ = 999999
        fde_ = 999999
        print("Testing ....")

        ad, fd, all_experts, goal_errors = test(
            20, dataset_name=dataset_name, online_expert=online_expert
        )

        if online_expert:
            with open(
                os.path.join("test_{}_expert.npy".format(dataset_name)), "wb"
            ) as f:
                np.save(f, np.concatenate(all_experts, 0))

            with open(
                os.path.join("{}_expert_goal_error.npy".format(dataset_name)),
                "wb",
            ) as f:
                np.save(f, np.concatenate(goal_errors, 0))

            data = np.concatenate(goal_errors, 0)
            print(data.mean())

        ade_ = min(ade_, ad)
        fde_ = min(fde_, fd)
        ade_ls.append(ade_)
        fde_ls.append(fde_)
    print(
        "ADE:",
        min(ade_ls),
        " FDE:",
        min(fde_ls),
    )
