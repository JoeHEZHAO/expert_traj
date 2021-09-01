import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
from torch.utils.data import DataLoader
import argparse

from social_utils import *
import yaml

from model_sdd import *
import numpy as np
import pdb
from gmm2d import *

from metrics import *
from utils import *
from soft_dtw_cuda import *
import time

parser = argparse.ArgumentParser(description="Expert_Goal_Exampls")

parser.add_argument("--num_workers", "-nw", type=int, default=0)
parser.add_argument("--gpu_index", "-gi", type=int, default=0)
parser.add_argument("--config_filename", "-cfn", type=str, default="optimal.yaml")
parser.add_argument("--verbose", "-v", action="store_true")
parser.add_argument("--lr", type=float, default=0.0003, help="learning rate")
parser.add_argument("--input_feat", type=int, default=2, help="learning rate")
parser.add_argument("--output_feat", type=int, default=128, help="learning rate")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="./checkpoint_sdd",
    help="specifying the model folder to load for testing",
)
parser.add_argument(
    "--model_name",
    type=str,
    default="sdd_best.pth*",
    help="specifying which model to use for testing",
)
parser.add_argument(
    "--eval_opt",
    type=int,
    default=1,
    help="specify ways to search: 1 for dtw; 2 for dtw + clustering",
)

args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)
device = (
    torch.device("cuda", index=args.gpu_index)
    if torch.cuda.is_available()
    else torch.device("cpu")
)

if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)
# print(device)


with open("./config/" + args.config_filename, "r") as file:
    try:
        hyper_params = yaml.load(file, Loader=yaml.FullLoader)
    except:
        hyper_params = yaml.load(file)
file.close()
# print(hyper_params)

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
model_path = os.path.join(args.checkpoint, args.model_name)
model_paths = glob.glob(model_path)
# print("Evalutating model {}".format(model_path))
model.load_state_dict(torch.load(model_paths[0]))


# origin shift as pre-processing
for traj in train_dataset.trajectory_batches:
    traj -= traj[:, :1, :]
    traj *= 0.2

for traj in test_dataset.trajectory_batches:

    traj -= traj[:, :1, :]
    traj *= 0.2


def rotate_pc(coords, alpha):
    alpha = alpha * np.pi / 180
    M = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    return M @ coords


def dist_func_cos(x, y):
    """
    Helper function to compute pair-wise cosine dissimilarity. This is meant to be used with sDTW.
    :param x: input tensor
    :param y: input tensor
    :return: output tensor, suitable for sDTW computation
    """
    # Convert to direction vectors
    x = x[:, 1:, :] - x[:, :-1, :]
    y = y[:, 1:, :] - y[:, :-1, :]
    n = x.size(1)
    m = y.size(1)
    d = x.size(2)
    x = x.unsqueeze(2).expand(-1, n, m, d)
    y = y.unsqueeze(1).expand(-1, n, m, d)

    # Convert to dissimilarity
    return 1.0 - torch.nn.CosineSimilarity(dim=3)(x, y)


def expert_find(data, data_ori, expert_set, expert_ori, angles=None):
    global args
    """
    data: [test_batch, seq, 2]
    expert_set : [train_batch, seq ,2]
    """

    all_min_end = []
    rest_diff = []
    ceriterion = SoftDTW(
        use_cuda=True,
        gamma=2.0,
        normalize=True,
    )

    mse = torch.nn.MSELoss()

    num_of_trajs = data.shape[0]
    # print("Total number of searched data {}".format(num_of_trajs))
    """Pre-process to velocity and accer"""

    gradient_eff = 0.6
    traj_v = np.gradient(np.transpose(data, (0, 2, 1)), gradient_eff, axis=-1)
    traj_a = np.gradient(traj_v, gradient_eff, axis=-1)
    traj_v = torch.from_numpy(traj_v).permute(0, 2, 1).cuda()
    traj_a = torch.from_numpy(traj_a).permute(0, 2, 1).cuda()

    # TODO: apply random rotation here
    extra_data = []
    extra_ori = []
    if angles is not None:
        for ang in angles:
            expert_copy = np.copy(expert_set)
            expert_ori_copy = np.copy(expert_ori)
            B, T, C = expert_copy.shape
            expert_copy = expert_copy.reshape(B * T, C).transpose()
            expert_ori_copy = expert_ori_copy.reshape(B * T, C).transpose()

            expert_copy = rotate_pc(expert_copy, ang).transpose()
            expert_ori_copy = rotate_pc(expert_ori_copy, ang).transpose()
            extra_data.append(expert_copy.reshape(B, T, C))
            extra_ori.append(expert_ori_copy.reshape(B, T, C))

    expert_set = np.concatenate(extra_data, 0)
    expert_ori = np.concatenate(extra_ori, 0)

    expert_traj_v = np.gradient(
        np.transpose(expert_set, (0, 2, 1)), gradient_eff, axis=-1
    )
    expert_traj_a = np.gradient(expert_traj_v, gradient_eff, axis=-1)
    expert_traj_v = torch.from_numpy(expert_traj_v).permute(0, 2, 1).cuda()
    expert_traj_a = torch.from_numpy(expert_traj_a).permute(0, 2, 1).cuda()

    expert_set = torch.from_numpy(expert_set).cuda()
    expert_ori = torch.from_numpy(expert_ori).cuda()
    data = torch.DoubleTensor(data).to(device)
    data_ori = torch.DoubleTensor(data_ori).to(device).squeeze()

    """
        For random few shot ablation study 
    """
    # random_split_ratio = 0.9
    # expert_size = expert_traj_v.shape[0]
    # print(int(expert_size * random_split_ratio))
    # indice = random.sample(range(expert_size), int(expert_size * random_split_ratio))
    # indice = torch.tensor(indice)
    # print(len(set(indice)))
    # expert_traj_v = expert_traj_v[indice]
    # expert_set = expert_set[indice]
    # print(expert_traj_v.shape)

    # t0 = time.time()
    for i in range(num_of_trajs):

        tmp_traj = traj_v[i, :8].unsqueeze(0)
        tmp_traj_abs = data[i, :8].unsqueeze(0)

        expert_num = expert_traj_v.shape[0]

        tmp_traj = tmp_traj.repeat(expert_num, 1, 1)
        tmp_traj_abs = tmp_traj_abs.repeat(expert_num, 1, 1)

        loss = ceriterion(tmp_traj, expert_traj_v[:, :8])

        if args.eval_opt == 1:
            """Opt1: for dtw matching only"""
            min_k, min_k_indices = torch.topk(loss, 20, largest=False)

        elif args.eval_opt == 2:
            """Opt2: for dtw matching + clustering matching"""
            min_k, min_k_indices = torch.topk(loss, 65, largest=False)
            retrieved_expert = expert_set[min_k_indices][:, -1]
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=20, random_state=0).fit(
                retrieved_expert.cpu().numpy()
            )

        iter_target = min_k_indices

        min_k_end = []
        end_point_appr = []

        """Back to indexing in real coords domain"""
        if args.eval_opt == 1:
            for k in iter_target:
                test_end = data[i, -1]
                exp_end = expert_set[k, -1]
                min_k_end.append(torch.norm(test_end - exp_end, 2))
                end_point_appr.append(exp_end)

            all_min_end.append(min(min_k_end))
            rest_diff.append(end_point_appr[min_k_end.index(min(min_k_end))])
        else:
            for k in kmeans.cluster_centers_:
                test_end = data[i, -1]
                exp_end = torch.from_numpy(k).cuda()

                min_k_end.append(torch.norm(test_end - exp_end, 2))
                end_point_appr.append(exp_end)

            all_min_end.append(min(min_k_end))

            rest_diff.append(end_point_appr[min_k_end.index(min(min_k_end))])

    return all_min_end, rest_diff


def test(test_dataset, train_dataset, best_of_n=20):
    global model
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

        traj_v = np.gradient(np.transpose(traj, (0, 2, 1)), 0.2, axis=-1)
        traj_a = np.gradient(traj_v, 0.2, axis=-1)
        traj_v = torch.from_numpy(traj_v).permute(0, 2, 1)
        traj_a = torch.from_numpy(traj_a).permute(0, 2, 1)

        traj_np = np.copy(traj)

        traj, mask, initial_pos, traj_a, traj_v = (
            torch.DoubleTensor(traj).to(device),
            torch.DoubleTensor(mask).to(device),
            torch.DoubleTensor(initial_pos).to(device),
            torch.DoubleTensor(traj_a).to(device),
            torch.DoubleTensor(traj_v).to(device),
        )

        expert_ori = train_dataset.trajectory_ori
        expert_ori_list = [x for x in expert_ori]
        expert_ori = np.concatenate(expert_ori_list, 0)

        expert_traj = train_dataset.trajectory_batches
        expert_traj_list = [x for x in expert_traj]
        expert_traj = np.concatenate(expert_traj_list, 0)

        angles = [0]
        end_error, rst = expert_find(
            traj_np,
            test_dataset.trajectory_ori,
            expert_traj,
            expert_ori,
            angles,
        )
        rst = torch.stack(rst)  # [num_of_objs, 2]

        """Find the goal retrieval that is too wrong, i.e. > 100 pixels, do not trust this result anymore;
        """
        end_error = torch.stack(end_error)

        """ Use estimated goals """
        input_traj = traj[:, : hyper_params["past_length"]]

        dest = rst.unsqueeze(1).reshape(traj.shape[0], 1, 2).repeat(1, 8, 1)

        """Goal-shift encoding"""
        input_traj = traj[:, : hyper_params["past_length"]] - (dest / 1.0)

        init_traj = traj[
            :, hyper_params["past_length"] - 1 : hyper_params["past_length"], :
        ]

        """GT for later evaluation """
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

            """Evaluate rel output"""

            """Adding the input initial last position"""
            V_pred = torch.cumsum(V_pred, dim=1) + init_traj.repeat(1, 12, 1)

            """Plug dest to replace the last position"""
            """Comment out if allow end-point refinement"""
            V_pred[:, -1] = dest[:, -1]

            for n in range(traj.shape[0]):
                ade_ls[n].append(torch.norm(V_pred[n] - V_tr[n], dim=-1).mean())
                fde_ls[n].append(torch.norm(V_pred[n, -1] - V_tr[n, -1]))

        # Metrics
        for n in range(traj.shape[0]):
            ade_bigls.append(min(ade_ls[n]))
            fde_bigls.append(min(fde_ls[n]))

    ade_ = sum(ade_bigls) / len(ade_bigls)
    fde_ = sum(fde_bigls) / len(fde_bigls)
    return ade_.item() * 5.0, fde_.item() * 5.0


t0 = time.time()
print(test(test_dataset, train_dataset, best_of_n=20))
